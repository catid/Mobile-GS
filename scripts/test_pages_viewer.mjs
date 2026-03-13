import http from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { chromium, firefox, webkit } from "playwright";

const DOCS_ROOT = path.resolve("docs");
const OUTPUT_ROOT = path.resolve("artifacts/browser-tests");
const HOST = "127.0.0.1";
const PORT = 4173;
const PYTHON = existsSync(path.resolve(".venv/bin/python")) ? path.resolve(".venv/bin/python") : "python3";

const MIME_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".bin", "application/octet-stream"],
  [".png", "image/png"],
]);

async function serveStatic(request, response) {
  const requestUrl = new URL(request.url, `http://${HOST}:${PORT}`);
  let filePath = path.join(DOCS_ROOT, decodeURIComponent(requestUrl.pathname));
  if (filePath.endsWith(path.sep)) {
    filePath = path.join(filePath, "index.html");
  }

  try {
    const fileStats = await stat(filePath);
    if (fileStats.isDirectory()) {
      filePath = path.join(filePath, "index.html");
    }
    const payload = await readFile(filePath);
    const contentType = MIME_TYPES.get(path.extname(filePath)) ?? "application/octet-stream";
    response.writeHead(200, {
      "Content-Type": contentType,
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    response.end(payload);
  } catch (error) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
}

function startServer() {
  const server = http.createServer((request, response) => {
    serveStatic(request, response).catch((error) => {
      response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      response.end(String(error));
    });
  });
  return new Promise((resolve) => {
    server.listen(PORT, HOST, () => resolve(server));
  });
}

async function ensureDir(dirPath) {
  await import("node:fs/promises").then(({ mkdir }) => mkdir(dirPath, { recursive: true }));
}

async function startVirtualDisplay(display = ":99") {
  const xvfb = spawn("Xvfb", [display, "-screen", "0", "1440x960x24", "-ac"], {
    stdio: "ignore",
  });
  await new Promise((resolve, reject) => {
    xvfb.once("error", reject);
    setTimeout(resolve, 1500);
  });
  return {
    display,
    process: xvfb,
    async close() {
      if (xvfb.killed || xvfb.exitCode !== null) {
        return;
      }
      xvfb.kill("SIGTERM");
      await new Promise((resolve) => xvfb.once("exit", resolve));
    },
  };
}

function formatLogEntries(entries) {
  if (entries.length === 0) {
    return "none";
  }
  return entries.map((entry) => `${entry.type}: ${entry.text}`).join("\n");
}

function analyzeScreenshot(screenshotPath) {
  const analysis = spawnSync(
    PYTHON,
    [
      "-c",
      `
from PIL import Image
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
image = Image.open(path).convert("RGB")
width, height = image.size
crop = image.crop((420, 40, width - 40, height - 80))
pixels = list(crop.getdata())
total = max(len(pixels), 1)
warm = sum(1 for r, g, b in pixels if r > 120 and g > 90 and b < 140 and (r + g) > b + 80)
print(json.dumps({"warmRatio": warm / total}))
      `,
      screenshotPath,
    ],
    { encoding: "utf8" },
  );

  if (analysis.status !== 0) {
    throw new Error(`Screenshot analysis failed for ${screenshotPath}: ${analysis.stderr || analysis.stdout}`);
  }
  return JSON.parse(analysis.stdout.trim());
}

async function runBrowserTest({ name, browserType, launchOptions }) {
  const browser = await browserType.launch(launchOptions);
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const logs = [];

  page.on("console", (message) => logs.push({ type: `console:${message.type()}`, text: message.text() }));
  page.on("pageerror", (error) => logs.push({ type: "pageerror", text: error.stack ?? error.message }));
  page.on("requestfailed", (request) =>
    logs.push({ type: "requestfailed", text: `${request.url()} ${request.failure()?.errorText ?? ""}`.trim() }),
  );

  try {
    await page.goto(`http://${HOST}:${PORT}/`, { waitUntil: "networkidle" });
    await page.waitForFunction(
      () => window.__viewerInfo?.ready === true || Boolean(window.__viewerInfo?.error),
      null,
      { timeout: 30000 },
    );
    await page.waitForTimeout(1200);

    const viewerInfo = await page.evaluate(() => window.__viewerInfo);
    if (!viewerInfo?.ready) {
      const statusText = await page.locator("#status-badge").textContent();
      throw new Error(
        `${name} viewer failed: ${viewerInfo?.error ?? "unknown error"}\nstatus=${statusText}\nlogs=\n${formatLogEntries(logs)}`,
      );
    }

    const screenshotPath = path.join(OUTPUT_ROOT, `${name}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    const renderStats = analyzeScreenshot(screenshotPath);
    if (renderStats.warmRatio < 0.01) {
      throw new Error(
        `${name} screenshot validation failed: warmRatio=${renderStats.warmRatio.toFixed(3)}\nlogs=\n${formatLogEntries(logs)}`,
      );
    }
    console.log(
      `${name}: renderer=${viewerInfo.renderer} scene=${viewerInfo.scene} warmRatio=${renderStats.warmRatio.toFixed(3)} screenshot=${screenshotPath}`,
    );
  } finally {
    await browser.close();
  }
}

async function main() {
  await ensureDir(OUTPUT_ROOT);
  const server = await startServer();
  const linuxDisplay = process.platform === "linux" ? await startVirtualDisplay(":99") : null;

  try {
    await runBrowserTest({
      name: "chromium",
      browserType: chromium,
      launchOptions: {
        headless: linuxDisplay ? false : true,
        env: linuxDisplay ? { ...process.env, DISPLAY: linuxDisplay.display } : process.env,
        args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,UseSkiaRenderer,WebGPU"],
      },
    });
    await runBrowserTest({
      name: "firefox",
      browserType: firefox,
      launchOptions: {
        headless: linuxDisplay ? false : true,
        env: linuxDisplay ? { ...process.env, DISPLAY: linuxDisplay.display } : process.env,
        firefoxUserPrefs: {
          "dom.webgpu.enabled": true,
          "gfx.webgpu.force-enabled": true,
          "webgl.disabled": false,
          "webgl.enable-webgl2": true,
          "webgl.force-enabled": true,
          "layers.acceleration.force-enabled": true,
          "gfx.webrender.all": true,
        },
      },
    });
    await runBrowserTest({
      name: "webkit",
      browserType: webkit,
      launchOptions: {
        headless: true,
      },
    });
  } finally {
    if (linuxDisplay) {
      await linuxDisplay.close();
    }
    await new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
