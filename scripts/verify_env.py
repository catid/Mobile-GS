import platform
import sys

import torch
import tinycudann as tcnn
from cuml.cluster import KMeans
from simple_knn._C import distCUDA2

import diff_gaussian_rasterization
import diff_gaussian_rasterization_ms
import diff_gaussian_rasterization_msori


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"visible_gpus={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        sample = torch.randn(16, device=f"cuda:{idx}")
        print(
            f"gpu[{idx}] name={name} capability={capability[0]}.{capability[1]} sample_mean={sample.mean().item():.6f}"
        )

    distances = distCUDA2(torch.randn(1024, 3, device="cuda"))
    print(f"simple_knn_mean={distances.mean().item():.6f}")

    network = tcnn.Network(
        n_input_dims=8,
        n_output_dims=4,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 16,
            "n_hidden_layers": 1,
        },
    ).cuda()
    output = network(torch.randn(32, 8, device="cuda", dtype=torch.float16))
    print(f"tinycudann_output_shape={tuple(output.shape)}")

    print(f"cuml_kmeans={KMeans.__module__}.{KMeans.__name__}")
    print(f"diff_gaussian_rasterization={diff_gaussian_rasterization.__file__}")
    print(f"diff_gaussian_rasterization_ms={diff_gaussian_rasterization_ms.__file__}")
    print(f"diff_gaussian_rasterization_msori={diff_gaussian_rasterization_msori.__file__}")
    print("verification=ok")


if __name__ == "__main__":
    main()
