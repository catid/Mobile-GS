import torch

from .modules.lpips import LPIPS

_CRITERION_CACHE = {}


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    cache_key = (net_type, version, str(device))
    criterion = _CRITERION_CACHE.get(cache_key)
    if criterion is None:
        criterion = LPIPS(net_type, version).to(device)
        criterion.eval()
        _CRITERION_CACHE[cache_key] = criterion
    return criterion(x, y)
