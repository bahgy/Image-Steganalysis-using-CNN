import torch
import torch.nn.functional as F


def extract_residual(images):

    # Simple high-pass filter
    kernel = torch.tensor(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ],
        dtype=images.dtype,
        device=images.device
    )

    kernel = kernel.view(1, 1, 3, 3)

    # Apply the same filter independently to each channel
    channels = images.shape[1]

    kernel = kernel.repeat(channels, 1, 1, 1)

    residual = F.conv2d(
        images,
        kernel,
        padding=1,
        groups=channels
    )

    return residual
