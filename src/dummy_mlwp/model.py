"""The dummy network, and the device handling around it.

The network is meaningless as meteorology and deliberately so. What it is *not* is a
no-op: a real convolutional forward pass runs on the selected device for every output
mode, because catching a broken CUDA setup here — rather than in the real model three
weeks later — is the main reason this application exists.

Two knobs, MODEL_HIDDEN_CHANNELS and MODEL_LAYERS, scale the amount of work so a run can
be made heavy enough to show up in ``nvidia-smi``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from loguru import logger
from torch import nn

from .errors import DeviceError
from .varspec import VarSpec


def select_device(requested: str) -> torch.device:
    """Resolve DEVICE to a concrete torch device, failing loudly when 'cuda' is a lie."""
    available = torch.cuda.is_available()

    if requested == "cuda":
        if not available:
            raise DeviceError(
                "DEVICE=cuda but no CUDA device is visible to this process. "
                "Check that the container was started with GPU access "
                "(e.g. `docker run --gpus all`) and that the NVIDIA driver is present. "
                f"torch {torch.__version__}, built for CUDA {torch.version.cuda}."
            )
        device = torch.device("cuda")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:  # auto
        device = torch.device("cuda" if available else "cpu")
        if not available:
            logger.warning("DEVICE=auto and no CUDA device is visible; falling back to CPU")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            f"Using CUDA device {props.name!r} "
            f"(compute capability {props.major}.{props.minor}, "
            f"{props.total_memory / 1024**3:.1f} GiB, "
            f"torch {torch.__version__}/cu{torch.version.cuda})"
        )
    else:
        logger.info(f"Using CPU (torch {torch.__version__})")
    return device


class DummyNet(nn.Module):
    """A plain convolutional stack mapping input channels to output channels.

    Weights are drawn from a seeded CPU generator and then moved to the device, so a
    given RANDOM_SEED gives the same weights whether the run is on CPU or GPU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        seed: int,
    ) -> None:
        super().__init__()
        widths = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], kernel_size=3, padding=1, padding_mode="replicate")
            for i in range(n_layers)
        )
        self.activation = nn.GELU()
        self._init_weights(seed)

    def _init_weights(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        for i, conv in enumerate(self.convs):
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            # Roughly variance-preserving: He scaling for the GELU layers, Xavier-ish
            # for the linear output layer. Keeps a deep stack from saturating or
            # exploding, so the output stays finite and worth looking at.
            gain = 2.0 if i < len(self.convs) - 1 else 1.0
            std = math.sqrt(gain / fan_in)
            with torch.no_grad():
                conv.weight.copy_(
                    torch.empty_like(conv.weight).normal_(0.0, std, generator=generator)
                )
                conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = self.activation(conv(x))
        return self.convs[-1](x)


def channel_stats(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of a ``(time, channel, y, x)`` array."""
    mean = fields.mean(axis=(0, 2, 3), dtype="float64").astype("float32")
    std = fields.std(axis=(0, 2, 3), dtype="float64").astype("float32")
    # A constant channel has zero spread; keep the scale at 1 so normalisation is a
    # plain shift rather than a division by zero.
    std = np.where(std > 0, std, 1.0).astype("float32")
    return mean, std


def output_stats(
    in_layout: list[tuple[VarSpec, int | None]],
    out_layout: list[tuple[VarSpec, int | None]],
    in_mean: np.ndarray,
    in_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a plausible scale for each output channel.

    Output variables that also appear in the input inherit that variable's statistics,
    so a predicted ``t2m`` lands near 273 K rather than near 0. Genuinely new output
    variables fall back to the average input scale.
    """
    by_channel = {(spec.name, level): i for i, (spec, level) in enumerate(in_layout)}
    by_name: dict[str, list[int]] = {}
    for i, (spec, _) in enumerate(in_layout):
        by_name.setdefault(spec.name, []).append(i)

    mean = np.empty(len(out_layout), dtype="float32")
    std = np.empty(len(out_layout), dtype="float32")
    for i, (spec, level) in enumerate(out_layout):
        if (spec.name, level) in by_channel:
            source = [by_channel[(spec.name, level)]]
        elif spec.name in by_name:
            source = by_name[spec.name]
        else:
            source = list(range(len(in_layout)))
        mean[i] = in_mean[source].mean()
        std[i] = in_std[source].mean()
    return mean, std


def feedback_index(
    in_layout: list[tuple[VarSpec, int | None]],
    out_layout: list[tuple[VarSpec, int | None]],
) -> np.ndarray:
    """For each input channel, the output channel that feeds it on the next step.

    ``-1`` means the network predicts nothing for that input channel, so the previous
    value is carried forward during the rollout.
    """
    by_channel = {(spec.name, level): i for i, (spec, level) in enumerate(out_layout)}
    return np.array(
        [by_channel.get((spec.name, level), -1) for spec, level in in_layout],
        dtype="int64",
    )


def predict(
    fields: np.ndarray,
    net: DummyNet,
    device: torch.device,
    n_forecast_steps: int,
    in_mean: np.ndarray,
    in_std: np.ndarray,
    out_mean: np.ndarray,
    out_std: np.ndarray,
    feedback: np.ndarray,
) -> np.ndarray:
    """Run the network, returning a ``(time, channel, y, x)`` float32 array.

    ``n_forecast_steps == -1`` runs one batched pass over every input timestep.
    A positive count rolls the network forward autoregressively, one forward pass per
    step, so wall-clock time scales with the forecast length the way a real model does.
    """
    net = net.to(device).eval()
    height, width = fields.shape[-2:]
    logger.info(
        f"Running {type(net).__name__} on {device.type}: {fields.shape[1]} input "
        f"channel(s) -> {out_mean.size} output channel(s), grid {height}x{width}"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_mean = torch.from_numpy(in_mean).to(device).view(1, -1, 1, 1)
    t_std = torch.from_numpy(in_std).to(device).view(1, -1, 1, 1)
    o_mean = torch.from_numpy(out_mean).to(device).view(1, -1, 1, 1)
    o_std = torch.from_numpy(out_std).to(device).view(1, -1, 1, 1)

    x = (torch.from_numpy(fields).to(device) - t_mean) / t_std

    started = time.perf_counter()
    with torch.inference_mode():
        if n_forecast_steps == -1:
            y = net(x)
        else:
            y = _rollout(x, net, n_forecast_steps, feedback, device)
        y = y * o_std + o_mean
        result = y.to("cpu").numpy().astype("float32")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    _log_work(device, elapsed, result.shape[0])
    return result


def _rollout(
    x: torch.Tensor,
    net: DummyNet,
    n_steps: int,
    feedback: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive rollout from the last input timestep."""
    state = x[-1:]  # (1, C_in, H, W)
    t_feedback = torch.from_numpy(feedback).to(device)
    has_source = t_feedback >= 0
    source = t_feedback.clamp(min=0)

    steps = []
    for _ in range(n_steps):
        prediction = net(state)
        steps.append(prediction)
        # Feed predicted channels back where a matching output variable exists; carry
        # the previous value forward for input-only channels (e.g. static fields).
        fed = prediction.index_select(1, source)
        state = torch.where(has_source.view(1, -1, 1, 1), fed, state)
    return torch.cat(steps, dim=0)


def _log_work(device: torch.device, elapsed: float, n_steps: int) -> None:
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        logger.info(
            f"Forward pass complete: {n_steps} output timestep(s) in {elapsed:.3f} s "
            f"on {torch.cuda.get_device_name(device)}, peak GPU memory {peak:.1f} MiB"
        )
        if peak <= 0:
            raise DeviceError(
                "The forward pass allocated no GPU memory, which means it did not run "
                "on the GPU. Treating this as a device failure."
            )
    else:
        logger.info(
            f"Forward pass complete: {n_steps} output timestep(s) in {elapsed:.3f} s on CPU"
        )
