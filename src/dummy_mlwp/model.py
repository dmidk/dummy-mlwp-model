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
    """Resolve DEVICE to a concrete torch device.

    Parameters
    ----------
    requested : {'auto', 'cuda', 'cpu'}
        The requested device. ``'auto'`` prefers CUDA and falls back to CPU with a
        warning; ``'cuda'`` is a hard requirement.

    Returns
    -------
    torch.device
        The device to run on. Its name and capabilities are logged.

    Raises
    ------
    DeviceError
        If ``'cuda'`` was requested but no CUDA device is visible. The message points
        at the usual causes — a container started without GPU access, or a missing
        driver — because silently running on CPU would defeat the purpose.
    """
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
    else:
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
    given seed gives the same weights whether the run is on CPU or GPU.

    Parameters
    ----------
    in_channels : int
        Number of 2D input fields.
    out_channels : int
        Number of 2D output fields.
    hidden_channels : int
        Width of the intermediate convolutions.
    n_layers : int
        Total number of convolutions, at least 2.
    seed : int
        Seed for the weight initialisation.
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
        """Fill the convolutions with reproducible, variance-preserving weights.

        Parameters
        ----------
        seed : int
            Seed for the CPU generator used to draw the weights.

        Notes
        -----
        He scaling is used for the GELU layers and Xavier-ish scaling for the linear
        output layer. This keeps a deep stack from saturating or exploding, so the
        output stays finite and worth looking at.
        """
        generator = torch.Generator().manual_seed(seed)
        for i, conv in enumerate(self.convs):
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            gain = 2.0 if i < len(self.convs) - 1 else 1.0
            std = math.sqrt(gain / fan_in)
            with torch.no_grad():
                conv.weight.copy_(
                    torch.empty_like(conv.weight).normal_(0.0, std, generator=generator)
                )
                conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional stack.

        Parameters
        ----------
        x : torch.Tensor
            A ``(batch, in_channels, y, x)`` tensor.

        Returns
        -------
        torch.Tensor
            A ``(batch, out_channels, y, x)`` tensor.
        """
        for conv in self.convs[:-1]:
            x = self.activation(conv(x))
        return self.convs[-1](x)


def channel_stats(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel normalisation statistics.

    Parameters
    ----------
    fields : numpy.ndarray
        A ``(time, channel, y, x)`` array.

    Returns
    -------
    mean : numpy.ndarray
        Per-channel mean, float32.
    std : numpy.ndarray
        Per-channel standard deviation, float32. A constant channel has zero spread,
        so its scale is forced to 1 and normalisation becomes a plain shift rather
        than a division by zero.
    """
    mean = fields.mean(axis=(0, 2, 3), dtype="float64").astype("float32")
    std = fields.std(axis=(0, 2, 3), dtype="float64").astype("float32")
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

    Parameters
    ----------
    in_layout, out_layout : list of tuple
        Channel layouts from :func:`~dummy_mlwp.varspec.channel_layout`.
    in_mean, in_std : numpy.ndarray
        Per-input-channel statistics from :func:`channel_stats`.

    Returns
    -------
    mean : numpy.ndarray
        Per-output-channel mean to shift by, float32.
    std : numpy.ndarray
        Per-output-channel scale to multiply by, float32.

    Notes
    -----
    Matching is tried on ``(name, level)`` first, then on name alone, then falls back
    to the mean over all input channels.
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
    """Map each input channel to the output channel that feeds it on the next step.

    Parameters
    ----------
    in_layout, out_layout : list of tuple
        Channel layouts from :func:`~dummy_mlwp.varspec.channel_layout`.

    Returns
    -------
    numpy.ndarray
        One int64 entry per input channel: the index of the matching output channel,
        or ``-1`` when the network predicts nothing for it. A ``-1`` channel keeps its
        previous value during the rollout, which is what a static field should do.
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
    """Run the network on the given device.

    Parameters
    ----------
    fields : numpy.ndarray
        A ``(time, channel, y, x)`` float32 array of input fields.
    net : DummyNet
        The network to run. It is moved to ``device`` and put in eval mode.
    device : torch.device
        Where to run.
    n_forecast_steps : int
        ``-1`` for one batched pass over every input timestep, or a positive number of
        autoregressive rollout steps.
    in_mean, in_std : numpy.ndarray
        Per-input-channel normalisation statistics.
    out_mean, out_std : numpy.ndarray
        Per-output-channel denormalisation statistics.
    feedback : numpy.ndarray
        Input-to-output channel mapping from :func:`feedback_index`, used by the
        rollout.

    Returns
    -------
    numpy.ndarray
        A ``(time, channel, y, x)`` float32 array of predictions.

    Raises
    ------
    DeviceError
        If the run was meant to be on a GPU but allocated no GPU memory.

    Notes
    -----
    A positive ``n_forecast_steps`` costs one forward pass per step, so wall-clock time
    scales with the forecast length the way a real model does — which is the property a
    scheduler test actually cares about.
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
    """Roll the network forward autoregressively from the last input timestep.

    Parameters
    ----------
    x : torch.Tensor
        Normalised ``(time, channel, y, x)`` input.
    net : DummyNet
        The network, already on ``device``.
    n_steps : int
        Number of forecast steps to produce.
    feedback : numpy.ndarray
        Input-to-output channel mapping from :func:`feedback_index`.
    device : torch.device
        Where to run.

    Returns
    -------
    torch.Tensor
        A ``(n_steps, out_channels, y, x)`` tensor of normalised predictions.

    Notes
    -----
    Predicted channels are fed back where a matching output variable exists; channels
    with no counterpart carry their previous value forward.
    """
    state = x[-1:]
    t_feedback = torch.from_numpy(feedback).to(device)
    has_source = t_feedback >= 0
    source = t_feedback.clamp(min=0)

    steps = []
    for _ in range(n_steps):
        prediction = net(state)
        steps.append(prediction)
        fed = prediction.index_select(1, source)
        state = torch.where(has_source.view(1, -1, 1, 1), fed, state)
    return torch.cat(steps, dim=0)


def _log_work(device: torch.device, elapsed: float, n_steps: int) -> None:
    """Report what the forward pass actually cost.

    Parameters
    ----------
    device : torch.device
        The device the pass ran on.
    elapsed : float
        Wall-clock seconds taken.
    n_steps : int
        Number of output timesteps produced.

    Raises
    ------
    DeviceError
        If the device is CUDA but no GPU memory was allocated, which means the work
        did not actually run there.
    """
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
