"""
Advanced loss functions for full UnMarker implementation.
Ported from ai-watermark/modules/attack/unmark/losses.py

These losses are used in the two-stage UnMarker attack:
- Stage 1 (high_freq): FFTLoss targeting high frequencies where AI fingerprints hide
- Stage 2 (low_freq): FFTLoss targeting low frequencies for robustness
- Both stages: perceptual constraint (LPIPS or Deeploss-VGG)
"""

import os
from pathlib import Path
from collections import namedtuple

import torch
import lpips
try:
    from lpips import util as lpips_util
except ImportError:  # pragma: no cover - lpips provides util in normal installs
    lpips_util = None
from torchvision import transforms, models

class NormLoss(torch.nn.Module):
    """Basic L-p norm loss between two tensors."""
    def __init__(self, norm=2, power=2):
        super().__init__()
        self.norm, self.power = norm, power

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, x.shape[-2], x.shape[-1])
        return (
            torch.pow(
                torch.norm(
                    x - y, p=self.norm, dim=tuple(list(range(1, len((x).shape))))
                ),
                self.power,
            )
            / torch.prod(torch.tensor(x.shape[1:]))
        ).view(x.shape[0], -1)


class PerceptualLoss(torch.nn.Module):
    """Wrapper for perceptual loss functions like LPIPS."""
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
        return self.loss_fn(x, y).view(x.shape[0], -1)


class FFTLoss(NormLoss):
    """
    CRITICAL: This is the core of UnMarker.
    Computes loss in the frequency domain to directly attack spectral fingerprints.

    IMPORTANT: Processes in GRAYSCALE to avoid blue hue color shift!
    Research warning: "By independently manipulating the spectra of RGB channels,
    the FFT algorithm destroys natural inter-channel correlation, causing severe
    color artifacts and blue/purple color cast."

    Args:
        norm: L-p norm to use (1 or 2)
        power: Power to raise the norm to
        n_fft: FFT size (None = same as input)
        use_tanh: Apply tanh to frequency magnitudes (helps with stability)
        use_grayscale: Convert to grayscale before FFT (DEFAULT: True to prevent color shift)
    """
    def __init__(self, norm=1, power=1, n_fft=None, use_tanh=False, use_grayscale=True):
        super().__init__(norm=norm, power=power)
        self.tanh = torch.nn.Tanh() if use_tanh else (lambda x: x)
        self.fft_norm = "ortho" if use_tanh else None
        self.n_fft = n_fft
        self.use_grayscale = use_grayscale

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])

        # CRITICAL FIX: Convert to grayscale to prevent color shift
        if self.use_grayscale and x.shape[1] == 3:
            # RGB to grayscale (ITU-R BT.601)
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            y = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]

        # Convert to frequency domain
        x_f = self.tanh(
            torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(x, dim=(-1, -2)),
                    norm=self.fft_norm,
                    s=self.n_fft,
                ),
                dim=(-1, -2),
            )
        )
        y_f = self.tanh(
            torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(y, dim=(-1, -2)),
                    norm=self.fft_norm,
                    s=self.n_fft,
                ),
                dim=(-1, -2),
            )
        )

        # Compute loss in frequency domain
        result = super().forward(x_f, y_f)

        # DEBUG: Check if loss is zero
        if result.abs().max() < 1e-10:
            import warnings
            warnings.warn(f"FFTLoss returning near-zero! x_f range: [{x_f.abs().min():.8f}, {x_f.abs().max():.8f}], "
                         f"y_f range: [{y_f.abs().min():.8f}, {y_f.abs().max():.8f}]")

        return result


class LpipsVGG(PerceptualLoss):
    """
    LPIPS perceptual loss using VGG backbone.
    This ensures the attack remains imperceptible to human vision.
    """
    def __init__(self, model_path=None):
        # Try to load from pretrained models dir, fallback to auto-download
        try:
            if model_path and torch.cuda.is_available():
                lps_loss = lpips.LPIPS(
                    net="vgg",
                    model_path=model_path,
                    verbose=False,
                ).eval()
            else:
                # Auto-download if no path provided
                lps_loss = lpips.LPIPS(net="vgg", verbose=False).eval()
        except Exception as e:
            print(f"Warning: Failed to load VGG LPIPS, falling back to Alex: {e}")
            lps_loss = lpips.LPIPS(net="alex", verbose=False).eval()

        loss_fn = lambda x, y: lps_loss(x, y, normalize=True)
        super().__init__(loss_fn)
        self.lps_loss = lps_loss


class LpipsAlex(PerceptualLoss):
    """
    LPIPS perceptual loss using AlexNet backbone.
    Faster than VGG, slightly less accurate.
    """
    def __init__(self, model_path=None):
        try:
            if model_path:
                lps_loss = lpips.LPIPS(
                    net="alex",
                    model_path=model_path,
                    lpips=False,
                    verbose=False,
                ).eval()
            else:
                lps_loss = lpips.LPIPS(net="alex", lpips=False, verbose=False).eval()
        except:
            lps_loss = lpips.LPIPS(net="alex", verbose=False).eval()

        loss_fn = lambda x, y: lps_loss(x, y, normalize=True)
        super().__init__(loss_fn)
        self.lps_loss = lps_loss


class MeanLoss(torch.nn.Module):
    """
    Mean pooling loss for multi-scale analysis.
    Used to ensure consistency across different scales.
    """
    def __init__(self, kernels=[(5, 5)]):
        super().__init__()
        assert isinstance(kernels, list) and len(kernels) > 0
        for kernel in kernels:
            self.__check_kernel(kernel)
        self.kernels = kernels
        self.mean_pools = [
            torch.nn.AvgPool3d(kernel_size=(3, *kernel), stride=(3, 1, 1))
            for kernel in self.kernels
        ]

    def __check_kernel(self, kernel):
        assert isinstance(kernel, tuple) and len(kernel) == 2
        assert isinstance(kernel[0], int) and isinstance(kernel[1], int)
        assert kernel[0] > 0 and kernel[1] > 0
        assert kernel[0] % 2 == 1 and kernel[1] % 2 == 1

    def _mean_pad(self, shape, kernel, stride):
        return [
            0,
            kernel[1] - (shape[-1] - (shape[-1] // kernel[1]) * kernel[1]) % kernel[1],
            0,
            kernel[0] - (shape[-2] - (shape[-2] // kernel[0]) * kernel[0]) % kernel[0],
        ]

    def _mean_diff(self, x, y, pool, padding):
        x_p = pool(
            torch.nn.functional.pad(x, padding, mode="reflect").unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            y_p = pool(
                torch.nn.functional.pad(y, padding, mode="reflect").unsqueeze(1)
            ).squeeze(1)
        return (x_p - y_p).abs().flatten(1).sum(-1, keepdims=True)

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
        paddings = [self._mean_pad(x.shape, kernel, (1, 1)) for kernel in self.kernels]
        return torch.concat(
            [
                self._mean_diff(x, y, pool, padding)
                for pool, padding in zip(self.mean_pools, paddings)
            ],
            -1,
        ).sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# Deeploss-VGG (full perceptual loss from ai-watermark)
# ---------------------------------------------------------------------------

_DEEPloss_WEIGHT_NAME = "rgb_pnet_lin_vgg_trial0.pth"


def _find_deeploss_weight_file(filename=_DEEPloss_WEIGHT_NAME) -> Path:
    """Locate the Deeploss weight file in common install locations."""
    candidates = []
    env = os.getenv("DEEPloss_WEIGHTS_DIR") or os.getenv("DEEPLOSS_WEIGHTS_DIR")
    if env:
        candidates.append(Path(env))

    here = Path(__file__).resolve()
    comfy_root = None
    for parent in here.parents:
        if (parent / "pretrained").exists():
            comfy_root = parent
            break
    if comfy_root:
        candidates.append(comfy_root / "pretrained" / "deeploss")
    for parent in here.parents:
        candidates.append(parent / "pretrained_models" / "loss_provider" / "weights")
        candidates.append(
            parent
            / "ai-watermark"
            / "pretrained_models"
            / "loss_provider"
            / "weights"
        )

    seen = set()
    for directory in candidates:
        if directory is None:
            continue
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Deeploss weight '{filename}' not found. Run ai-watermark/download_data_and_models.sh "
        "or set DEEPLOSS_WEIGHTS_DIR to the directory containing the weights."
    )


class _NetLinLayer(torch.nn.Module):
    """A single linear layer applied on top of normalized VGG features."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = []
        layers.append(torch.nn.Dropout(p=0.5 if use_dropout else 0.0))
        layers.append(
            torch.nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.model = torch.nn.Sequential(*layers)


def _normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)).view(
        in_feat.size()[0], 1, in_feat.size()[2], in_feat.size()[3]
    )
    return in_feat / (norm_factor.expand_as(in_feat) + eps)


class _VGG16Features(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class _PNetLin(torch.nn.Module):
    """LPIPS-style network with learned linear heads."""

    def __init__(
        self,
        colorspace="RGB",
        reduction="none",
        use_dropout=False,
        pnet_rand=False,
    ):
        super().__init__()
        self.reduction = reduction
        self.colorspace = colorspace
        self.net = _VGG16Features(pretrained=not pnet_rand, requires_grad=False)
        self.chns = [64, 128, 256, 512, 512]
        self.lin0 = _NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = _NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = _NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = _NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = _NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        self.shift = torch.nn.Parameter(
            torch.Tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1), requires_grad=False
        )
        self.scale = torch.nn.Parameter(
            torch.Tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1), requires_grad=False
        )

    def forward(self, in0, in1):
        in0_sc = (in0 - self.shift) / self.scale
        in1_sc = (in1 - self.shift) / self.scale

        if self.colorspace.upper() in ["GRAY", "GREY", "LA"]:
            if lpips_util is None:
                raise RuntimeError("lpips.util is required for grayscale Deeploss.")
            gray_fn = getattr(lpips_util, "tensor2tensorGrayscaleLazy", None)
            if gray_fn is None:
                gray_fn = lpips_util.tensor2tensorGrayscale
            in0_sc = gray_fn(in0_sc)
            in1_sc = gray_fn(in1_sc)

        outs0 = self.net(in0_sc)
        outs1 = self.net(in1_sc)

        diffs = []
        for kk, out0 in enumerate(outs0):
            feats0 = _normalize_tensor(out0)
            feats1 = _normalize_tensor(outs1[kk])
            diffs.append((feats0 - feats1) ** 2)

        result = 0
        for kk, lin in enumerate(self.lins):
            result = result + torch.mean(torch.mean(lin.model(diffs[kk]), dim=3), dim=2)

        result = result.view(result.size()[0], result.size()[1], 1, 1)
        if self.reduction == "sum":
            result = torch.sum(result)
        elif self.reduction == "mean":
            result = torch.mean(result)
        return result


class DeeplossVGG(torch.nn.Module):
    """
    Full Deeploss-VGG perceptual loss (as used in the UnMarker paper).

    Requires the pretrained linear-head weights from ai-watermark's
    download_data_and_models.sh script (file: rgb_pnet_lin_vgg_trial0.pth).
    """

    def __init__(
        self,
        colorspace="RGB",
        reduction="none",
        weights_path: str | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _PNetLin(colorspace=colorspace, reduction="none", use_dropout=False)

        weight_file = Path(weights_path) if weights_path else _find_deeploss_weight_file()
        state_dict = torch.load(weight_file, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Deeploss] Warning: missing keys {missing}")
        if unexpected:
            print(f"[Deeploss] Warning: unexpected keys {unexpected}")

        self.model.to(self.device).eval()
        self.reduction = reduction

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            val = self.model(x, y)
        if self.reduction == "mean":
            val = val.mean()
        elif self.reduction == "sum":
            val = val.sum()
        return val.view(x.shape[0], -1)


def get_loss(loss_type, **kwargs):
    """
    Factory function to create loss instances.

    Args:
        loss_type: One of ["NormLoss", "FFTLoss", "LpipsVGG", "LpipsAlex", "MeanLoss"]
        **kwargs: Arguments to pass to the loss constructor

    Returns:
        Loss function instance
    """
    loss_dict = {
        "NormLoss": NormLoss,
        "MeanLoss": MeanLoss,
        "FFTLoss": FFTLoss,
        "LpipsVGG": LpipsVGG,
        "LpipsAlex": LpipsAlex,
    }
    assert loss_type in list(loss_dict.keys()), f"Unknown loss type: {loss_type}"
    return loss_dict[loss_type](**kwargs)
