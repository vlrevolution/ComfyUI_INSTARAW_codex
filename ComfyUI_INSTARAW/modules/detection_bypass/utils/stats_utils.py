# Filename: ComfyUI_INSTARAW/modules/detection_bypass/utils/stats_utils.py
# ---
# Helper for matching images to reference iPhone statistics
import numpy as np
import torch
import torch.nn.functional as F


class StatsMatcher:
    def __init__(self, stats_path, device, weights=None):
        data = np.load(stats_path)
        self.device = torch.device(device)
        self.weights = weights or {
            "spectrum": 0.7,
            "chroma": 0.25,
            "texture": 0.05,
        }

        spectra = data["spectra"]
        chroma = data["chroma_mean"]
        chroma_cov = data["chroma_cov"]
        glcm = data["glcm"]

        self.spectrum_mean = torch.tensor(
            spectra.mean(axis=0), dtype=torch.float32, device=self.device
        )
        self.spectrum_std = torch.tensor(
            spectra.std(axis=0) + 1e-6, dtype=torch.float32, device=self.device
        )
        self.chroma_mean = torch.tensor(
            chroma.mean(axis=0), dtype=torch.float32, device=self.device
        )
        self.chroma_std = torch.tensor(
            chroma.std(axis=0) + 1e-6, dtype=torch.float32, device=self.device
        )
        self.chroma_cov_mean = torch.tensor(
            chroma_cov.mean(axis=0), dtype=torch.float32, device=self.device
        )
        self.chroma_cov_std = torch.tensor(
            chroma_cov.std(axis=0) + 1e-6, dtype=torch.float32, device=self.device
        )
        self.texture_mean = torch.tensor(
            glcm.mean(axis=0), dtype=torch.float32, device=self.device
        )
        self.texture_std = torch.tensor(
            glcm.std(axis=0) + 1e-6, dtype=torch.float32, device=self.device
        )

        self.radial_cache = {}
        sobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.sobel_x = sobel.view(1, 1, 3, 3).to(self.device)
        self.sobel_y = sobel.t().view(1, 1, 3, 3).to(self.device)

    def _radial_profile(self, mag, bins):
        b, h, w = mag.shape
        key = (mag.device, h, w, bins)
        cache = self.radial_cache.get(key)
        if cache is None:
            y = torch.arange(h, device=mag.device, dtype=torch.float32)
            x = torch.arange(w, device=mag.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            r = torch.clamp(r.long(), max=bins - 1)
            counts = torch.bincount(r.view(-1), minlength=bins).float().clamp_min(1.0)
            cache = {
                "index": r.view(-1),
                "counts": counts,
            }
            self.radial_cache[key] = cache

        profile = []
        for sample in mag:
            binsums = torch.zeros(bins, device=mag.device)
            binsums.scatter_add_(0, cache["index"], sample.view(-1))
            profile.append(binsums / cache["counts"])
        return torch.stack(profile, dim=0)

    def __call__(self, img_tensor):
        b, c, h, w = img_tensor.shape
        gray = (
            0.299 * img_tensor[:, 0:1]
            + 0.587 * img_tensor[:, 1:2]
            + 0.114 * img_tensor[:, 2:3]
        )
        fft = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-1, -2))
        mag = torch.log1p(torch.abs(fft)).mean(1)
        spectrum = self._radial_profile(mag, bins=self.spectrum_mean.shape[0])
        spec_target = self.spectrum_mean.unsqueeze(0).expand_as(spectrum)
        spec_std = self.spectrum_std.unsqueeze(0).expand_as(spectrum)
        loss_spectrum = F.mse_loss(
            (spectrum - spec_target) / spec_std,
            torch.zeros_like(spectrum),
        )

        pixels = img_tensor.view(b, c, -1)
        mean = pixels.mean(dim=-1, keepdim=True)
        chroma_mean = mean.squeeze(-1)
        target_chroma = self.chroma_mean.unsqueeze(0).expand_as(chroma_mean)
        chroma_std = self.chroma_std.unsqueeze(0).expand_as(chroma_mean)
        z = (chroma_mean - target_chroma) / chroma_std
        loss_chroma_mean = F.mse_loss(z, torch.zeros_like(z))
        centered = pixels - mean
        cov = centered @ centered.transpose(1, 2) / pixels.shape[-1]
        target_cov = self.chroma_cov_mean.unsqueeze(0).expand_as(cov)
        cov_std = self.chroma_cov_std.unsqueeze(0).expand_as(cov)
        z_cov = (cov - target_cov) / cov_std
        loss_chroma_cov = F.mse_loss(z_cov, torch.zeros_like(z_cov))

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        contrast = grad_mag.mean(dim=(1, 2, 3))
        mu = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        var = F.avg_pool2d(gray ** 2, 5, 1, 2) - mu ** 2
        homogeneity = (1.0 / (1.0 + var)).mean(dim=(1, 2, 3))
        texture_vec = torch.stack([contrast.mean(), homogeneity.mean()])
        loss_texture = F.mse_loss(
            (texture_vec - self.texture_mean) / self.texture_std,
            torch.zeros_like(texture_vec),
        )

        total = (
            self.weights["spectrum"] * loss_spectrum
            + self.weights["chroma"] * (loss_chroma_mean + loss_chroma_cov)
            + self.weights["texture"] * loss_texture
        )
        details = {
            "spectrum": loss_spectrum.detach(),
            "chroma_mean": loss_chroma_mean.detach(),
            "chroma_cov": loss_chroma_cov.detach(),
            "texture": loss_texture.detach(),
        }
        return total, details