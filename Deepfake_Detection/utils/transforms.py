import numpy as np
import cv2
from scipy.fftpack import dct, idct, fft2, fftshift
from scipy.ndimage import gaussian_filter
from config import Config


def extract_lnp_fast(image: np.ndarray) -> np.ndarray:
    img_uint8 = image.astype(np.uint8)
    if Config.LNP_USE_CV2_DENOISE:
        try:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_uint8, None,
                h=Config.LNP_CV2_H, hColor=Config.LNP_CV2_H,
                templateWindowSize=7, searchWindowSize=21,
            )
            lnp = (img_uint8.astype(np.float32) - denoised.astype(np.float32)) / 255.0
        except Exception:
            img_f = image.astype(np.float32) / 255.0
            lnp = np.stack([
                ch - gaussian_filter(ch, sigma=Config.LNP_GAUSSIAN_SIGMA)
                for ch in img_f.transpose(2, 0, 1)
            ], axis=2)
    else:
        img_f = image.astype(np.float32) / 255.0
        lnp = np.stack([
            ch - gaussian_filter(ch, sigma=Config.LNP_GAUSSIAN_SIGMA)
            for ch in img_f.transpose(2, 0, 1)
        ], axis=2)
    return lnp.astype(np.float32)


def compute_amplitude_spectrum(lnp: np.ndarray) -> np.ndarray:
    lnp_gray = lnp.mean(axis=2)
    amp = np.abs(fftshift(fft2(lnp_gray)))
    amp_log = np.log1p(amp)
    denom = amp_log.max() - amp_log.min() + 1e-8
    return ((amp_log - amp_log.min()) / denom).astype(np.float32)


def compute_phase_spectrum(lnp: np.ndarray) -> np.ndarray:
    lnp_gray = lnp.mean(axis=2)
    phase = np.angle(fftshift(fft2(lnp_gray)))
    denom = phase.max() - phase.min() + 1e-8
    return ((phase - phase.min()) / denom).astype(np.float32)


def create_dct_frequency_bands_fast(image: np.ndarray) -> np.ndarray:
    H, W, C = image.shape
    bs = Config.BLOCK_SIZE
    H2 = (H // bs) * bs
    W2 = (W // bs) * bs
    img = image[:H2, :W2].astype(np.float32)

    bands = []
    for c in range(C):
        ch = img[:, :, c]
        blocks = (ch.reshape(H2 // bs, bs, W2 // bs, bs)
                    .transpose(0, 2, 1, 3))
        d = dct(dct(blocks, axis=2, norm='ortho'), axis=3, norm='ortho')

        low_m  = np.zeros((bs, bs), np.float32); low_m[:3, :3]   = 1
        mid_m  = np.zeros((bs, bs), np.float32); mid_m[3:6, 3:6] = 1
        high_m = np.zeros((bs, bs), np.float32); high_m[6:, 6:]  = 1

        def recon(mask):
            masked = d * mask
            rec = idct(idct(masked, axis=3, norm='ortho'), axis=2, norm='ortho')
            return rec.transpose(0, 2, 1, 3).reshape(H2, W2)

        bands.extend([recon(low_m), recon(mid_m), recon(high_m)])

    out = np.stack(bands, axis=-1)
    if out.shape[:2] != (H, W):
        pad_h, pad_w = H - H2, W - W2
        out = np.pad(out, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return out.astype(np.float32)


def extract_dct_coefficients(image: np.ndarray) -> np.ndarray:
    H, W, C = image.shape
    bs = Config.BLOCK_SIZE
    H2 = (H // bs) * bs
    W2 = (W // bs) * bs
    img = image[:H2, :W2]
    coeffs = []
    for c in range(C):
        ch = img[:, :, c].astype(np.float32)
        blocks = (ch.reshape(H2 // bs, bs, W2 // bs, bs)
                    .transpose(0, 2, 1, 3))
        d = dct(dct(blocks, axis=2, norm='ortho'), axis=3, norm='ortho')
        coeffs.append(d.reshape(-1, bs, bs))
    return np.array(coeffs, dtype=np.float32)


def compute_beta_statistics(dct_coeffs: np.ndarray) -> np.ndarray:
    all_blocks = dct_coeffs.reshape(-1, Config.BLOCK_SIZE, Config.BLOCK_SIZE)
    flat = []
    for i in range(Config.BLOCK_SIZE):
        for j in range(Config.BLOCK_SIZE):
            if i == 0 and j == 0:
                continue
            flat.append(all_blocks[:, i, j])
    flat = np.stack(flat, axis=1)
    return (np.std(flat, axis=0) / np.sqrt(2)).astype(np.float32)