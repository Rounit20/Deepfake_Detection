import random
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageFilter
from config import Config


class CutoutTransform:
    def __init__(self, size=32):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1:]
        y1 = max(0, np.random.randint(h) - self.size // 2)
        x1 = max(0, np.random.randint(w) - self.size // 2)
        img[:, y1:min(h, y1+self.size), x1:min(w, x1+self.size)] = 0
        return img


def jpeg_compress(image, quality):
    try:
        img = Image.fromarray(image.astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=int(np.clip(quality, 1, 100)),
                 subsampling=random.choice([0, 1, 2]))
        buf.seek(0)
        return np.array(Image.open(buf))
    except Exception:
        return image


def blur_aug(image):
    t = random.choice(['gaussian', 'motion', 'median'])
    if t == 'gaussian':
        return np.array(Image.fromarray(image).filter(
            ImageFilter.GaussianBlur(random.uniform(0.5, 2.0))))
    elif t == 'motion':
        k = random.choice([3, 5, 7])
        kern = np.zeros((k, k)); kern[k // 2] = 1 / k
        return cv2.filter2D(image, -1, kern)
    else:
        return cv2.medianBlur(image, random.choice([3, 5]))


def noise_aug(image):
    t = random.choice(['gaussian', 'salt_pepper', 'speckle'])
    if t == 'gaussian':
        return np.clip(
            image + np.random.normal(0, random.uniform(1, 10), image.shape),
            0, 255).astype(np.uint8)
    elif t == 'salt_pepper':
        p = random.uniform(0.001, 0.01)
        noisy = image.copy()
        noisy[np.random.random(image.shape[:2]) < p] = 255
        noisy[np.random.random(image.shape[:2]) < p] = 0
        return noisy
    else:
        return np.clip(
            image + image * np.random.randn(*image.shape) * 0.03,
            0, 255).astype(np.uint8)


def center_crop(image):
    H, W = image.shape[:2]
    s = min(H, W)
    top = (H - s) // 2
    left = (W - s) // 2
    return cv2.resize(
        image[top:top+s, left:left+s],
        (Config.IMG_SIZE, Config.IMG_SIZE),
        interpolation=cv2.INTER_AREA)