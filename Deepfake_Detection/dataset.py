import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import Config
from utils.transforms import (
    extract_lnp_fast, compute_amplitude_spectrum, compute_phase_spectrum,
    create_dct_frequency_bands_fast, extract_dct_coefficients, compute_beta_statistics,
)
from utils.augmentation import (
    CutoutTransform, jpeg_compress, blur_aug, noise_aug, center_crop,
)


train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_mask=False,
                 is_training=False, max_samples=None):
        self.root_dir    = root_dir
        self.transform   = transform
        self.use_mask    = use_mask
        self.is_training = is_training

        fake, real = [], []
        for lname in ['Fake', 'Real', 'fake', 'real']:
            label = 1 if lname.lower() == 'fake' else 0
            ldir  = os.path.join(root_dir, lname)
            if not os.path.isdir(ldir): continue
            found = [(os.path.join(ldir, f), label)
                     for f in os.listdir(ldir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            (fake if label == 1 else real).extend(found)
            print(f"   ✓ {lname}: {len(found)} images")

        print(f"   Total  Fake: {len(fake)}  Real: {len(real)}")

        if max_samples is not None:
            n = max_samples // 2
            rng = random.Random(Config.SEED)
            rng.shuffle(fake); rng.shuffle(real)
            fake, real = fake[:n], real[:n]

        self.samples = fake + real
        random.Random(Config.SEED).shuffle(self.samples)
        print(f"✓ Dataset ready: {len(self.samples)} samples\n")

    def __len__(self):
        return len(self.samples)

    def _load_lnp(self, img_path, face):
        if Config.USE_LNP_DISK_CACHE:
            os.makedirs(Config.LNP_CACHE_DIR, exist_ok=True)
            key  = os.path.basename(img_path).replace('.', '_') + '.npy'
            path = os.path.join(Config.LNP_CACHE_DIR, key)
            if os.path.exists(path):
                return np.load(path)
            lnp = extract_lnp_fast(face)
            np.save(path, lnp)
            return lnp
        return extract_lnp_fast(face)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            raw = cv2.imread(img_path)
            if raw is None:
                raise ValueError(f"Cannot read {img_path}")
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            if self.is_training and Config.USE_AUGMENTATION:
                if random.random() < Config.FLIP_ROTATION_PROB:
                    if random.random() < 0.5:
                        image = cv2.flip(image, 1)
                    ang = random.uniform(-5, 5)
                    M   = cv2.getRotationMatrix2D(
                        (image.shape[1]//2, image.shape[0]//2), ang, 1.0)
                    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                if random.random() < Config.JPEG_AUG_PROB:
                    image = jpeg_compress(image, np.random.choice(Config.JPEG_QUALITIES))
                if Config.USE_BLUR_AUGMENTATION and random.random() < Config.BLUR_AUG_PROB:
                    image = blur_aug(image)
                if Config.USE_NOISE_AUGMENTATION and random.random() < Config.NOISE_AUG_PROB:
                    image = noise_aug(image)

            face = center_crop(image)
            lnp  = self._load_lnp(img_path, face)

            if Config.USE_AMPLITUDE_PHASE:
                amp   = cv2.resize(compute_amplitude_spectrum(lnp),
                                   (Config.IMG_SIZE, Config.IMG_SIZE))
                phase = cv2.resize(compute_phase_spectrum(lnp),
                                   (Config.IMG_SIZE, Config.IMG_SIZE))
                lnp_feat = np.concatenate([lnp, amp[..., None], phase[..., None]], axis=2)
            else:
                lnp_feat = lnp

            dct_bands = create_dct_frequency_bands_fast(face)
            dct_c     = extract_dct_coefficients(face)
            beta      = compute_beta_statistics(dct_c)

            face_img = Image.fromarray(face)
            if self.transform:
                spatial = self.transform(face_img)
                if self.is_training and Config.USE_CUTOUT and \
                        random.random() < Config.CUTOUT_AUG_PROB:
                    spatial = CutoutTransform(Config.CUTOUT_SIZE)(spatial)
            else:
                spatial = torch.from_numpy(
                    face.astype(np.float32) / 255.0).permute(2, 0, 1)

            mask = torch.ones(1, Config.IMG_SIZE, Config.IMG_SIZE) if (self.use_mask and label == 1) \
                   else torch.zeros(1, Config.IMG_SIZE, Config.IMG_SIZE)

            return (spatial,
                    torch.from_numpy(dct_bands).permute(2, 0, 1),
                    torch.from_numpy(beta),
                    torch.from_numpy(lnp_feat).permute(2, 0, 1),
                    mask,
                    label)

        except Exception:
            return (
                torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE),
                torch.zeros(Config.DCT_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                torch.zeros(Config.NUM_AC_COEFFICIENTS),
                torch.zeros(Config.LNP_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                torch.zeros(1, Config.IMG_SIZE, Config.IMG_SIZE),
                label,
            )