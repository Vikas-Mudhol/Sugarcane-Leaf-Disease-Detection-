# =============================================================
#  dataset.py
# =============================================================
import os
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from config import CLASSES, DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, NUM_CLASSES
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ── Augmentations ─────────────────────────────────────────────
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.08, scale=(0.85,1.15), rotate=(-30,30), p=0.6),
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
        A.HueSaturationValue(20,30,20,p=1.0),
        A.RGBShift(15,15,15,p=1.0),
    ], p=0.6),
    A.OneOf([
        A.RandomBrightnessContrast(0.25,0.25,p=1.0),
        A.CLAHE(clip_limit=3.0,p=1.0),
        A.RandomGamma((80,120),p=1.0),
    ], p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3,5),p=1.0),
        A.MotionBlur(blur_limit=5,p=1.0),
    ], p=0.3),
    A.GaussNoise(std_range=(0.03,0.15), p=0.3),
    A.CoarseDropout(num_holes_range=(1,6), hole_height_range=(8,20),
                    hole_width_range=(8,20), fill=0, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

tta_tfs = [
    A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.Normalize(mean=MEAN,std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.HorizontalFlip(p=1), A.Normalize(mean=MEAN,std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.VerticalFlip(p=1), A.Normalize(mean=MEAN,std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.RandomRotate90(p=1), A.Normalize(mean=MEAN,std=STD), ToTensorV2()]),
]


class SugarcaneDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert('RGB'))
        return self.transform(image=img)['image'], self.labels[idx]


def get_loaders():
    VALID = ('.jpg','.jpeg','.png','.bmp','.webp','.tiff')
    c2i   = {c:i for i,c in enumerate(CLASSES)}
    paths, labels = [], []

    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            print(f'[WARN] Missing: {folder}'); continue
        for f in os.listdir(folder):
            if f.lower().endswith(VALID):
                paths.append(os.path.join(folder, f))
                labels.append(c2i[cls])

    print(f'\nDataset: {len(paths)} images across {NUM_CLASSES} classes')
    cnt = Counter(labels)
    for i,c in enumerate(CLASSES):
        print(f'  {c:<15}: {cnt.get(i,0):>5}')

    tr_p, va_p, tr_l, va_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
    print(f'\nTrain: {len(tr_p)}  |  Val: {len(va_p)}\n')

    # Weighted sampler for class balance
    lc      = Counter(tr_l)
    weights = [1.0/lc[l] for l in tr_l]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_loader = DataLoader(SugarcaneDataset(tr_p, tr_l, train_tf),
                           batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=NUM_WORKERS, pin_memory=False)
    va_loader = DataLoader(SugarcaneDataset(va_p, va_l, val_tf),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=False)
    return tr_loader, va_loader, va_p, va_l
