# =============================================================
#  evaluate.py — TTA evaluation, confusion matrix, report
# =============================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from config import CLASSES, BATCH_SIZE, NUM_WORKERS, SAVE_DIR
from dataset import SugarcaneDataset, tta_tfs


@torch.no_grad()
def tta_predict(model, val_paths, val_labels, device):
    model.eval()
    all_probs  = None
    targets    = np.array(val_labels)

    for tfm in tta_tfs:
        ds  = SugarcaneDataset(val_paths, val_labels, tfm)
        dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)
        probs = []
        for imgs, _ in dl:
            imgs = imgs.to(device)
            with torch.amp.autocast(device_type='cuda'):
                out = torch.softmax(model(imgs), dim=1)
            probs.append(out.cpu().numpy())
        probs = np.concatenate(probs)
        all_probs = probs if all_probs is None else all_probs + probs

    all_probs /= len(tta_tfs)
    preds = np.argmax(all_probs, axis=1)
    acc   = (preds == targets).mean()
    torch.cuda.empty_cache()
    return preds, targets, acc, all_probs


def plot_curves(history, save_dir=SAVE_DIR):
    ep = range(1, len(history['tr_acc'])+1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(ep, [a*100 for a in history['tr_acc']], label='Train', color='#1565C0', lw=2)
    axes[0].plot(ep, [a*100 for a in history['va_acc']], label='Val',   color='#2E7D32', lw=2)
    axes[0].axhline(97, ls='--', color='red', alpha=0.5, label='97% target')
    axes[0].set_title('Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history['tr_loss'], label='Train', color='#1565C0', lw=2)
    axes[1].plot(ep, history['va_loss'], label='Val',   color='#2E7D32', lw=2)
    axes[1].set_title('Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('EfficientNet-B4 Training', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/training_curves.png'
    plt.savefig(path, dpi=150); plt.close()
    print(f'  Curves saved → {path}')


def plot_confusion(preds, targets, acc, save_dir=SAVE_DIR):
    cm  = confusion_matrix(targets, preds)
    df  = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='d', cmap='Greens',
                linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix  (TTA Acc = {acc*100:.2f}%)',
              fontsize=12, fontweight='bold')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    path = f'{save_dir}/confusion_matrix.png'
    plt.savefig(path, dpi=150); plt.close()
    print(f'  Confusion matrix → {path}')


def full_eval(model, val_paths, val_labels, device):
    print('\n  Running TTA evaluation...')
    preds, targets, acc, probs = tta_predict(model, val_paths, val_labels, device)
    print(f'\n  TTA Validation Accuracy: {acc*100:.2f}%')
    print('─'*50)
    print(classification_report(targets, preds, target_names=CLASSES, digits=4))
    plot_curves.__doc__  # just a reference
    plot_confusion(preds, targets, acc)
    return acc
