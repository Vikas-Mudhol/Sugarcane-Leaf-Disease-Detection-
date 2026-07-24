# =============================================================
#  main.py — Train EfficientNet-B4 for sugarcane disease
#
#  Usage:
#    python main.py
# =============================================================
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from utils import set_seed
from config import SEED, NUM_CLASSES, DROPOUT
from dataset import get_loaders
from model import build_model
from train import run
from evaluate import full_eval, plot_curves


def main():
    # ── Device ────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise SystemExit('❌  CUDA GPU not found. Training requires GPU.')
    device = torch.device('cuda')
    prop   = torch.cuda.get_device_properties(0)
    print(f'\n  GPU  : {prop.name}')
    print(f'  VRAM : {prop.total_memory/1e9:.1f} GB')
    print(f'  CUDA : {torch.version.cuda}\n')

    set_seed(SEED)
    torch.cuda.empty_cache()

    # ── Data ──────────────────────────────────────────────────
    tr_loader, va_loader, va_paths, va_labels = get_loaders()

    # ── Model ─────────────────────────────────────────────────
    model = build_model(NUM_CLASSES, DROPOUT).to(device)
    total = sum(p.numel() for p in model.parameters())
    head  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model : EfficientNet-B4')
    print(f'  Total params     : {total:,}')
    print(f'  Trainable (head) : {head:,}\n')

    # ── Train ─────────────────────────────────────────────────
    model, history, best_acc = run(model, tr_loader, va_loader, device)

    # ── Evaluate ──────────────────────────────────────────────
    plot_curves(history)
    tta_acc = full_eval(model, va_paths, va_labels, device)

    print('\n' + '━'*50)
    print(f'  Final TTA Accuracy : {tta_acc*100:.2f}%')
    if tta_acc >= 0.97:
        print('  ✅ Target 97%+ achieved!')
    else:
        print('  ℹ️  Below 97% — try adding more images or run again')
    print('━'*50 + '\n')


if __name__ == '__main__':
    main()
