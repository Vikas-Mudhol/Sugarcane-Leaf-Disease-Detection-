# =============================================================
#  train.py — Clean training engine
#  - AMP for memory efficiency (RTX 4060 safe)
#  - Two-phase training (head → full fine-tune)
#  - Early stopping, best model auto-save
# =============================================================
import os, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import (EPOCHS, LR, MIN_LR, WEIGHT_DECAY, LABEL_SMOOTHING,
                    WARMUP_EPOCHS, EARLY_STOP, MIXUP_PROB,
                    MIXUP_ALPHA, CUTMIX_ALPHA, MODEL_PATH)
from utils import mixup_data, cutmix_data, mixed_criterion
from model import unfreeze


# ── Warmup + Cosine LR ────────────────────────────────────────
class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.opt = optimizer; self.warmup = warmup_epochs
        self.total = total_epochs; self.min_lr = min_lr; self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / (self.total - self.warmup)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + np.cos(np.pi*progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr


# ── One train epoch ───────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda'):
            r = np.random.rand()
            if r < MIXUP_PROB * 0.5:                          # CutMix
                imgs, ya, yb, lam = cutmix_data(imgs, labels, CUTMIX_ALPHA)
                out  = model(imgs)
                loss = mixed_criterion(criterion, out, ya, yb, lam)
                primary = ya
            elif r < MIXUP_PROB:                              # Mixup
                imgs, ya, yb, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
                out  = model(imgs)
                loss = mixed_criterion(criterion, out, ya, yb, lam)
                primary = ya
            else:                                             # Normal
                out  = model(imgs)
                loss = criterion(out, labels)
                primary = labels

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.detach().argmax(1) == primary).sum().item()
        total    += imgs.size(0)

    return loss_sum/total, correct/total


# ── Validation ────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        with torch.amp.autocast(device_type='cuda'):
            out  = model(imgs)
            loss = criterion(out, labels)
        p = out.argmax(1)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (p == labels).sum().item()
        total    += imgs.size(0)
        preds_all.extend(p.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    torch.cuda.empty_cache()
    return loss_sum/total, correct/total, preds_all, labels_all


# ── Full training run ─────────────────────────────────────────
def run(model, tr_loader, va_loader, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler    = torch.amp.GradScaler(device='cuda')
    history   = dict(tr_loss=[], tr_acc=[], va_loss=[], va_acc=[])
    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())

    def gpu_mem():
        return f"{torch.cuda.memory_allocated()/1e9:.1f}GB"

    # ── Phase 1 — Head only (5 epochs) ───────────────────────
    P1_EPOCHS = 5
    opt1  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=WEIGHT_DECAY)
    sched1 = WarmupCosine(opt1, warmup_epochs=1, total_epochs=P1_EPOCHS,
                          min_lr=MIN_LR, base_lr=LR)

    print('\n' + '━'*62)
    print('  Phase 1 — Training head only (backbone frozen)')
    print('━'*62)

    for ep in range(P1_EPOCHS):
        torch.cuda.empty_cache()
        lr = sched1.step(ep)
        t0 = time.time()
        tl, ta = train_epoch(model, tr_loader, criterion, opt1, scaler, device)
        vl, va, _, _ = val_epoch(model, va_loader, criterion, device)

        for k,v in zip(['tr_loss','tr_acc','va_loss','va_acc'],[tl,ta,vl,va]):
            history[k].append(v)

        star = ''
        if va > best_acc:
            best_acc = va; best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_PATH); star = ' ★'

        print(f'  [{ep+1:02d}/{P1_EPOCHS}]  '
              f'Tr {ta*100:5.2f}% / {tl:.4f}  '
              f'Va {va*100:5.2f}% / {vl:.4f}  '
              f'LR={lr:.2e}  {gpu_mem()}  {time.time()-t0:.0f}s{star}')

    print(f'\n  Phase 1 best val acc: {best_acc*100:.2f}%')

    # ── Phase 2 — Unfreeze last 3 blocks ─────────────────────
    print('\n' + '━'*62)
    print('  Phase 2 — Unfreeze last 3 backbone blocks')
    print('━'*62)
    model = unfreeze(model, unfreeze_blocks=3)

    opt2 = optim.AdamW([
        {'params': [p for n,p in model.named_parameters()
                    if 'classifier' not in n and p.requires_grad], 'lr': LR/10},
        {'params': model.classifier.parameters(), 'lr': LR},
    ], weight_decay=WEIGHT_DECAY)
    sched2 = WarmupCosine(opt2, warmup_epochs=WARMUP_EPOCHS,
                          total_epochs=EPOCHS//2, min_lr=MIN_LR, base_lr=LR)
    no_improve = 0

    for ep in range(EPOCHS//2):
        torch.cuda.empty_cache()
        lr = sched2.step(ep)
        t0 = time.time()
        tl, ta = train_epoch(model, tr_loader, criterion, opt2, scaler, device)
        vl, va, _, _ = val_epoch(model, va_loader, criterion, device)

        for k,v in zip(['tr_loss','tr_acc','va_loss','va_acc'],[tl,ta,vl,va]):
            history[k].append(v)

        star = ''
        if va > best_acc:
            best_acc = va; best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_PATH); star = ' ★'; no_improve = 0
        else:
            no_improve += 1

        print(f'  [{P1_EPOCHS+ep+1:02d}/{P1_EPOCHS+EPOCHS//2}]  '
              f'Tr {ta*100:5.2f}% / {tl:.4f}  '
              f'Va {va*100:5.2f}% / {vl:.4f}  '
              f'LR={lr:.2e}  {gpu_mem()}  {time.time()-t0:.0f}s{star}')

        if no_improve >= EARLY_STOP:
            print(f'  Early stop — no improvement for {EARLY_STOP} epochs'); break

    # ── Phase 3 — Full unfreeze ───────────────────────────────
    print('\n' + '━'*62)
    print('  Phase 3 — Full backbone unfreeze (final push to 97%+)')
    print('━'*62)
    model = unfreeze(model, unfreeze_blocks=99)

    opt3 = optim.AdamW([
        {'params': [p for n,p in model.named_parameters()
                    if 'classifier' not in n and p.requires_grad], 'lr': LR/100},
        {'params': model.classifier.parameters(), 'lr': LR/10},
    ], weight_decay=WEIGHT_DECAY)
    sched3 = WarmupCosine(opt3, warmup_epochs=1, total_epochs=EPOCHS//4,
                          min_lr=MIN_LR/10, base_lr=LR/100)
    no_improve = 0

    for ep in range(EPOCHS//4):
        torch.cuda.empty_cache()
        lr = sched3.step(ep)
        t0 = time.time()
        tl, ta = train_epoch(model, tr_loader, criterion, opt3, scaler, device)
        vl, va, _, _ = val_epoch(model, va_loader, criterion, device)

        for k,v in zip(['tr_loss','tr_acc','va_loss','va_acc'],[tl,ta,vl,va]):
            history[k].append(v)

        star = ''
        if va > best_acc:
            best_acc = va; best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_PATH); star = ' ★'; no_improve = 0
        else:
            no_improve += 1

        print(f'  [{P1_EPOCHS+EPOCHS//2+ep+1:02d}]  '
              f'Tr {ta*100:5.2f}% / {tl:.4f}  '
              f'Va {va*100:5.2f}% / {vl:.4f}  '
              f'LR={lr:.2e}  {gpu_mem()}  {time.time()-t0:.0f}s{star}')

        if no_improve >= EARLY_STOP:
            print(f'  Early stop'); break

    print(f'\n  ✅ Best Val Accuracy: {best_acc*100:.2f}%')
    print(f'  Saved → {MODEL_PATH}')
    model.load_state_dict(best_wts)
    return model, history, best_acc
