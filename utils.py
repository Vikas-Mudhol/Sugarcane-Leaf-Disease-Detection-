# =============================================================
#  utils.py
# =============================================================
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mixup_data(x, y, alpha=0.3):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def cutmix_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)
    B,C,H,W = x.size()
    idx = torch.randperm(B).to(x.device)
    cut_rat = np.sqrt(1 - lam)
    cut_h, cut_w = int(H*cut_rat), int(W*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1,x2 = max(cx-cut_w//2,0), min(cx+cut_w//2,W)
    y1,y2 = max(cy-cut_h//2,0), min(cy+cut_h//2,H)
    mixed = x.clone()
    mixed[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    lam = 1-(x2-x1)*(y2-y1)/(W*H)
    return mixed, y, y[idx], lam

def mixed_criterion(criterion, pred, ya, yb, lam):
    return lam*criterion(pred,ya) + (1-lam)*criterion(pred,yb)
