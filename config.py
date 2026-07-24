# =============================================================
#  config.py — RTX 4060 8GB optimised
# =============================================================
import os

# ── Dataset ───────────────────────────────────────────────────
DATA_DIR    = r'D:\sugarcane_v2\Dataset'   # ← change this
CLASSES     = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
SEED        = 42

# ── DataLoader ────────────────────────────────────────────────
BATCH_SIZE  = 32    # RTX 4060 8GB with AMP — do NOT go higher
NUM_WORKERS = 0     # Windows must stay 0

# ── Training ──────────────────────────────────────────────────
EPOCHS          = 60     # ~5200 images / 5 classes — enough epochs to fully converge
LR              = 1e-4   # slightly lower LR — more data = more stable gradients
MIN_LR          = 1e-7
WEIGHT_DECAY    = 5e-5   # less regularisation needed — data itself regularises
LABEL_SMOOTHING = 0.08   # reduce smoothing — large dataset learns real boundaries
DROPOUT         = 0.35   # reduce dropout — more data reduces overfitting risk
WARMUP_EPOCHS   = 5      # longer warmup for larger dataset
EARLY_STOP      = 12     # more patience — large datasets converge slower

# ── Augmentation ──────────────────────────────────────────────
MIXUP_ALPHA   = 0.3
CUTMIX_ALPHA  = 0.8
MIXUP_PROB    = 0.5      # probability of applying mixup/cutmix

# ── Output ────────────────────────────────────────────────────
SAVE_DIR   = 'outputs'
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
