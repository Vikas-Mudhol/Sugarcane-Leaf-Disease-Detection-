# =============================================================
#  predict.py — Single image prediction
#
#  Validation pipeline:
#    1. Green heuristic  (fast colour check)
#    2. CLIP validation  (confirms sugarcane leaf specifically)
#    3. Disease model    (EfficientNet-B4 + TTA)
#    4. Confidence gate  (rejects uncertain predictions)
#
#  Usage:
#    python predict.py --image path/to/leaf.jpg
# =============================================================
import argparse, io, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch

from config import CLASSES, NUM_CLASSES, MODEL_PATH, SAVE_DIR
from model import build_model
from dataset import val_tf, tta_tfs

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRES = 0.55
GREEN_MIN  = 0.28

# ── CLIP validator ────────────────────────────────────────────
_clip_model = _clip_proc = None

SUGARCANE_PROMPTS = [
    'a sugarcane leaf', 'a sugarcane plant leaf',
    'a close-up of a sugarcane leaf with disease',
    'a healthy sugarcane leaf',
]
REJECT_PROMPTS = [
    'a human face', 'a person', 'a rose leaf', 'a mango leaf',
    'a banana leaf', 'a random object', 'grass', 'food',
    'an animal', 'a building', 'a maize leaf', 'a wheat leaf',
]
ALL_PROMPTS = SUGARCANE_PROMPTS + REJECT_PROMPTS
SC_THRESHOLD = 0.38


def _load_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print('  Loading CLIP validator...')
        _clip_model = CLIPModel.from_pretrained(
            'openai/clip-vit-base-patch32', use_safetensors=True).to(DEVICE)
        _clip_proc  = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        _clip_model.eval()
        print('  CLIP ready.')
    return _clip_model, _clip_proc


@torch.no_grad()
def clip_validate(image_path):
    model, proc = _load_clip()
    image  = Image.open(image_path).convert('RGB')
    inputs = proc(text=ALL_PROMPTS, images=image,
                  return_tensors='pt', padding=True).to(DEVICE)
    probs  = model(**inputs).logits_per_image.softmax(dim=1).cpu().squeeze().numpy()
    sc_score = float(probs[:len(SUGARCANE_PROMPTS)].sum())
    best_idx = int(np.argmax(probs))
    best_lbl = ALL_PROMPTS[best_idx]
    best_p   = float(probs[best_idx])
    if sc_score >= SC_THRESHOLD:
        return True, sc_score, 'OK'
    return False, sc_score, f'Best match: "{best_lbl}" ({best_p*100:.1f}%). Not a sugarcane leaf.'


def green_check(img_np):
    f = img_np.astype(np.float32)
    r,g,b = f[:,:,0], f[:,:,1], f[:,:,2]
    gr = (g / (r+g+b+1e-6)).mean()
    if f.std() < 15:
        return False, 'Image is blank or uniform.'
    if gr < GREEN_MIN:
        return False, f'Not enough green (ratio={gr:.2f}). Probably not a plant.'
    return True, 'OK'


def load_model():
    model = build_model(NUM_CLASSES).to(DEVICE)
    # unfreeze all for inference
    for p in model.parameters():
        p.requires_grad = False
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(image_path, model, use_tta=True):
    img_np = np.array(Image.open(image_path).convert('RGB'))

    print('\n── Validation ────────────────────────────────')
    # Step 1
    ok, reason = green_check(img_np)
    print(f'  [1/3] Green check ... {"PASS" if ok else "FAIL"}')
    if not ok:
        return None, None, f'Rejected: {reason}'

    # Step 2
    ok, sc_score, reason = clip_validate(image_path)
    print(f'  [2/3] CLIP check  ... {"PASS" if ok else "FAIL"} (score={sc_score:.2f})')
    if not ok:
        return None, None, f'Rejected: {reason}'

    # Step 3
    print('  [3/3] Disease model ...')
    tfms = tta_tfs if use_tta else [val_tf]
    all_p = []
    for tfm in tfms:
        t = tfm(image=img_np)['image'].unsqueeze(0).to(DEVICE)
        with torch.amp.autocast(device_type='cuda'):
            p = torch.softmax(model(t), dim=1).cpu().squeeze().numpy()
        all_p.append(p)
    avg  = np.mean(all_p, axis=0)
    idx  = int(np.argmax(avg))
    conf = float(avg[idx])

    if conf < CONF_THRES:
        return None, avg, f'Low confidence ({conf*100:.1f}%). Use a clearer leaf image.'

    return CLASSES[idx], avg, None


def visualise(image_path, disease, probs, sc_score, error=None):
    img = np.array(Image.open(image_path).convert('RGB'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(img); axes[0].axis('off')
    axes[0].set_title('Input Image', fontweight='bold')
    if sc_score is not None:
        axes[0].text(0.02, 0.04, f'CLIP: {sc_score*100:.1f}%',
                     transform=axes[0].transAxes, fontsize=8, color='white',
                     bbox=dict(facecolor='#1565C0', alpha=0.85, boxstyle='round'))

    if error:
        axes[1].text(0.5, 0.5, f'⚠  {error}', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=11,
                     color='#B71C1C', fontweight='bold',
                     bbox=dict(facecolor='#FFEBEE', edgecolor='#B71C1C',
                               boxstyle='round,pad=0.6'))
        axes[1].axis('off')
        plt.suptitle('⚠  Invalid Input', color='#B71C1C', fontsize=13, fontweight='bold')
    else:
        pidx   = CLASSES.index(disease)
        colors = ['#4CAF50' if i==pidx else '#90A4AE' for i in range(len(CLASSES))]
        axes[1].barh(CLASSES, probs*100, color=colors, edgecolor='black', lw=0.4)
        axes[1].axvline(CONF_THRES*100, ls='--', color='red', lw=1, alpha=0.6)
        axes[1].set_xlabel('Confidence (%)'); axes[1].set_xlim(0,115)
        axes[1].set_title('Class Probabilities', fontweight='bold')
        for i,p in enumerate(probs):
            axes[1].text(p*100+0.5, i, f'{p*100:.1f}%', va='center', fontsize=9)
        color = '#2E7D32' if disease=='Healthy' else '#B71C1C'
        label = 'Healthy Leaf ✔' if disease=='Healthy' else f'Disease: {disease}'
        plt.suptitle(label, color=color, fontsize=14, fontweight='bold')

    plt.tight_layout()
    out = f'{SAVE_DIR}/prediction.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'\n  Result : {disease or "REJECTED"}')
    if probs is not None and disease:
        print(f'  Confidence : {probs[CLASSES.index(disease)]*100:.2f}%')
    print(f'  Saved  → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--no-tta', action='store_true')
    args = ap.parse_args()

    m = load_model()
    disease, probs, error = predict(args.image, m, not args.no_tta)
    sc_score = None
    if not error:
        _, sc_score, _ = clip_validate(args.image)
    visualise(args.image, disease, probs, sc_score, error)
