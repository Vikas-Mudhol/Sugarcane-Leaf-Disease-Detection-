# =============================================================
#  leaf_validator.py — CLIP-based sugarcane leaf validator
#
#  Uses OpenAI CLIP to verify the image is a sugarcane leaf
#  before passing it to the disease classifier.
#  Downloads ~600MB model on first run (cached after that).
# =============================================================

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Positive prompts — what we WANT
SUGARCANE_PROMPTS = [
    "a sugarcane leaf",
    "a sugarcane plant leaf with disease",
    "a healthy sugarcane leaf",
    "a close-up of a sugarcane leaf",
]

# Negative prompts — what we REJECT
REJECT_PROMPTS = [
    "a human face",
    "a person",
    "a rose leaf",
    "a mango leaf",
    "a banana leaf",
    "a tree leaf",
    "a random object",
    "grass",
    "food",
    "an animal",
    "a building",
    "a document or paper",
]

ALL_PROMPTS = SUGARCANE_PROMPTS + REJECT_PROMPTS

# Minimum fraction of softmax score that must go to sugarcane prompts
SUGARCANE_SCORE_THRESHOLD = 0.40


_clip_model     = None
_clip_processor = None

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print('Loading CLIP validator (first run: ~600MB download, cached after)...')
        _clip_model     = CLIPModel.from_pretrained(
                            "openai/clip-vit-base-patch32",
                            use_safetensors=True
                          ).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print('CLIP loaded.')
    return _clip_model, _clip_processor


@torch.no_grad()
def validate_sugarcane_leaf(image_path: str) -> tuple[bool, float, str]:
    """
    Check whether the image is a sugarcane leaf using CLIP.

    Returns:
        is_valid  (bool)   — True if image passes as sugarcane leaf
        sc_score  (float)  — Combined softmax score for sugarcane prompts (0–1)
        reason    (str)    — Human-readable explanation
    """
    model, processor = _load_clip()

    image  = Image.open(image_path).convert('RGB')
    inputs = processor(
        text   = ALL_PROMPTS,
        images = image,
        return_tensors = 'pt',
        padding = True
    ).to(DEVICE)

    outputs = model(**inputs)
    # logits_per_image shape: (1, num_prompts)
    probs   = outputs.logits_per_image.softmax(dim=1).cpu().squeeze().numpy()

    # Sum probabilities for all sugarcane prompts
    n_sc        = len(SUGARCANE_PROMPTS)
    sc_score    = float(probs[:n_sc].sum())
    reject_score= float(probs[n_sc:].sum())

    # Find the single best matching prompt for the rejection message
    best_idx    = int(np.argmax(probs))
    best_prompt = ALL_PROMPTS[best_idx]
    best_prob   = float(probs[best_idx])

    if sc_score >= SUGARCANE_SCORE_THRESHOLD:
        return True, sc_score, f'Sugarcane leaf confirmed (score={sc_score:.2f})'

    return (
        False,
        sc_score,
        f'Not a sugarcane leaf.\n'
        f'Best match: "{best_prompt}" ({best_prob*100:.1f}%)\n'
        f'Sugarcane score: {sc_score*100:.1f}%  (need ≥{SUGARCANE_SCORE_THRESHOLD*100:.0f}%)'
    )
