# =============================================================
#  app.py — Flask frontend
#  Run:  python app.py
#  Open: http://localhost:5000
# =============================================================
import os, io, base64, traceback
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch

from config import CLASSES, NUM_CLASSES, MODEL_PATH
from model import build_model
from dataset import tta_tfs
from predict import green_check, clip_validate, CONF_THRES

app    = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('static/uploads', exist_ok=True)

# ── Treatment database ────────────────────────────────────────
TREATMENTS = {
    "Healthy": {
        "status": "healthy",
        "description": "Your sugarcane crop is healthy! Maintain current practices for maximum yield.",
        "fertilizer": [
            {"icon":"🌿","name":"Nitrogen (N)",       "dose":"150–200 kg/ha","timing":"Split: planting + 30 + 60 days"},
            {"icon":"🌱","name":"Phosphorus (P₂O₅)", "dose":"60–80 kg/ha",  "timing":"Basal at planting"},
            {"icon":"💧","name":"Potassium (K₂O)",   "dose":"100–120 kg/ha","timing":"Split: planting + 60 days"},
        ],
        "soil": [
            "Maintain pH 6.0–7.5",
            "Apply compost 10 t/ha before planting",
            "Deep plough to 30–40 cm for root penetration",
            "Ensure good drainage — avoid waterlogging",
        ],
        "care": [
            "Irrigate every 10–15 days based on rainfall",
            "Earthing-up at 45 and 90 days after planting",
            "Trash removal at 120–150 days",
            "Weekly monitoring for early disease signs",
        ],
        "pesticide": [],
        "warning": None,
    },
    "Mosaic": {
        "status": "disease",
        "description": "Sugarcane Mosaic Virus (SCMV) — yellow-green mosaic on leaves, spread by aphids. Act immediately.",
        "fertilizer": [
            {"icon":"🛡️","name":"Potassium Silicate",  "dose":"5 kg/ha",    "timing":"Foliar every 15 days — boosts immunity"},
            {"icon":"⚡","name":"Zinc Sulphate",       "dose":"25 kg/ha",   "timing":"Soil application — cell wall strength"},
            {"icon":"🌿","name":"Urea (foliar)",       "dose":"2% solution","timing":"Every 10 days — compensates growth loss"},
        ],
        "soil": [
            "Sterilise soil with Formalin 2% before replanting",
            "Remove and burn all infected residues",
            "Crop rotation with groundnut or pulses",
            "Avoid monoculture — reduces virus reservoir",
        ],
        "care": [
            "🚨 Remove infected plants immediately",
            "Use certified virus-free seed cane only",
            "Plant resistant varieties: Co 86032, Co 94012",
            "Disinfect tools with 1% Sodium Hypochlorite",
        ],
        "pesticide": [
            {"name":"Imidacloprid 17.8 SL","dose":"0.5 ml/L","target":"Aphid vector control",    "frequency":"Every 15 days"},
            {"name":"Thiamethoxam 25 WG",  "dose":"0.3 g/L", "target":"Whitefly & aphid control","frequency":"Every 21 days"},
            {"name":"Neem oil 5%",         "dose":"5 ml/L",  "target":"Organic aphid repellent", "frequency":"Every 10 days"},
        ],
        "warning": "No chemical cure for mosaic virus. Focus entirely on vector control and roguing infected plants.",
    },
    "RedRot": {
        "status": "disease",
        "description": "Red Rot (Colletotrichum falcatum) — red discolouration with white patches inside stalks. Most destructive sugarcane disease.",
        "fertilizer": [
            {"icon":"🍄","name":"Trichoderma + FYM",   "dose":"2.5 kg + 50 kg/acre","timing":"Soil incorporation 2 weeks before planting"},
            {"icon":"🛡️","name":"Potassium Phosphonate","dose":"3 ml/L",             "timing":"Foliar — systemic resistance booster"},
            {"icon":"💪","name":"Calcium Nitrate",     "dose":"20 kg/ha",            "timing":"Soil — strengthens cell walls"},
        ],
        "soil": [
            "Drench soil with Copper Oxychloride 0.3%",
            "Apply Trichoderma viride 4 kg/acre with compost",
            "Avoid waterlogging — Red Rot spreads in wet conditions",
            "Burn infected stools — never compost them",
        ],
        "care": [
            "🚨 Burn all infected stalks immediately",
            "Hot water treat setts at 50°C for 2 hours",
            "Treat setts with Carbendazim 0.1% for 15 mins",
            "Resistant varieties: Co 86032, Co 99004",
            "Never keep ratoon from infected fields",
        ],
        "pesticide": [
            {"name":"Carbendazim 50 WP",       "dose":"1 g/L",  "target":"Primary Red Rot fungicide",       "frequency":"Every 10 days, 3 sprays"},
            {"name":"Thiophanate-methyl 70 WP", "dose":"1.5 g/L","target":"Systemic soil drench + spray",   "frequency":"Every 15 days"},
            {"name":"Propiconazole 25 EC",      "dose":"1 ml/L", "target":"Preventive + curative fungicide","frequency":"Every 21 days"},
            {"name":"Copper Oxychloride 50 WP", "dose":"3 g/L",  "target":"Soil drench + protective spray","frequency":"Every 14 days"},
        ],
        "warning": "Red Rot spreads through infected setts, soil, and irrigation water. Entire infected plots may need to be abandoned for one season.",
    },
    "Rust": {
        "status": "disease",
        "description": "Sugarcane Rust (Puccinia melanocephala) — orange-brown pustules on leaves, spreads by wind rapidly.",
        "fertilizer": [
            {"icon":"💪","name":"Potassium Chloride","dose":"80–100 kg/ha","timing":"Adequate K reduces rust severity"},
            {"icon":"🛡️","name":"Silicon Gel",       "dose":"150 kg/ha",  "timing":"Soil — strengthens leaf cuticle"},
            {"icon":"⚗️","name":"Sulphur 80 WG",    "dose":"3 g/L",      "timing":"Foliar — fungicide + micronutrient"},
        ],
        "soil": [
            "Ensure good drainage — wet soil worsens rust",
            "Apply rice husk ash 500 kg/ha for silicon",
            "Avoid excess Nitrogen — lush growth is vulnerable",
            "Incorporate organic matter to improve drainage",
        ],
        "care": [
            "Remove and burn infected leaves immediately",
            "Avoid overhead irrigation — wet leaves spread spores",
            "Wider row spacing for air circulation",
            "Resistant varieties: CoC 671, Co 86032",
            "Monitor weekly during post-monsoon humidity",
        ],
        "pesticide": [
            {"name":"Propiconazole 25 EC","dose":"1 ml/L",  "target":"Best systemic rust fungicide",         "frequency":"Every 15 days, 2–3 sprays"},
            {"name":"Mancozeb 75 WP",     "dose":"2.5 g/L", "target":"Protective broad-spectrum fungicide",  "frequency":"Every 10–14 days"},
            {"name":"Tebuconazole 25 EC", "dose":"1 ml/L",  "target":"Curative for established rust",        "frequency":"Every 21 days"},
            {"name":"Hexaconazole 5 EC",  "dose":"2 ml/L",  "target":"Preventive + curative, systemic action","frequency":"Every 14 days"},
        ],
        "warning": "Rust spreads fast through wind. Start fungicide at first sign of orange pustules — don't delay.",
    },
    "Yellow": {
        "status": "disease",
        "description": "Yellow Leaf Disease (SCYLV) — yellowing of leaf midrib, transmitted by aphids and whiteflies.",
        "fertilizer": [
            {"icon":"🌿","name":"Urea (foliar)",        "dose":"2% solution", "timing":"Every 10 days — compensates N loss"},
            {"icon":"💛","name":"Ferrous Sulphate",     "dose":"0.5% solution","timing":"Foliar — corrects iron deficiency"},
            {"icon":"🌾","name":"Magnesium Sulphate",   "dose":"1% solution", "timing":"Foliar — corrects Mg deficiency"},
            {"icon":"⚡","name":"NPK 19:19:19 (soluble)","dose":"5 g/L",      "timing":"Foliar every 15 days for recovery"},
        ],
        "soil": [
            "Test soil for Fe, Mn, Zn, Mg deficiencies",
            "Apply FYM 25 t/ha before planting",
            "Apply ZnSO₄ 25 kg/ha + FeSO₄ 25 kg/ha as basal",
            "Ensure good drainage — virus worsens in stressed plants",
        ],
        "care": [
            "🚨 Remove severely infected plants immediately",
            "Use Yellow Leaf Disease-indexed certified seed cane",
            "Install yellow sticky traps 8–10 per acre",
            "Control aphids and whiteflies aggressively",
            "Avoid planting near infected ratoon crops",
        ],
        "pesticide": [
            {"name":"Thiamethoxam 25 WG",         "dose":"0.3 g/L","target":"Best for aphid/whitefly control","frequency":"Every 21 days"},
            {"name":"Imidacloprid 17.8 SL",       "dose":"0.5 ml/L","target":"Vector control",               "frequency":"Every 15 days"},
            {"name":"Spirotetramat 150 OD",        "dose":"1.5 ml/L","target":"Translaminar aphicide",       "frequency":"Every 21 days"},
            {"name":"Neem oil (Azadirachtin 1%)", "dose":"5 ml/L",  "target":"Organic repellent",            "frequency":"Every 10 days"},
        ],
        "warning": "No direct cure for SCYLV. Early vector control and roguing infected plants is the only strategy.",
    },
}

# ── Load model ────────────────────────────────────────────────
print('\nLoading EfficientNet-B4...')
_model = build_model(NUM_CLASSES).to(DEVICE)
for p in _model.parameters(): p.requires_grad = False
ckpt   = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
_model.eval()
print('Model ready.\n')


@torch.no_grad()
def run_prediction(image_path):
    img_np = np.array(Image.open(image_path).convert('RGB'))

    # Step 1: green check
    ok, reason = green_check(img_np)
    if not ok:
        return {'valid': False, 'error': f'Not a plant: {reason}'}

    # Step 2: CLIP
    ok, sc_score, reason = clip_validate(image_path)
    if not ok:
        return {'valid': False, 'error': f'Not a sugarcane leaf. {reason}'}

    # Step 3: inference + TTA
    all_p = []
    for tfm in tta_tfs:
        t = tfm(image=img_np)['image'].unsqueeze(0).to(DEVICE)
        with torch.amp.autocast(device_type='cuda'):
            p = torch.softmax(_model(t), dim=1).cpu().squeeze().numpy()
        all_p.append(p)
    avg  = np.mean(all_p, axis=0)
    idx  = int(np.argmax(avg))
    conf = float(avg[idx])
    torch.cuda.empty_cache()

    # Step 4: confidence gate
    if conf < CONF_THRES:
        return {'valid': False,
                'error': f'Low confidence ({conf*100:.1f}%). Use a clearer leaf image.'}

    # Sharpened image as base64
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    disease = CLASSES[idx]
    return {
        'valid':      True,
        'disease':    disease,
        'confidence': round(conf*100, 2),
        'all_confs':  {CLASSES[i]: round(float(avg[i])*100, 2) for i in range(len(CLASSES))},
        'clip_score': round(sc_score*100, 1),
        'treatment':  TREATMENTS[disease],
        'image_b64':  img_b64,
    }


@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'valid': False, 'error': 'No image uploaded.'})
    f = request.files['image']
    if not f.filename:
        return jsonify({'valid': False, 'error': 'No file selected.'})
    path = os.path.join('static/uploads', f.filename)
    f.save(path)
    try:
        result = run_prediction(path)
    except Exception as e:
        traceback.print_exc()
        result = {'valid': False, 'error': str(e)}
    finally:
        if os.path.exists(path): os.remove(path)
    return jsonify(result)


if __name__ == '__main__':
    print('Open → http://localhost:5000\n')
    app.run(debug=False, port=5000)
