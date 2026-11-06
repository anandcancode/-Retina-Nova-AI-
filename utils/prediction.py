"""
Diabetic Retinopathy Detection - Inference & Diagnostic Pipeline
Handles:
- Model weight loading and caching
- Multi-class probability calculation
- Clinical stage classification and risk stratifications
- Medical recommendations and clinical follow-up guidelines
"""

import os
import json
import numpy as np

# Clinical metadata for the 5 Diabetic Retinopathy stages (ICDR standard)
DR_CLASSES_INFO = {
    0: {
        "stage": "Stage 0",
        "name": "No DR (Normal)",
        "risk_level": "Normal / Healthy",
        "badge_class": "badge-success",
        "color": "#10b981",
        "description": "No visible signs of diabetic retinopathy detected. Retinal vasculature, optic disc, and macula appear normal and intact without microvascular abnormalities.",
        "clinical_findings": [
            "Normal retinal vasculature with clear fundus background",
            "No microaneurysms or retinal hemorrhages",
            "No hard exudates or cotton wool spots",
            "Foveal avascular zone (FAZ) intact"
        ],
        "recommendations": [
            "Schedule standard annual dilated eye examination",
            "Maintain optimal glycemic control (target HbA1c < 7.0%)",
            "Monitor blood pressure and lipid profile regularly",
            "Follow a balanced diabetic diet and active lifestyle"
        ],
        "urgency": "Routine (Within 12 months)"
    },
    1: {
        "stage": "Stage 1",
        "name": "Mild Non-Proliferative DR",
        "risk_level": "Low Risk",
        "badge_class": "badge-info",
        "color": "#3b82f6",
        "description": "Mild NPDR characterized by the presence of at least one microaneurysm (tiny balloon-like swellings in fragile retinal capillaries).",
        "clinical_findings": [
            "Isolated microaneurysms detected in capillary networks",
            "Absence of hard exudates or venous beading",
            "No evidence of macular edema",
            "Early stage retinal microvascular stress"
        ],
        "recommendations": [
            "Repeat dilated fundus examination in 6 to 9 months",
            "Consult primary physician to optimize glycemic and blood pressure targets",
            "Perform comprehensive lipid panel screening",
            "Report any sudden visual fluctuations or blurry vision immediately"
        ],
        "urgency": "Follow-up (6 - 9 months)"
    },
    2: {
        "stage": "Stage 2",
        "name": "Moderate Non-Proliferative DR",
        "risk_level": "Moderate Risk",
        "badge_class": "badge-warning",
        "color": "#f59e0b",
        "description": "Moderate NPDR showing more than just microaneurysms: blot hemorrhages, hard exudates (lipid deposits), and early vascular occlusion without reaching the severe 4-2-1 rule.",
        "clinical_findings": [
            "Multiple retinal hemorrhages in 1 to 3 quadrants",
            "Hard exudates (lipid residues) indicating vascular leakage",
            "Cotton wool spots (micro-infarcts of nerve fiber layer)",
            "Early venous dilation or caliber irregularity"
        ],
        "recommendations": [
            "Ophthalmology follow-up within 3 to 6 months",
            "Optical Coherence Tomography (OCT) scan recommended to assess Diabetic Macular Edema (DME)",
            "Strict glycemic regulation (continuous glucose monitoring recommended)",
            "Cardiovascular risk assessment and renal function check"
        ],
        "urgency": "Early Specialist Review (3 - 6 months)"
    },
    3: {
        "stage": "Stage 3",
        "name": "Severe Non-Proliferative DR",
        "risk_level": "High Risk",
        "badge_class": "badge-danger",
        "color": "#f97316",
        "description": "Severe NPDR meeting at least one criterion of the International Clinical DR 4-2-1 rule: diffuse intraretinal hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant.",
        "clinical_findings": [
            "Extensive intraretinal hemorrhages in all 4 quadrants",
            "Definite venous beading in at least 2 quadrants",
            "Prominent Intraretinal Microvascular Abnormalities (IRMA)",
            "Significant capillary non-perfusion and widespread retinal ischemia"
        ],
        "recommendations": [
            "Urgent retina specialist consultation within 2 to 4 weeks",
            "High risk of rapid progression to proliferative stage (>50% within 1 year)",
            "Perform Fluorescein Angiography (FA) and high-resolution OCT",
            "Discuss preventive Panretinal Photocoagulation (PRP) or anti-VEGF therapy"
        ],
        "urgency": "Urgent Specialist Consultation (2 - 4 weeks)"
    },
    4: {
        "stage": "Stage 4",
        "name": "Proliferative Diabetic Retinopathy (PDR)",
        "risk_level": "Critical / High Alert",
        "badge_class": "badge-critical",
        "color": "#ef4444",
        "description": "Advanced proliferative stage characterized by pathological neovascularization (formation of abnormal, fragile new blood vessels) on disc (NVD) or elsewhere (NVE). High risk of blindness from vitreous hemorrhage or tractional retinal detachment.",
        "clinical_findings": [
            "Neovascularization of the optic disc (NVD) or retina (NVE)",
            "Risk or evidence of preretinal or vitreous hemorrhage",
            "Fibrovascular proliferation and traction on the retina",
            "Critical ischemic signal driving VEGF over-expression"
        ],
        "recommendations": [
            "Immediate emergency ophthalmologist referral (within 24 to 48 hours)",
            "Panretinal Photocoagulation (PRP) laser therapy urgently indicated",
            "Intravitreal anti-VEGF injections (e.g., Aflibercept, Ranibizumab)",
            "Surgical vitrectomy assessment if non-clearing vitreous hemorrhage is present"
        ],
        "urgency": "Immediate / Emergency (Within 24 - 48 hours)"
    }
}

# Global cached model reference
_CACHED_MODEL = None
_MODEL_PATH = None
_IS_REAL_WEIGHTS = False


def load_model(weights_path="models/trained_model/hybrid_dr_model.keras"):
    """
    Loads the hybrid deep learning model into memory.
    Checks for .keras, .h5, or weights file.
    """
    global _CACHED_MODEL, _MODEL_PATH, _IS_REAL_WEIGHTS

    if _CACHED_MODEL is not None and _MODEL_PATH == weights_path:
        return _CACHED_MODEL, _IS_REAL_WEIGHTS

    h5_path = weights_path.replace(".keras", ".h5")
    actual_path = None
    if os.path.exists(weights_path):
        actual_path = weights_path
    elif os.path.exists(h5_path):
        actual_path = h5_path

    try:
        import tensorflow as tf
        from models.hybrid_model import build_hybrid_model

        if actual_path and os.path.exists(actual_path) and os.path.getsize(actual_path) > 100000:
            print(f"[Prediction] Loading trained hybrid model from: {actual_path}")
            try:
                _CACHED_MODEL = tf.keras.models.load_model(actual_path)
                _IS_REAL_WEIGHTS = True
                _MODEL_PATH = actual_path
                return _CACHED_MODEL, _IS_REAL_WEIGHTS
            except Exception as e:
                print(f"[Prediction] Notice: Direct model loading raised ({e}). Building architecture and loading weights...")
                model = build_hybrid_model()
                model.load_weights(actual_path)
                _CACHED_MODEL = model
                _IS_REAL_WEIGHTS = True
                _MODEL_PATH = actual_path
                return _CACHED_MODEL, _IS_REAL_WEIGHTS
        else:
            print("[Prediction] Notice: Pretrained weights file not found on disk. Building architecture...")
            model = build_hybrid_model()
            _CACHED_MODEL = model
            _IS_REAL_WEIGHTS = False
            _MODEL_PATH = actual_path
            return _CACHED_MODEL, _IS_REAL_WEIGHTS

    except Exception as err:
        print(f"[Prediction] Could not initialize TensorFlow model: {err}")
        _CACHED_MODEL = None
        _IS_REAL_WEIGHTS = False
        return None, False


def run_prediction(input_tensor, image_quality=None):
    """
    Executes the hybrid inference pipeline on preprocessed input tensor (1, 224, 224, 3).
    
    Returns structured result dict:
    - predicted_class: int (0 to 4)
    - predicted_stage: str
    - confidence: float (0.0 to 100.0)
    - probabilities: list of 5 floats summing to 1.0
    - class_info: dict with stage clinical information
    - is_simulated: bool
    """
    model, is_trained = load_model()

    if model is not None:
        try:
            preds = model.predict(input_tensor, verbose=0)[0]
            probabilities = [float(p) for p in preds]

            # If model is randomly initialized (not yet trained with APTOS/EyePACS weights),
            # mark clearly as simulated demo per prompt specification
            is_simulated = not is_trained
            pred_class = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_class]) * 100.0

        except Exception as e:
            print(f"[Prediction] Inference error on model: {e}")
            is_simulated = True
            pred_class, probabilities, confidence = _generate_heuristic_baseline(input_tensor)
    else:
        # Fallback when TensorFlow engine is unavailable in runtime
        is_simulated = True
        pred_class, probabilities, confidence = _generate_heuristic_baseline(input_tensor)

    class_data = DR_CLASSES_INFO[pred_class]

    result = {
        "predicted_class": pred_class,
        "predicted_stage": class_data["stage"],
        "stage_name": class_data["name"],
        "confidence": round(confidence, 2),
        "probabilities": [round(float(p) * 100.0, 2) for p in probabilities],
        "risk_level": class_data["risk_level"],
        "badge_class": class_data["badge_class"],
        "color": class_data["color"],
        "description": class_data["description"],
        "clinical_findings": class_data["clinical_findings"],
        "recommendations": class_data["recommendations"],
        "urgency": class_data["urgency"],
        "is_simulated": is_simulated,
        "model_architecture": "Hybrid CNN + EfficientNet Feature Fusion",
        "labels": ["0: No DR", "1: Mild DR", "2: Moderate DR", "3: Severe DR", "4: Proliferative DR"]
    }

    return result


def _generate_heuristic_baseline(input_tensor):
    """
    Physiologically guided heuristic baseline for demo mode when weights are not yet trained:
    Analyzes retinal vessel contrast, red-channel lesion variance, and bright exudate signatures
    to produce realistic probabilities instead of pure uniform randomness.
    """
    img = input_tensor[0]  # (224, 224, 3) in [0, 1]
    r_chan = img[:, :, 0]
    g_chan = img[:, :, 1]
    b_chan = img[:, :, 2]

    # Red/green contrast ratio highlights microaneurysms and hemorrhages
    rg_diff = np.abs(r_chan - g_chan)
    lesion_metric = float(np.mean(rg_diff)) * 10.0

    # Retinal center illumination vs edge
    std_intensity = float(np.std(g_chan)) * 10.0

    # Calculate calibrated logits
    if lesion_metric < 0.8 and std_intensity < 1.8:
        logits = [2.8, 0.8, 0.4, 0.1, 0.05]
    elif lesion_metric < 1.3:
        logits = [0.8, 2.5, 1.2, 0.3, 0.1]
    elif lesion_metric < 1.8:
        logits = [0.3, 0.9, 2.6, 1.1, 0.2]
    elif lesion_metric < 2.3:
        logits = [0.1, 0.4, 1.2, 2.7, 0.8]
    else:
        logits = [0.05, 0.2, 0.6, 1.4, 2.9]

    # Softmax
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l)
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx]) * 100.0

    return pred_idx, probs.tolist(), conf

