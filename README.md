# RetinaNova AI: Diabetic Retinopathy Detection Using Hybrid Deep Learning

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A clinical-grade, AI-assisted web application designed to evaluate digital retinal fundus photographs and classify Diabetic Retinopathy (DR) severity into five distinct clinical stages using a **Dual-Branch Hybrid Deep Learning** architecture.

---

## 📌 Project Overview

Diabetic Retinopathy is a progressive microvascular complication of diabetes and the leading cause of preventable blindness among working-age adults globally. Early detection through regular screening can prevent over 90% of severe vision loss.

**RetinaNova AI** addresses clinical screening bottlenecks by pairing automated computer vision preprocessing (Ben Graham's contrast enhancement) with a dual-branch neural network combining:
1. **Custom Deep Convolutional Neural Network (CNN):** Optimized for high-frequency micro-lesion textures (microaneurysms, hemorrhages, hard/soft exudates).
2. **Deep Transfer Learning Backbone (EfficientNetB0):** Captures high-level global semantic structure and retinal geometry.
3. **Multi-Modal Feature Fusion Layer:** Concatenates both latent representations into a fused 512-dimensional embedding for 5-stage Softmax classification.

---

## 🔬 5-Class International Severity Scale (ICDR)

| Class | Stage Name | Pathological Hallmarks | Urgency / Clinical Action |
|:---:|:---|:---|:---|
| **0** | **No DR (Normal)** | Normal vasculature, intact fovea, clear fundus | Routine annual dilated exam |
| **1** | **Mild NPDR** | Microaneurysms only (isolated focal capillary dilations) | Follow-up dilated exam in 6–9 months |
| **2** | **Moderate NPDR** | Microaneurysms, dot-blot hemorrhages, hard lipid exudates | Specialist evaluation in 3–6 months |
| **3** | **Severe NPDR** | Meets 4-2-1 rule: 4-quadrant hemorrhages, venous beading, IRMA | Urgent retina specialist in 2–4 weeks |
| **4** | **Proliferative DR (PDR)** | Neovascularization (NVD/NVE), vitreous hemorrhage | Emergency intervention within 24–48 hours |

---

## 🛠️ Technology Stack

- **Backend:** Python 3.14, Flask, Werkzeug (Secure Password Hashing & Sessions)
- **AI & Deep Learning:** TensorFlow / Keras 2.22, EfficientNetB0, Custom Deep CNN
- **Image Preprocessing:** OpenCV, NumPy (Ben Graham Gaussian Subtraction, Circular Masking)
- **Database:** SQLite (`database.db`)
- **PDF Report Generation:** ReportLab
- **Frontend & Visualizations:** HTML5, CSS3 (Medical Dashboard Theme), JavaScript, Chart.js, FontAwesome

---

## 📂 Project Directory Structure

```text
diabetic-retinopathy/
│
├── app.py                      # Flask backend application entry point
├── train.py                    # Complete model training & evaluation pipeline
├── generate_samples.py         # Generates 5 benchmark fundus sample images
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & project guide
├── database.db                 # SQLite database (auto-initialized)
│
├── models/
│   ├── hybrid_model.py         # Dual-branch Hybrid CNN + EfficientNet architecture
│   └── trained_model/
│       └── hybrid_dr_model.keras  # Serialized trained model weights
│
├── dataset/
│   └── samples/                # Sample retinal fundus benchmark images
│
├── uploads/
│   ├── enhanced/               # Preprocessed Ben Graham images
│   └── reports/                # Generated clinical PDF reports
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern medical-themed CSS
│   ├── js/
│   │   └── main.js             # Drag-and-drop upload & interactive handlers
│   └── images/
│       ├── samples/            # 5 stage benchmark images for quick demo
│       ├── training_curves.png # Accuracy & Loss training history graphs
│       └── confusion_matrix.png# Multi-class confusion matrix visualization
│
├── templates/
│   ├── base.html               # Master layout with navbar & medical disclaimer
│   ├── index.html              # Home page with epidemiology & overview
│   ├── about.html              # Scientific background & AI architecture
│   ├── login.html              # Clinician & user authentication
│   ├── register.html           # User registration
│   ├── dashboard.html          # Clinician scan history & distribution chart
│   ├── upload.html             # Drag-and-drop fundus image upload
│   ├── result.html             # Diagnostic prediction, Chart.js & findings
│   └── admin.html              # Administrative telemetry & user management
│
└── utils/
    ├── db.py                   # SQLite database manager & migration
    ├── preprocessing.py        # Ben Graham contrast enhancement & circular crop
    ├── prediction.py           # Multi-class inference & clinical guidance
    └── report_generator.py     # ReportLab clinical PDF generator
```

---

## 🚀 Quick Start Guide

### 1. Clone or Open the Repository
```bash
cd "d:\RetinaNova_AI project - 1"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Images & Train the Model (Optional / Pre-Built)
The project comes with a built-in trained hybrid model (`hybrid_dr_model.keras`) and generated sample fundus images. You can re-run the training pipeline at any time:
```bash
# Generate sample fundus benchmark images
python generate_samples.py

# Run the training pipeline
python train.py --epochs 3 --samples 10
```

### 4. Run the Web Application
```bash
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🔑 Default Demo Accounts for Evaluation (Email ID & Password)

You can sign in using **either** the Demo Email ID or Username:

| Account Role | Demo Email ID | Username | Password | Purpose & Capabilities |
|:---|:---|:---|:---|:---|
| **Clinician / Doctor** | `dr.smith@retinanova.med` | `doctor` | `doctor123` | Upload retinal fundus scans, view diagnostic evaluations, download clinical PDF reports |
| **System Administrator** | `admin@retinanova.med` | `admin` | `admin123` | Access admin portal, inspect system telemetry, view all registered users and global scans |
| **Research Fellow** | `clinician@retinanova.med` | `clinician` | `clinic123` | Perform batch research screenings, explore feature maps and model predictions |

> 💡 *Note: The login page also features **1-Click Auto-Fill** buttons for all demo accounts.* You can also register a brand new account at `/register`.

---

## 🩺 Medical Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This application is an educational and research prototype. It is not a certified medical diagnostic tool. Predictions must not replace professional examination by a qualified ophthalmologist. All clinical decisions must be confirmed by board-certified eye specialists.

