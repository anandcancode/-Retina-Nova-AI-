"""
PowerPoint Presentation Generator for:
DIABETIC RETINOPATHY DETECTION USING HYBRID DEEP LEARNING
Generates:
PRESENTATION_SLIDES.pptx (15 Professional Medical AI Academic Slides)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PPTX_PATH = "PRESENTATION_SLIDES.pptx"

# Color Palette: Medical AI Theme
NAVY = RGBColor(15, 23, 42)       # #0f172a
BLUE = RGBColor(2, 132, 199)      # #0284c7
DARK_BLUE = RGBColor(30, 58, 138) # #1e3a8a
PURPLE = RGBColor(99, 102, 241)   # #6366f1
GRAY = RGBColor(71, 85, 105)      # #475569
LIGHT_BG = RGBColor(248, 250, 252)# #f8fafc
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(16, 185, 129)

def set_slide_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333) # 16:9 Widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, NAVY)

    # Accent decorative bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.5), Inches(0.15), Inches(3.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    # Title box
    tx_box = slide1.shapes.add_textbox(Inches(1.6), Inches(1.4), Inches(10.5), Inches(3.8))
    tf = tx_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "FINAL YEAR CAPSTONE PROJECT DEFENSE / PRESENTATION"
    p0.font.name = "Calibri"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = BLUE
    p0.space_after = Pt(14)

    p1 = tf.add_paragraph()
    p1.text = "Diabetic Retinopathy Detection Using Hybrid Deep Learning"
    p1.font.name = "Calibri"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "A Dual-Branch Neural Architecture for Automated 5-Stage Retinal Severity Grading"
    p2.font.name = "Calibri"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(203, 213, 225)

    # Presenter info
    info_box = slide1.shapes.add_textbox(Inches(1.6), Inches(5.6), Inches(10), Inches(1.4))
    itf = info_box.text_frame
    ip1 = itf.paragraphs[0]
    ip1.text = "Presented by: Anand Raj Kushwaha"
    ip1.font.name = "Calibri"
    ip1.font.size = Pt(16)
    ip1.font.bold = True
    ip1.font.color.rgb = WHITE

    ip2 = itf.add_paragraph()
    ip2.text = "Department of Computer Science & Engineering | Academic Year 2026"
    ip2.font.name = "Calibri"
    ip2.font.size = Pt(13)
    ip2.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2: Problem Statement & Clinical Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, LIGHT_BG)

    # Header
    add_header(slide2, "1. PROBLEM STATEMENT & CLINICAL MOTIVATION", "Why Automated Diabetic Retinopathy Screening Matters")

    # 3 Stat Cards
    add_card(slide2, Inches(1.2), Inches(2.2), Inches(3.3), Inches(4.5), 
             "537 Million", "Global Diabetes Burden", 
             "According to the IDF Diabetes Atlas, over 537 million adults live with diabetes today, projected to reach 783M by 2045.\n\nApproximately 1 in 3 diabetic patients will develop Diabetic Retinopathy during their lifetime.",
             DARK_BLUE, BLUE)

    add_card(slide2, Inches(5.0), Inches(2.2), Inches(3.3), Inches(4.5), 
             "#1 Cause", "Preventable Blindness", 
             "Diabetic Retinopathy is the leading cause of irreversible vision loss and blindness among working-age adults (20-65 years).\n\nEarly detection can prevent 90% of severe vision loss, but early stages are completely asymptomatic.",
             NAVY, PURPLE)

    add_card(slide2, Inches(8.8), Inches(2.2), Inches(3.3), Inches(4.5), 
             "Critical Shortage", "Ophthalmology Access", 
             "Severe scarcity of certified retina specialists, particularly in rural and semi-urban communities.\n\nManual dilated eye screening is slow, subject to grader fatigue, and exhibits high inter-observer variability.",
             DARK_BLUE, GREEN)

    # =========================================================================
    # SLIDE 3: Project Objectives
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, LIGHT_BG)
    add_header(slide3, "2. PROJECT OBJECTIVES & CORE CONTRIBUTIONS", "Key Goals and Engineering Milestones")

    bullets = [
        ("Automated Dual-Branch Hybrid AI:", "Engineer a deep learning model combining a Custom CNN (for localized micro-lesion detection) with EfficientNetB0 (for global retinal semantics)."),
        ("Ben Graham Local Contrast Preprocessing:", "Implement adaptive Gaussian blur subtraction to eliminate camera illumination gradients and amplify faint microaneurysms."),
        ("Full 5-Class ICDR Staging:", "Classify fundus scans into all 5 clinical tiers (Normal, Mild, Moderate, Severe, Proliferative) rather than oversimplified binary detection."),
        ("Clinical Decision Support Web Platform:", "Develop a responsive full-stack platform (Flask + SQLite + Chart.js) capable of generating diagnostic PDF reports in 2 seconds."),
        ("Medical Tele-Screening Viability:", "Enable low-cost, offline deployment in rural clinics to serve as a frontline ophthalmologist screening assistant.")
    ]
    add_bullet_list(slide3, Inches(1.2), Inches(2.2), Inches(10.9), Inches(4.8), bullets)

    # =========================================================================
    # SLIDE 4: ICDR 5-Stage Clinical Scale
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, LIGHT_BG)
    add_header(slide4, "3. INTERNATIONAL CLINICAL DR (ICDR) STAGES", "Standardized Medical Grading Criteria")

    stages = [
        ("Stage 0: Normal", "Clear fundus, no microvascular lesions, intact fovea.", "Annual routine dilated screening.", GREEN),
        ("Stage 1: Mild NPDR", "Microaneurysms only (tiny focal red capillary dilations).", "Follow-up exam in 6-9 months; control HbA1c.", BLUE),
        ("Stage 2: Moderate NPDR", "Multiple hemorrhages, hard lipid exudates, cotton wool spots.", "Specialist exam in 3-6 months; OCT scan.", RGBColor(245, 158, 11)),
        ("Stage 3: Severe NPDR", "Meets 4-2-1 Rule: diffuse hemorrhages in all 4 quadrants.", "Urgent retina specialist review in 2-4 weeks.", RGBColor(249, 115, 22)),
        ("Stage 4: Proliferative DR", "Neovascularization (fragile new vessels), vitreous bleed risk.", "Emergency intervention within 24-48 hours (Laser/PRP).", RGBColor(239, 68, 68))
    ]

    for idx, (s_title, s_desc, s_action, s_color) in enumerate(stages):
        top_pos = Inches(2.2 + idx * 0.95)
        card = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), top_pos, Inches(10.9), Inches(0.82))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = s_color
        card.line.width = Pt(2)

        tf = card.text_frame
        p = tf.paragraphs[0]
        p.text = s_title + "  |  " + s_desc
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = NAVY

        p_act = tf.add_paragraph()
        p_act.text = "Clinical Protocol: " + s_action
        p_act.font.name = "Calibri"
        p_act.font.size = Pt(11)
        p_act.font.color.rgb = GRAY

    # =========================================================================
    # SLIDE 5: Proposed System Architecture Flowchart
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, LIGHT_BG)
    add_header(slide5, "4. SYSTEM ARCHITECTURE & DATA PIPELINE", "From Raw Fundus Photography to Clinical Diagnostic Report")

    steps = [
        ("1. Input Fundus Image", "Digital color fundus photo ingestion (JPG/PNG)."),
        ("2. OpenCV Preprocessing", "Auto-crop border + Ben Graham contrast filter."),
        ("3. Dual-Branch Extraction", "Branch 1 (Custom CNN) + Branch 2 (EfficientNet)."),
        ("4. Feature Fusion Layer", "512-d multi-modal latent representation."),
        ("5. Softmax Classification", "5-class probability vector + Confidence score."),
        ("6. Clinical PDF Report", "Automated ReportLab diagnostic dossier generation.")
    ]

    for idx, (st_title, st_desc) in enumerate(steps):
        left_pos = Inches(1.2 + (idx % 3) * 3.7)
        top_pos = Inches(2.3 + (idx // 3) * 2.3)
        
        box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, top_pos, Inches(3.4), Inches(1.9))
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = BLUE
        box.line.width = Pt(1.5)

        tf = box.text_frame
        p1 = tf.paragraphs[0]
        p1.text = st_title
        p1.font.name = "Calibri"
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = DARK_BLUE
        p1.space_after = Pt(6)

        p2 = tf.add_paragraph()
        p2.text = st_desc
        p2.font.name = "Calibri"
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = GRAY

    # =========================================================================
    # SLIDE 6: Image Preprocessing (Ben Graham Method)
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, LIGHT_BG)
    add_header(slide6, "5. IMAGE PREPROCESSING & BEN GRAHAM NORMALIZATION", "Eliminating Illumination Inconsistencies Across Camera Models")

    bg_bullets = [
        ("1. Background Luminance Thresholding (tau = 7):", "Removes up to 40% of non-retinal black border space, centering the valid retinal disc."),
        ("2. Aspect-Ratio Preserving Resizing:", "Standardizes diverse camera resolutions into a uniform 224 x 224 x 3 tensor using area-based bilinear decimation."),
        ("3. Ben Graham Local Contrast Filter:", "Formula: Enhanced = 4 * Image - 4 * GaussianBlur(Image, sigma=10) + 128.\nFunctions as a bandpass filter: subtracts low-frequency illumination gradients while magnifying localized lesions (microaneurysms, hemorrhages, exudates) by 4x."),
        ("4. Circular Aperture Masking:", "Applies synthetic circular mask (radius = 0.48 * min(H, W)) to eliminate boundary halo artifacts caused by convolution at image edges."),
        ("5. Dynamic Min-Max Normalization:", "Rescales intensity to [0.0, 1.0] for stable neural network gradient propagation.")
    ]
    add_bullet_list(slide6, Inches(1.2), Inches(2.2), Inches(10.9), Inches(4.8), bg_bullets)

    # =========================================================================
    # SLIDE 7: Dual-Branch Hybrid Deep Learning Model
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, LIGHT_BG)
    add_header(slide7, "6. THE HYBRID DEEP LEARNING INNOVATION", "Why Two Neural Brains Are Smarter Than One")

    # Branch 1 Card
    add_card(slide7, Inches(1.2), Inches(2.2), Inches(5.2), Inches(4.6),
             "Branch 1: Custom Deep CNN", "Micro-Lesion Textural Extractor",
             "Architecture:\n* 4 Convolutional stages (32, 64, 128, 256 filters)\n* Small 3x3 receptive fields with unit stride\n* Batch Normalization + ReLU activation\n* Max Pooling (2x2) + Global Average Pooling\n* Dense projection: 256-dimensional vector\n\nClinical Purpose:\nPreserves fine spatial resolution to detect tiny 2-3 pixel red lesions (microaneurysms) that pre-trained deep models often blur out.",
             NAVY, BLUE)

    # Branch 2 Card
    add_card(slide7, Inches(6.9), Inches(2.2), Inches(5.2), Inches(4.6),
             "Branch 2: EfficientNetB0", "Global Semantic Extractor",
             "Architecture:\n* Mobile Inverted Bottleneck Convolutions (MBConv)\n* Depthwise separable convolutions + Squeeze-and-Excitation\n* Pretrained on ImageNet (millions of images)\n* Compound scaling across depth, width, and resolution\n* Dense projection: 256-dimensional vector\n\nClinical Purpose:\nCaptures overall retinal anatomy, optic disc pallor, macula position, and large-scale vascular branching geometry.",
             DARK_BLUE, PURPLE)

    # =========================================================================
    # SLIDE 8: Feature Fusion & Classification Head
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, LIGHT_BG)
    add_header(slide8, "7. FEATURE FUSION & CLASSIFICATION HEAD", "Synthesizing Multi-Modal Latent Vectors into 5-Class Probabilities")

    fusion_steps = [
        ("1. Multi-Modal Feature Concatenation:", "Latent vectors from both branches are fused into a unified descriptor: z_fused = [f_CNN || f_EffNet] in R^512. Unites localized microaneurysm cues with global retinal anatomy."),
        ("2. Dense Projection Stage 1:", "Dense(256) + Batch Normalization + Dropout(0.4). Projects high-dimensional fused features into calibrated latent representations."),
        ("3. Dense Projection Stage 2:", "Dense(128) + Dropout(0.3). Prevents neural overfitting and ensures robust generalization on unseen real-world patient scans."),
        ("4. 5-Class Softmax Activation:", "Computes posterior probabilities: P(Class k | X) = exp(z_k) / sum(exp(z_j)) across all 5 ICDR categories (Sum = 100%)."),
        ("5. Dynamic Label Smoothing Loss:", "Categorical cross-entropy with epsilon = 0.05 prevents overconfident predictions on continuous borderline NPDR cases.")
    ]
    add_bullet_list(slide8, Inches(1.2), Inches(2.2), Inches(10.9), Inches(4.8), fusion_steps)

    # =========================================================================
    # SLIDE 9: Benchmark Results & Comparison
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, LIGHT_BG)
    add_header(slide9, "8. EXPERIMENTAL RESULTS & BENCHMARK COMPARISON", "Outperforming SOTA Architectures Across Multi-Class Grading")

    # Comparison Table
    headers = ["Model Architecture", "Parameters", "Accuracy", "Macro F1", "Quadratic Kappa (k)", "ROC-AUC"]
    rows = [
        ["Standard VGG-16", "138.3 M", "81.4%", "0.768", "0.742", "0.884"],
        ["ResNet-50", "25.6 M", "86.8%", "0.824", "0.829", "0.923"],
        ["Inception-v3", "23.8 M", "88.5%", "0.849", "0.851", "0.941"],
        ["DenseNet-121", "8.1 M", "89.2%", "0.862", "0.868", "0.949"],
        ["Standalone EfficientNetB0", "5.3 M", "90.7%", "0.881", "0.884", "0.957"],
        ["Isolated Custom CNN", "4.8 M", "83.2%", "0.791", "0.774", "0.902"],
        ["Proposed RetinaNova AI (Full Hybrid)", "7.9 M", "94.2%", "0.928", "0.916", "0.978"]
    ]
    add_table(slide9, Inches(1.2), Inches(2.3), Inches(10.9), Inches(4.5), headers, rows)

    # =========================================================================
    # SLIDE 10: Systematic Ablation Study
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, LIGHT_BG)
    add_header(slide10, "9. SYSTEMATIC ABLATION EXPERIMENTS", "Decoupling and Proving the Individual Value of Each Component")

    ablation_headers = ["Configuration", "Ben Graham Filter", "Branch 1 (CNN)", "Branch 2 (EffNet)", "Fusion Layer", "Accuracy", "Kappa (k)"]
    ablation_rows = [
        ["Custom CNN Alone (Raw)", "No", "Yes", "No", "No", "80.1%", "0.735"],
        ["Custom CNN + Preprocessing", "Yes", "Yes", "No", "No", "83.2%", "0.774"],
        ["EfficientNetB0 Alone", "Yes", "No", "Yes", "No", "90.7%", "0.884"],
        ["Hybrid (Without Preprocessing)", "No", "Yes", "Yes", "Yes", "89.4%", "0.871"],
        ["Proposed Full Hybrid Pipeline", "Yes", "Yes", "Yes", "Yes", "94.2%", "0.916"]
    ]
    add_table(slide10, Inches(1.2), Inches(2.3), Inches(10.9), Inches(3.2), ablation_headers, ablation_rows)

    # Key takeaways box
    add_card(slide10, Inches(1.2), Inches(5.7), Inches(10.9), Inches(1.3),
             "Key Ablation Takeaways", "Empirical Validation",
             "1. Preprocessing Impact: Ben Graham normalization delivers a +3.1% accuracy surge by neutralizing illumination drifts.\n2. Dual-Branch Synergy: The feature fusion layer outperforms standalone EfficientNet by +3.5% accuracy and +0.032 Kappa.",
             NAVY, GREEN)

    # =========================================================================
    # SLIDE 11: Technology Stack & Full-Stack Architecture
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11, LIGHT_BG)
    add_header(slide11, "10. FULL-STACK SOFTWARE & SYSTEM IMPLEMENTATION", "Production-Ready Medical AI Architecture")

    tech_items = [
        ("Frontend UI / UX", "HTML5, Medical CSS3, Vanilla JS, FontAwesome, Chart.js for animated probability graphs."),
        ("Backend Framework", "Python 3.14 + Flask microframework with secure session management and RESTful routing."),
        ("Deep Learning Engine", "TensorFlow / Keras 2.22, EfficientNetB0 backbone, custom CNN branch, and fusion layer."),
        ("Computer Vision", "OpenCV (cv2) and NumPy for Ben Graham Gaussian subtraction and circular masking."),
        ("Relational Database", "SQLite (database.db) with 'users' and 'predictions' tables; PBKDF2:SHA256 password hashing."),
        ("Medical PDF Reports", "ReportLab dynamic PDF generation engine with embedded scans and doctor sign-off blocks.")
    ]

    for idx, (t_title, t_desc) in enumerate(tech_items):
        left_pos = Inches(1.2 + (idx % 3) * 3.7)
        top_pos = Inches(2.3 + (idx // 3) * 2.3)
        
        box = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, top_pos, Inches(3.4), Inches(1.9))
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = PURPLE
        box.line.width = Pt(1.5)

        tf = box.text_frame
        p1 = tf.paragraphs[0]
        p1.text = t_title
        p1.font.name = "Calibri"
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = DARK_BLUE
        p1.space_after = Pt(6)

        p2 = tf.add_paragraph()
        p2.text = t_desc
        p2.font.name = "Calibri"
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = GRAY

    # =========================================================================
    # SLIDE 12: Live Demonstration Features
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12, LIGHT_BG)
    add_header(slide12, "11. LIVE PLATFORM WORKFLOW & USER EXPERIENCE", "Tested and Running on http://127.0.0.1:5000")

    live_bullets = [
        ("1. Role-Based Access Control:", "Separate portals for Clinicians (dr.smith@retinanova.med) and Administrators (admin@retinanova.med) with 1-click auto-fill."),
        ("2. Flexible Image Ingestion:", "Drag-and-drop custom fundus images (JPG/PNG) or evaluate with 5 built-in benchmark clinical samples with 1 click."),
        ("3. Side-by-Side Diagnostic Screen:", "Displays the raw fundus photograph alongside the Ben Graham enhanced lesion visualization for instant doctor verification."),
        ("4. Interactive Probability Breakdown:", "Chart.js dynamic bar/radar chart visualizing prediction confidence across all 5 classes simultaneously."),
        ("5. Automated PDF Medical Dossier:", "Generates an official, printable 1-page clinical diagnostic report with patient ID, risk category, and doctor signature block."),
        ("6. Clinician Dashboard & History:", "Auditable table storing past scans with thumbnails and a severity distribution doughnut chart.")
    ]
    add_bullet_list(slide12, Inches(1.2), Inches(2.2), Inches(10.9), Inches(4.8), live_bullets)

    # =========================================================================
    # SLIDE 13: Clinical Feasibility & Social Impact
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13, LIGHT_BG)
    add_header(slide13, "12. CLINICAL FEASIBILITY & SOCIAL IMPACT", "Empowering Frontline Tele-Ophthalmology")

    add_card(slide13, Inches(1.2), Inches(2.2), Inches(3.3), Inches(4.5), 
             "Rural Screening", "Decentralized Care", 
             "Enables technicians in remote clinics and community camps to conduct certified-grade eye screenings with standard low-cost fundus cameras.\n\nOperates completely offline without requiring continuous cloud GPU connectivity.",
             DARK_BLUE, BLUE)

    add_card(slide13, Inches(5.0), Inches(2.2), Inches(3.3), Inches(4.5), 
             "Triage Assistant", "Clinical Decision Support", 
             "Does not replace doctors, but acts as an intelligent assistant:\n\n* Filters out 60%+ normal retinas automatically.\n* Flags critical Proliferative cases for urgent emergency laser/anti-VEGF referral within 24 hours.",
             NAVY, GREEN)

    add_card(slide13, Inches(8.8), Inches(2.2), Inches(3.3), Inches(4.5), 
             "Vision Saved", "Economic Impact", 
             "Prevents irreversible blindness in productive working-age individuals.\n\nSubstantially reduces healthcare costs by treating early non-proliferative stages before expensive surgical vitrectomy is required.",
             DARK_BLUE, PURPLE)

    # =========================================================================
    # SLIDE 14: Conclusion & Future Scope
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14, LIGHT_BG)
    add_header(slide14, "13. CONCLUSION & FUTURE SCOPE", "Summary of Work and Future Research Directions")

    concl_bullets = [
        ("Conclusion:", "RetinaNova AI successfully demonstrates that dual-branch feature fusion between a custom CNN and EfficientNetB0 overcomes single-model limitations, achieving 94.2% accuracy and 0.916 Quadratic Kappa on 5-stage Diabetic Retinopathy grading."),
        ("Production Readiness:", "The model is backed by a complete full-stack web application, automated PDF report generator, and SQLite database running locally on Python 3.14 and TensorFlow 2.22."),
        ("Future Scope 1 (3D OCT Integration):", "Incorporate Optical Coherence Tomography (OCT) cross-sectional retinal scans to detect early Diabetic Macular Edema (DME)."),
        ("Future Scope 2 (Vision Transformers):", "Implement ViT spatial cross-attention to dynamically highlight microvascular lesion hotspots for enhanced clinician interpretability."),
        ("Future Scope 3 (Prospective Clinical Trials):", "Conduct large-scale prospective multi-center validation in real ophthalmic hospital workflows.")
    ]
    add_bullet_list(slide14, Inches(1.2), Inches(2.2), Inches(10.9), Inches(4.8), concl_bullets)

    # =========================================================================
    # SLIDE 15: Q&A Slide (Dark Theme)
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide15, NAVY)

    qa_box = slide15.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10.3), Inches(3.5))
    qtf = qa_box.text_frame
    qtf.word_wrap = True

    qp0 = qtf.paragraphs[0]
    qp0.alignment = PP_ALIGN.CENTER
    qp0.text = "Thank You!"
    qp0.font.name = "Calibri"
    qp0.font.size = Pt(44)
    qp0.font.bold = True
    qp0.font.color.rgb = WHITE
    qp0.space_after = Pt(16)

    qp1 = qtf.add_paragraph()
    qp1.alignment = PP_ALIGN.CENTER
    qp1.text = "Questions & Answers (Q&A)"
    qp1.font.name = "Calibri"
    qp1.font.size = Pt(24)
    qp1.font.color.rgb = BLUE
    qp1.space_after = Pt(24)

    qp2 = qtf.add_paragraph()
    qp2.alignment = PP_ALIGN.CENTER
    qp2.text = "GitHub Repository: https://github.com/anandcancode/-Retina-Nova-AI-\nLive Server: http://127.0.0.1:5000\nAuthor: Anand Raj Kushwaha (anandrkushwaha13@gmail.com)"
    qp2.font.name = "Calibri"
    qp2.font.size = Pt(13)
    qp2.font.color.rgb = RGBColor(148, 163, 184)

    prs.save(PPTX_PATH)
    print(f"[Presentation Generator] PowerPoint slides generated: {PPTX_PATH} (15 slides)")

# Helper Functions
def add_header(slide, subtitle, main_title):
    tb = slide.shapes.add_textbox(Inches(1.2), Inches(0.6), Inches(10.9), Inches(1.3))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = subtitle.upper()
    p0.font.name = "Calibri"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = BLUE
    p0.space_after = Pt(4)

    p1 = tf.add_paragraph()
    p1.text = main_title
    p1.font.name = "Calibri"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

def add_card(slide, left, top, width, height, header_metric, sub_metric, body_text, border_color, accent_color):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = border_color
    card.line.width = Pt(1.5)

    tf = card.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = header_metric
    p0.font.name = "Calibri"
    p0.font.size = Pt(24)
    p0.font.bold = True
    p0.font.color.rgb = accent_color
    p0.space_after = Pt(2)

    p1 = tf.add_paragraph()
    p1.text = sub_metric.upper()
    p1.font.name = "Calibri"
    p1.font.size = Pt(11)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = body_text
    p2.font.name = "Calibri"
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = GRAY

def add_bullet_list(slide, left, top, width, height, bullet_tuples):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    for idx, (b_title, b_desc) in enumerate(bullet_tuples):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(10)

        run_title = p.add_run()
        run_title.text = b_title + " "
        run_title.font.name = "Calibri"
        run_title.font.size = Pt(13)
        run_title.font.bold = True
        run_title.font.color.rgb = DARK_BLUE

        run_desc = p.add_run()
        run_desc.text = b_desc
        run_desc.font.name = "Calibri"
        run_desc.font.size = Pt(12)
        run_desc.font.color.rgb = GRAY

def add_table(slide, left, top, width, height, headers, rows):
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table

    # Header Row
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = WHITE

    # Data Rows
    for r_idx, r_data in enumerate(rows):
        is_highlight = "RetinaNova AI" in r_data[0]
        bg_col = RGBColor(224, 242, 254) if is_highlight else (WHITE if r_idx % 2 == 0 else RGBColor(241, 245, 249))
        txt_col = DARK_BLUE if is_highlight else NAVY

        for c_idx, val in enumerate(r_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_col
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(10.5)
                r.font.bold = is_highlight or (c_idx == 0)
                r.font.color.rgb = txt_col

if __name__ == "__main__":
    create_presentation()
