"""
Diabetic Retinopathy Detection Using Hybrid Deep Learning
Flask Application Entry Point
"""

import os
import json
import uuid
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, jsonify, send_file, abort
)
from werkzeug.utils import secure_filename

from utils.db import (
    init_db, create_user, authenticate_user, get_user_by_id,
    insert_prediction, update_prediction_report, get_user_predictions,
    get_prediction_by_id, get_all_predictions, get_all_users, get_system_stats
)
from utils.preprocessing import preprocess_fundus_image, validate_image_file
from utils.prediction import run_prediction, DR_CLASSES_INFO
from utils.report_generator import generate_pdf_report

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "retinanova-medical-ai-deep-learning-secret-2026")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "uploads", "reports")
ENHANCED_FOLDER = os.path.join(BASE_DIR, "uploads", "enhanced")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Initialize database on module load
init_db()


# -------------------------------------------------------------
# Context Processors & Helpers
# -------------------------------------------------------------
@app.context_processor
def inject_user():
    """Injects current logged-in user into all templates."""
    user = None
    if "user_id" in session:
        user = get_user_by_id(session["user_id"])
    return {
        "current_user": user,
        "now_year": datetime.now().year,
        "disclaimer_text": (
            "This application is an educational and research prototype. "
            "It is not a certified medical diagnostic tool. Predictions must "
            "not replace professional examination by a qualified ophthalmologist."
        )
    }


def login_required(f):
    """Decorator to protect routes requiring authentication."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to access this feature.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to protect admin routes."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Administrator login required.", "warning")
            return redirect(url_for("login", next=request.url))
        user = get_user_by_id(session["user_id"])
        if not user or user.get("role") != "admin":
            flash("Access denied: Administrative privileges required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated_function


# -------------------------------------------------------------
# Public & Informational Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    """Home Page with project intro, AI architecture, and awareness."""
    stats = get_system_stats()
    return render_template("index.html", stats=stats)


@app.route("/about")
def about():
    """About Page detailing Diabetic Retinopathy, stages, and Hybrid AI."""
    return render_template("about.html")


# -------------------------------------------------------------
# Authentication Routes
# -------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration with password hashing."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        full_name = request.form.get("full_name", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All required fields must be filled out.", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match. Please verify and try again.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("register.html")

        user_id, err = create_user(username, email, password, full_name=full_name)
        if err:
            flash(err, "danger")
            return render_template("register.html")

        # Auto-login after registration
        session["user_id"] = user_id
        session["username"] = username
        flash(f"Welcome to RetinaNova AI, {full_name or username}! Your account is ready.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User and Clinician login."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    next_page = request.args.get("next")

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        if not identifier or not password:
            flash("Please enter both your username/email and password.", "danger")
            return render_template("login.html")

        user = authenticate_user(identifier, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash(f"Signed in successfully as {user.get('full_name') or user['username']}.", "success")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Please check your username/password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Sign out user and clear session."""
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("index"))


# -------------------------------------------------------------
# User Dashboard & History
# -------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    """User Dashboard showing scan statistics, recent history, and stage charts."""
    user_id = session["user_id"]
    user = get_user_by_id(user_id)
    predictions = get_user_predictions(user_id, limit=20)
    total_scans = len(predictions)

    # Compute stage distribution for user
    stage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for p in predictions:
        cls = p.get("predicted_class", 0)
        stage_counts[cls] = stage_counts.get(cls, 0) + 1

    return render_template(
        "dashboard.html",
        user=user,
        predictions=predictions,
        total_scans=total_scans,
        stage_counts=stage_counts
    )


# -------------------------------------------------------------
# Retinal Upload & AI Detection
# -------------------------------------------------------------
@app.route("/upload")
@login_required
def upload():
    """Image upload interface for retinal fundus analysis."""
    # List available sample fundus images
    samples_dir = os.path.join(BASE_DIR, "static", "images", "samples")
    sample_files = []
    if os.path.exists(samples_dir):
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return render_template("upload.html", sample_files=sample_files)


@app.route("/sample-scan/<filename>")
@login_required
def sample_scan(filename):
    """
    Shortcut to test detection with one of the preloaded clinical sample images.
    Copies sample into uploads directory and performs inference.
    """
    safe_name = secure_filename(filename)
    sample_src = os.path.join(BASE_DIR, "static", "images", "samples", safe_name)
    if not os.path.exists(sample_src):
        flash("Sample image not found.", "danger")
        return redirect(url_for("upload"))

    import shutil
    unique_prefix = f"sample_{uuid.uuid4().hex[:8]}"
    dest_filename = f"{unique_prefix}_{safe_name}"
    dest_path = os.path.join(UPLOAD_FOLDER, dest_filename)
    shutil.copyfile(sample_src, dest_path)

    return _process_image_and_redirect(dest_path, dest_filename)


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    """
    Processes uploaded retinal fundus image:
    1. Validates file format and size
    2. Runs Ben Graham contrast enhancement & circular crop
    3. Runs Hybrid Deep Learning inference
    4. Generates PDF clinical report
    5. Stores scan record in SQLite database
    """
    if "fundus_image" not in request.files:
        flash("No image file provided in upload request.", "danger")
        return redirect(url_for("upload"))

    file = request.files["fundus_image"]
    if file.filename == "":
        flash("No file was selected for upload.", "warning")
        return redirect(url_for("upload"))

    # Validate image file
    is_valid, msg = validate_image_file(file.filename)
    if not is_valid:
        flash(msg, "danger")
        return redirect(url_for("upload"))

    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_id = uuid.uuid4().hex[:10]
    safe_name = f"fundus_{unique_id}{file_ext}"
    saved_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(saved_path)

    # Process and redirect
    return _process_image_and_redirect(saved_path, safe_name)


def _process_image_and_redirect(saved_path, filename):
    """Internal helper to preprocess, run inference, generate PDF, and redirect."""
    user_id = session.get("user_id")
    user = get_user_by_id(user_id) if user_id else {"username": "Guest Clinician"}

    # Step 1: Preprocess image & create enhanced version
    unique_base = os.path.splitext(filename)[0]
    enhanced_filename = f"enhanced_{unique_base}.png"
    enhanced_path = os.path.join(ENHANCED_FOLDER, enhanced_filename)

    try:
        input_tensor, enhanced_bgr, quality_info = preprocess_fundus_image(
            saved_path,
            save_enhanced_path=enhanced_path
        )
    except Exception as e:
        flash(f"Image preprocessing failed: {e}", "danger")
        return redirect(url_for("upload"))

    # Step 2: Run Hybrid Deep Learning Prediction
    prediction_result = run_prediction(input_tensor, quality_info)

    # Relative paths for web serving
    rel_original = f"uploads/{filename}"
    rel_enhanced = f"uploads/enhanced/{enhanced_filename}"

    # Step 3: Insert initial record in database
    pred_id = insert_prediction(
        user_id=user_id,
        filename=filename,
        original_image_path=rel_original,
        enhanced_image_path=rel_enhanced,
        predicted_class=prediction_result["predicted_class"],
        predicted_stage=prediction_result["predicted_stage"],
        stage_name=prediction_result["stage_name"],
        confidence=prediction_result["confidence"],
        probabilities=prediction_result["probabilities"],
        risk_level=prediction_result["risk_level"],
        urgency=prediction_result["urgency"],
        description=prediction_result["description"],
        is_simulated=1 if prediction_result["is_simulated"] else 0
    )

    # Step 4: Generate downloadable PDF report
    report_filename = f"DR_Report_{pred_id}_{unique_base}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    try:
        generate_pdf_report(
            output_pdf_path=report_path,
            scan_id=pred_id,
            user_name=user.get("full_name") or user.get("username", "Clinician"),
            original_img_path=saved_path,
            enhanced_img_path=enhanced_path,
            prediction_data=prediction_result
        )
        rel_report = f"uploads/reports/{report_filename}"
        update_prediction_report(pred_id, rel_report)
    except Exception as err:
        print(f"[Report] Notice: PDF generation warning ({err})")

    flash("Retinal image analyzed successfully by Hybrid Deep Learning engine!", "success")
    return redirect(url_for("result", prediction_id=pred_id))


# -------------------------------------------------------------
# Result & PDF Report Routes
# -------------------------------------------------------------
@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    """
    Displays full diagnostic results:
    - Retinal images (original + enhanced)
    - Predicted DR stage badge & risk category
    - Confidence score
    - Interactive Chart.js probability breakdown
    - Pathological indicators & clinical findings
    - Actionable recommendations & PDF report download link
    """
    pred = get_prediction_by_id(prediction_id)
    if not pred:
        flash("Scan result not found.", "warning")
        return redirect(url_for("dashboard"))

    # Permission check: users only view own predictions unless admin
    current_uid = session.get("user_id")
    current_role = session.get("role")
    if pred["user_id"] != current_uid and current_role != "admin":
        flash("You are not authorized to view this diagnostic record.", "danger")
        return redirect(url_for("dashboard"))

    class_info = DR_CLASSES_INFO.get(pred["predicted_class"], DR_CLASSES_INFO[0])

    return render_template(
        "result.html",
        prediction=pred,
        class_info=class_info
    )


@app.route("/report/<int:prediction_id>/download")
@login_required
def download_report(prediction_id):
    """Serves the generated PDF diagnostic report for download."""
    pred = get_prediction_by_id(prediction_id)
    if not pred or not pred.get("report_pdf_path"):
        flash("PDF Report is not available for this scan.", "warning")
        return redirect(url_for("result", prediction_id=prediction_id))

    current_uid = session.get("user_id")
    current_role = session.get("role")
    if pred["user_id"] != current_uid and current_role != "admin":
        flash("You are not authorized to download this report.", "danger")
        return redirect(url_for("dashboard"))

    full_pdf_path = os.path.join(BASE_DIR, pred["report_pdf_path"])
    if not os.path.exists(full_pdf_path):
        flash("Report file not found on server.", "danger")
        return redirect(url_for("result", prediction_id=prediction_id))

    return send_file(
        full_pdf_path,
        as_attachment=True,
        download_name=os.path.basename(full_pdf_path),
        mimetype="application/pdf"
    )


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    """Static file server for uploaded retinal images and reports."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_file(file_path)


@app.route("/download/paper")
def download_paper():
    """Download the academic research paper (.docx or .md)."""
    format_type = request.args.get("format", "docx")
    if format_type == "md":
        paper_file = os.path.join(BASE_DIR, "RESEARCH_PAPER.md")
        mimetype = "text/markdown"
        filename = "RetinaNova_AI_Research_Paper.md"
    else:
        paper_file = os.path.join(BASE_DIR, "RESEARCH_PAPER.docx")
        mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        filename = "RetinaNova_AI_Research_Paper.docx"

    if not os.path.exists(paper_file):
        abort(404)
    return send_file(paper_file, as_attachment=True, download_name=filename, mimetype=mimetype)


# -------------------------------------------------------------
# Admin Dashboard
# -------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin():
    """
    Admin Dashboard:
    - Registered users table with scan counts
    - All predictions system-wide
    - Global system statistics & stage distributions
    """
    stats = get_system_stats()
    all_users = get_all_users()
    all_preds = get_all_predictions(limit=50)

    return render_template(
        "admin.html",
        stats=stats,
        users=all_users,
        predictions=all_preds
    )


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.route("/api/stats")
def api_stats():
    """Returns JSON system statistics for dynamic dashboard widgets."""
    stats = get_system_stats()
    return jsonify(stats)


if __name__ == "__main__":
    print("=" * 60)
    print("Starting RetinaNova AI - Diabetic Retinopathy Diagnostic Server")
    print("Access locally at: http://127.0.0.1:5000")
    print("Default Admin Login: admin / admin123")
    print("Default Clinician Login: doctor / doctor123")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)

