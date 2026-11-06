"""
Diabetic Retinopathy Detection - Database Management Module
Handles SQLite schema creation, migrations, user management, and prediction records.
"""

import os
import sqlite3
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database.db"))


def get_db():
    """
    Returns a connection to the SQLite database with row factory enabled.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the SQLite database tables and seeds a default admin account.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Predictions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_image_path TEXT NOT NULL,
            enhanced_image_path TEXT NOT NULL,
            predicted_class INTEGER NOT NULL,
            predicted_stage TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            urgency TEXT,
            description TEXT,
            report_pdf_path TEXT,
            is_simulated INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)

    conn.commit()

    # Seed default admin user if not exists
    cursor.execute("SELECT id FROM users WHERE username = ?", ('admin',))
    admin_row = cursor.fetchone()
    if not admin_row:
        admin_pass = generate_password_hash("admin123")
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, ('admin', 'admin@retinanova.med', admin_pass, 'Dr. Retina Administrator', 'admin'))
        conn.commit()
        print("[Database] Seeded default administrator account: admin / admin123")

    # Seed demo user if not exists
    cursor.execute("SELECT id FROM users WHERE username = ?", ('doctor',))
    doctor_row = cursor.fetchone()
    if not doctor_row:
        doc_pass = generate_password_hash("doctor123")
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, ('doctor', 'dr.smith@retinanova.med', doc_pass, 'Dr. Sarah Smith (Ophthalmologist)', 'user'))
        conn.commit()
        print("[Database] Seeded default clinician account: doctor / doctor123")

    conn.close()


def create_user(username, email, password, full_name=None, role='user'):
    """
    Registers a new user with hashed password.
    Returns (user_id, error_message).
    """
    conn = get_db()
    cursor = conn.cursor()
    pwd_hash = generate_password_hash(password)

    try:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, (username.strip(), email.strip().lower(), pwd_hash, full_name, role))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError as err:
        conn.close()
        if 'users.username' in str(err):
            return None, "Username already exists. Please choose a different username."
        elif 'users.email' in str(err):
            return None, "Email address is already registered."
        return None, "A user with that credential already exists."
    except Exception as e:
        conn.close()
        return None, str(e)


def authenticate_user(username_or_email, password):
    """
    Authenticates a user via username/email and password.
    Returns user dict or None.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM users
        WHERE username = ? OR email = ?
    """, (username_or_email.strip(), username_or_email.strip().lower()))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None


def get_user_by_id(user_id):
    """
    Retrieves user by primary ID.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def insert_prediction(
    user_id,
    filename,
    original_image_path,
    enhanced_image_path,
    predicted_class,
    predicted_stage,
    stage_name,
    confidence,
    probabilities,
    risk_level,
    urgency,
    description,
    report_pdf_path=None,
    is_simulated=0
):
    """
    Inserts a newly computed prediction record.
    Returns new prediction_id.
    """
    conn = get_db()
    cursor = conn.cursor()

    if isinstance(probabilities, (list, tuple)):
        prob_str = json.dumps(probabilities)
    else:
        prob_str = str(probabilities)

    cursor.execute("""
        INSERT INTO predictions (
            user_id, filename, original_image_path, enhanced_image_path,
            predicted_class, predicted_stage, stage_name, confidence,
            probabilities, risk_level, urgency, description,
            report_pdf_path, is_simulated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, filename, original_image_path, enhanced_image_path,
        predicted_class, predicted_stage, stage_name, confidence,
        prob_str, risk_level, urgency, description,
        report_pdf_path, 1 if is_simulated else 0
    ))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id


def update_prediction_report(prediction_id, report_pdf_path):
    """
    Updates the prediction record with its generated PDF report path.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE predictions
        SET report_pdf_path = ?
        WHERE id = ?
    """, (report_pdf_path, prediction_id))
    conn.commit()
    conn.close()


def get_user_predictions(user_id, limit=None):
    """
    Returns prediction history for a specific user, newest first.
    """
    conn = get_db()
    cursor = conn.cursor()
    query = """
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d['probabilities'] = json.loads(d['probabilities'])
        except Exception:
            pass
        result.append(d)
    return result


def get_prediction_by_id(prediction_id):
    """
    Returns a single prediction record by ID.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        d = dict(row)
        try:
            d['probabilities'] = json.loads(d['probabilities'])
        except Exception:
            pass
        return d
    return None


def get_all_predictions(limit=None):
    """
    Admin query: returns all predictions joined with user information.
    """
    conn = get_db()
    cursor = conn.cursor()
    query = """
        SELECT p.*, u.username, u.email, u.full_name
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d['probabilities'] = json.loads(d['probabilities'])
        except Exception:
            pass
        result.append(d)
    return result


def get_all_users():
    """
    Admin query: returns list of all users and their scan counts.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.username, u.email, u.full_name, u.role, u.created_at,
               COUNT(p.id) AS scan_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_system_stats():
    """
    Computes global statistics for dashboards (user count, scans, stage distributions).
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_scans = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0]
    avg_confidence = round(float(avg_conf), 2) if avg_conf else 0.0

    # Stage distribution counts
    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count
        FROM predictions
        GROUP BY predicted_class
    """)
    stage_rows = cursor.fetchall()
    stage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for row in stage_rows:
        stage_counts[row['predicted_class']] = row['count']

    conn.close()

    return {
        "total_users": total_users,
        "total_scans": total_scans,
        "avg_confidence": avg_confidence,
        "stage_distribution": stage_counts
    }

