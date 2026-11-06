"""
Automated End-to-End Test Suite for RetinaNova AI
Tests:
1. App initialization & routes
2. User registration and authentication
3. Image preprocessing pipeline
4. Hybrid AI model inference
5. PDF clinical report generation
6. User dashboard & admin telemetry
"""

import os
import unittest
import json
from app import app
from utils.db import init_db, get_db, authenticate_user
from utils.preprocessing import preprocess_fundus_image, validate_image_file
from utils.prediction import run_prediction
from utils.report_generator import generate_pdf_report

class RetinaNovaTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        init_db()

    def test_01_public_routes(self):
        """Test home and about pages return HTTP 200."""
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"RetinaNova", res_home.data)

        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)
        self.assertIn(b"About Diabetic Retinopathy", res_about.data)

    def test_02_authentication_and_registration(self):
        """Test login with default accounts and new user registration."""
        # Test default doctor login
        user = authenticate_user("doctor", "doctor123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "doctor")

        # Test default admin login
        admin = authenticate_user("admin", "admin123")
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "admin")

        # Test registration
        res_reg = self.client.post("/register", data={
            "full_name": "Dr. Test Clinician",
            "username": "testdoc",
            "email": "testdoc@retinanova.med",
            "password": "password123",
            "confirm_password": "password123"
        }, follow_redirects=True)
        self.assertEqual(res_reg.status_code, 200)
        self.assertIn(b"Dr. Test Clinician", res_reg.data)

    def test_03_image_preprocessing_and_inference(self):
        """Test Ben Graham preprocessing and Hybrid Model prediction."""
        sample_path = "static/images/samples/stage2_moderate_dr.png"
        self.assertTrue(os.path.exists(sample_path), "Sample image should exist")

        # Test validation
        valid, msg = validate_image_file("stage2_moderate_dr.png")
        self.assertTrue(valid)

        # Test preprocessing
        enhanced_out = "uploads/enhanced/test_enhanced.png"
        input_tensor, enh_img, quality = preprocess_fundus_image(
            sample_path,
            save_enhanced_path=enhanced_out
        )
        self.assertEqual(input_tensor.shape, (1, 224, 224, 3))
        self.assertTrue(os.path.exists(enhanced_out))

        # Test prediction
        pred_res = run_prediction(input_tensor, quality)
        self.assertIn("predicted_class", pred_res)
        self.assertIn("confidence", pred_res)
        self.assertIn("probabilities", pred_res)
        self.assertEqual(len(pred_res["probabilities"]), 5)
        print(f"\n[Test Result] Stage: {pred_res['stage_name']} | Confidence: {pred_res['confidence']}%")

    def test_04_pdf_report_generation(self):
        """Test generating clinical PDF report."""
        sample_path = "static/images/samples/stage0_normal.png"
        enhanced_path = "static/images/samples/stage0_normal.png"
        test_pdf = "uploads/reports/test_report.pdf"

        dummy_pred = {
            "predicted_class": 0,
            "predicted_stage": "Stage 0",
            "stage_name": "No DR (Normal)",
            "risk_level": "Normal / Healthy",
            "confidence": 98.5,
            "probabilities": [98.5, 1.0, 0.3, 0.1, 0.1],
            "urgency": "Routine",
            "description": "Retinal vasculature normal and intact.",
            "clinical_findings": ["No microaneurysms", "Clear fundus background"],
            "recommendations": ["Annual screening", "Glycemic control"]
        }

        out_path = generate_pdf_report(
            output_pdf_path=test_pdf,
            scan_id=999,
            user_name="Dr. Test User",
            original_img_path=sample_path,
            enhanced_img_path=enhanced_path,
            prediction_data=dummy_pred
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_05_sample_scan_workflow(self):
        """Test complete clinician workflow from sample scan to result view."""
        # Login as doctor
        with self.client:
            self.client.post("/login", data={
                "identifier": "doctor",
                "password": "doctor123"
            }, follow_redirects=True)

            # Trigger sample scan
            res = self.client.get("/sample-scan/stage1_mild_dr.png", follow_redirects=True)
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Retinal Scan #", res.data)
            self.assertIn(b"Download PDF Report", res.data)

            # Check dashboard has updated
            dash_res = self.client.get("/dashboard")
            self.assertEqual(dash_res.status_code, 200)
            self.assertIn(b"Your Analyzed Scans", dash_res.data)

    def test_06_admin_dashboard(self):
        """Test admin dashboard access control and telemetry display."""
        # Login as admin
        with self.client:
            self.client.post("/login", data={
                "identifier": "admin",
                "password": "admin123"
            }, follow_redirects=True)

            admin_res = self.client.get("/admin")
            self.assertEqual(admin_res.status_code, 200)
            self.assertIn(b"Registered Clinicians", admin_res.data)
            self.assertIn(b"System-Wide Scans", admin_res.data)


if __name__ == "__main__":
    unittest.main()

