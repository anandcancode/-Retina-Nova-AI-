"""
Selenium End-to-End Headless Browser Automation Test & Screenshot Generator
Navigates the real live web application, tests interactions, captures full-page screenshots.
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

SCREENSHOT_DIR = os.path.abspath("static/screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

edge_options = Options()
edge_options.add_argument("--headless=new")
edge_options.add_argument("--disable-gpu")
edge_options.add_argument("--no-sandbox")
edge_options.add_argument("--window-size=1280,960")

print("[Browser Test] Initializing Headless Edge WebDriver...")
driver = webdriver.Edge(options=edge_options)

try:
    BASE_URL = "http://127.0.0.1:5000"

    # 1. Test Home Page
    print("\n--- 1. Testing Home Page ---")
    driver.get(f"{BASE_URL}/")
    time.sleep(1.5)
    home_shot = os.path.join(SCREENSHOT_DIR, "01_homepage.png")
    driver.save_screenshot(home_shot)
    print(f"Captured: {home_shot} (Title: {driver.title})")

    # 2. Test About Page
    print("\n--- 2. Testing About Page ---")
    driver.get(f"{BASE_URL}/about")
    time.sleep(1.5)
    about_shot = os.path.join(SCREENSHOT_DIR, "02_about.png")
    driver.save_screenshot(about_shot)
    print(f"Captured: {about_shot} (Title: {driver.title})")

    # 3. Test Login Page
    print("\n--- 3. Testing Login Page ---")
    driver.get(f"{BASE_URL}/login")
    time.sleep(1.5)
    login_shot = os.path.join(SCREENSHOT_DIR, "03_login.png")
    driver.save_screenshot(login_shot)
    print(f"Captured: {login_shot}")

    # 4. Perform Login as Clinician
    print("\n--- 4. Logging in as Clinician (doctor / doctor123) ---")
    driver.find_element(By.ID, "identifier").send_keys("doctor")
    driver.find_element(By.ID, "password").send_keys("doctor123")
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(2)

    # 5. Dashboard View
    print("\n--- 5. Testing Clinician Dashboard ---")
    dash_shot = os.path.join(SCREENSHOT_DIR, "04_dashboard.png")
    driver.save_screenshot(dash_shot)
    print(f"Captured: {dash_shot} (URL: {driver.current_url})")

    # 6. Upload Page View
    print("\n--- 6. Testing Upload Page ---")
    driver.get(f"{BASE_URL}/upload")
    time.sleep(1.5)
    upload_shot = os.path.join(SCREENSHOT_DIR, "05_upload.png")
    driver.save_screenshot(upload_shot)
    print(f"Captured: {upload_shot}")

    # 7. Test Analyzing a Retinal Fundus Scan
    print("\n--- 7. Running Retinal Scan Analysis via Benchmark Sample ---")
    # Click the "Moderate NPDR" sample benchmark pill
    sample_btn = driver.find_element(By.XPATH, "//a[contains(@href, 'stage2_moderate_dr.png')]")
    sample_btn.click()
    print("Clicked benchmark sample scan, waiting for AI inference and report generation...")
    time.sleep(4)

    # 8. Result Page View
    print("\n--- 8. Testing Diagnostic Result Page ---")
    result_shot = os.path.join(SCREENSHOT_DIR, "06_result.png")
    driver.save_screenshot(result_shot)
    print(f"Captured: {result_shot} (URL: {driver.current_url})")

    # 9. Test Admin Portal
    print("\n--- 9. Logging out & Signing into Admin Portal ---")
    driver.get(f"{BASE_URL}/logout")
    time.sleep(1)
    driver.get(f"{BASE_URL}/login")
    time.sleep(1)
    driver.find_element(By.ID, "identifier").send_keys("admin")
    driver.find_element(By.ID, "password").send_keys("admin123")
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(2)

    driver.get(f"{BASE_URL}/admin")
    time.sleep(2)
    admin_shot = os.path.join(SCREENSHOT_DIR, "07_admin.png")
    driver.save_screenshot(admin_shot)
    print(f"Captured: {admin_shot} (URL: {driver.current_url})")

    print("\n[Browser Test] ALL 7 BROWSER SCENARIOS EXECUTED & CAPTURED SUCCESSFULLY!")

finally:
    driver.quit()

