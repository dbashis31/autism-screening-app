"""
Full demo walkthrough with Playwright video capture.

Exercises positive test cases (S-03, S-08, S-09, S-02, S-10),
negative cases (S-04, S-06), clinician/admin dashboards,
and image prediction with Grad-CAM.

Usage:
    python demo_video_capture.py
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://localhost:5173"
APP_DIR = Path(__file__).parent
VIDEO_DIR = APP_DIR / "demo_recordings"
VIDEO_DIR.mkdir(exist_ok=True)

for f in VIDEO_DIR.iterdir():
    f.unlink()


def wait(secs=1.5):
    time.sleep(secs)


def caregiver_session(page, child_id, label, prefix,
                      snr=None, age=None, conflict=False, force_abstain=False):
    """Run one full caregiver flow: consent → configure → submit → view result."""
    print(f"  -> {label}")
    page.goto(BASE + "/caregiver")
    page.evaluate("localStorage.setItem('role','caregiver')")
    wait(1)

    # Step 1: Fill consent form
    child_input = page.locator('input[placeholder*="SYN-001"]')
    child_input.wait_for(state="visible", timeout=5000)
    child_input.fill(child_id)
    wait(0.3)

    consent_btn = page.locator('button[type="submit"]')
    consent_btn.click()
    wait(2.5)

    # Step 2: Configure modalities
    submit_btn = page.locator('button[type="submit"]')
    submit_btn.wait_for(state="visible", timeout=5000)

    if snr is not None:
        slider = page.locator('input[type="range"]')
        if slider.count() > 0:
            slider.fill(str(snr))
            wait(0.3)

    if age is not None:
        age_input = page.locator('input[type="number"]')
        if age_input.count() > 0:
            age_input.fill(str(age))
            wait(0.3)

    if conflict or force_abstain:
        summary = page.locator('summary')
        if summary.count() > 0:
            summary.click()
            wait(0.3)
            checkboxes = page.locator('details input[type="checkbox"]')
            if conflict and checkboxes.count() >= 1:
                checkboxes.nth(0).check()
                wait(0.2)
            if force_abstain and checkboxes.count() >= 2:
                checkboxes.nth(1).check()
                wait(0.2)

    page.screenshot(path=str(VIDEO_DIR / f"{prefix}_a_config.png"))

    submit_btn.click()
    wait(4)

    page.screenshot(path=str(VIDEO_DIR / f"{prefix}_b_result.png"))
    wait(1.5)

    return page


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 900},
        )
        page = ctx.new_page()

        # Initial load
        page.goto(BASE + "/caregiver")
        page.evaluate("localStorage.setItem('role','caregiver')")
        wait(1)

        # =================================================================
        # POSITIVE TEST CASES — pipeline completes successfully
        # =================================================================
        print("\n=== POSITIVE TEST CASES ===")

        caregiver_session(page, "CHILD-ALPHA",
                          "S-03: High confidence — full complete report",
                          "01_s03_positive")

        caregiver_session(page, "CHILD-BETA",
                          "S-08: Standard vocabulary compliance check",
                          "02_s08_positive")

        caregiver_session(page, "CHILD-GAMMA",
                          "S-09: Age boundary (18mo) — warns but completes",
                          "03_s09_positive", age=18)

        caregiver_session(page, "CHILD-DELTA",
                          "S-02: Low SNR (10dB) — audio excluded, completes",
                          "04_s02_positive", snr=10)

        caregiver_session(page, "CHILD-EPSILON",
                          "S-10: Mid-session scope change — completes",
                          "05_s10_positive")

        # =================================================================
        # NEGATIVE / GOVERNANCE TEST CASES
        # =================================================================
        print("\n=== NEGATIVE / GOVERNANCE TEST CASES ===")

        caregiver_session(page, "CHILD-ZETA",
                          "S-04: Cross-modal conflict → abstention",
                          "06_s04_negative", conflict=True)

        caregiver_session(page, "CHILD-ESCALATE",
                          "S-06a: Force abstention (1st)",
                          "07_s06a_negative", force_abstain=True)

        caregiver_session(page, "CHILD-ESCALATE",
                          "S-06b: Force abstention (2nd → escalation)",
                          "08_s06b_negative", force_abstain=True)

        # =================================================================
        # CLINICIAN DASHBOARD
        # =================================================================
        print("\n=== CLINICIAN DASHBOARD ===")

        page.evaluate("localStorage.setItem('role','clinician')")
        page.goto(BASE + "/clinician")
        wait(3)
        page.screenshot(path=str(VIDEO_DIR / "09_clinician_overview.png"))

        refresh_btn = page.locator('button:has-text("Refresh")')
        if refresh_btn.count() > 0:
            refresh_btn.click()
            wait(3)

        page.screenshot(path=str(VIDEO_DIR / "10_clinician_sessions.png"))

        page.evaluate("window.scrollTo(0, 500)")
        wait(2)

        raw_btns = page.locator('button:has-text("Show raw")')
        if raw_btns.count() > 0:
            raw_btns.first.click()
            wait(2)
            page.screenshot(path=str(VIDEO_DIR / "11_clinician_raw_json.png"))
            raw_btns.first.click()
            wait(0.5)

        page.evaluate("window.scrollTo(0, 0)")
        wait(1)

        esc = page.locator("text=CHILD-ESCALATE").first
        if esc.is_visible():
            esc.click()
            wait(2)
            page.screenshot(path=str(VIDEO_DIR / "12_escalation_queue.png"))

        wait(1)

        # =================================================================
        # ADMIN DASHBOARD
        # =================================================================
        print("\n=== ADMIN DASHBOARD ===")

        page.evaluate("localStorage.setItem('role','admin')")
        page.goto(BASE + "/admin")
        wait(3)

        refresh_metrics = page.locator('button:has-text("Refresh Metrics")')
        if refresh_metrics.count() > 0:
            refresh_metrics.click()
            wait(3)

        page.screenshot(path=str(VIDEO_DIR / "13_admin_metrics.png"))

        page.evaluate("window.scrollTo(0, 600)")
        wait(2)
        page.screenshot(path=str(VIDEO_DIR / "14_admin_audit_log.png"))

        expand_btns = page.locator('button:has-text("▼")')
        if expand_btns.count() > 0:
            expand_btns.first.click()
            wait(2)
            page.screenshot(path=str(VIDEO_DIR / "15_audit_detail.png"))

        wait(1)

        # =================================================================
        # IMAGE PREDICTION with Grad-CAM
        # =================================================================
        print("\n=== IMAGE PREDICTION (Grad-CAM) ===")

        page.goto(BASE + "/predict")
        wait(2)
        page.screenshot(path=str(VIDEO_DIR / "16_predict_page.png"))

        test_image = APP_DIR / "test_face.jpg"
        if test_image.exists():
            page.evaluate("""
                document.querySelector('input[type="file"]').style.display = 'block';
                document.querySelector('input[type="file"]').style.opacity = '1';
            """)
            wait(0.5)
            page.set_input_files('input[type="file"]', str(test_image))
            wait(2)
            page.screenshot(path=str(VIDEO_DIR / "17_predict_image_loaded.png"))

            predict_btn = page.locator('button:has-text("Run Prediction")')
            if predict_btn.count() > 0 and predict_btn.is_enabled():
                predict_btn.click()
                wait(6)
                page.screenshot(path=str(VIDEO_DIR / "18_predict_result.png"))

                page.evaluate("window.scrollTo(0, 400)")
                wait(2)
                page.screenshot(path=str(VIDEO_DIR / "19_gradcam_overlay.png"))

        wait(2)

        # =================================================================
        # DONE
        # =================================================================
        print("\n=== Demo complete! ===")
        wait(1)
        ctx.close()
        browser.close()

    # Convert video to MP4
    import subprocess
    webm_files = list(VIDEO_DIR.glob("*.webm"))
    for wf in webm_files:
        mp4_path = VIDEO_DIR / "demo_full_walkthrough.mp4"
        subprocess.run([
            "ffmpeg", "-i", str(wf),
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-c:a", "aac", str(mp4_path), "-y"
        ], capture_output=True)
        print(f"\nMP4 video: {mp4_path} ({mp4_path.stat().st_size / 1024 / 1024:.1f} MB)")

    screenshots = sorted(VIDEO_DIR.glob("*.png"))
    print(f"\n{len(screenshots)} screenshots:")
    for s in screenshots:
        print(f"  {s.name}")


if __name__ == "__main__":
    run()
