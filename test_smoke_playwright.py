import pytest
from playwright.sync_api import sync_playwright


def test_dashboard_smoke():
    """Verify that dashboard loads and model checkboxes are not checked by default"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("http://localhost:5000", timeout=10000)
        except Exception:
            try:
                page.goto("http://127.0.0.1:5000", timeout=10000)
            except Exception as e:
                pytest.skip(f"Could not connect to live dashboard server at port 5000: {e}")

        # 1. Verify Page Title
        assert "Alpaca" in page.title()

        # 2. Verify Navigation tabs work
        general_btn = page.locator("#tab-btn-general")
        assert general_btn.is_visible()
        general_btn.click()

        # 3. Verify target models checkboxes are not checked by default
        page.wait_for_selector("#model-checkboxes", timeout=5000)
        checkboxes = page.locator("#model-checkboxes input[type='checkbox']")

        count = checkboxes.count()
        for i in range(count):
            assert not checkboxes.nth(i).is_checked(), (
                f"Checkbox at index {i} was checked by default"
            )

        # 4. Verify system monitor tab is clickable
        monitor_btn = page.locator("#tab-btn-monitor")
        assert monitor_btn.is_visible()
        monitor_btn.click()

        browser.close()
