import re
import pytest
from playwright.sync_api import sync_playwright


def get_dashboard_page(p):
    """Helper to launch browser and navigate to dashboard with fallback"""
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    try:
        page.goto("http://localhost:5000", timeout=10000)
    except Exception:
        try:
            page.goto("http://127.0.0.1:5000", timeout=10000)
        except Exception as e:
            browser.close()
            pytest.skip(f"Could not connect to live dashboard server at port 5000: {e}")
    return browser, page


def test_main_navigation_tabs():
    """Verify navigation to all top-level tabs and container panel visibilities"""
    with sync_playwright() as p:
        browser, page = get_dashboard_page(p)

        tabs_to_test = [
            ("#tab-btn-monitor", "#view-monitor"),
            ("#tab-btn-general", "#view-general"),
            ("#tab-btn-shared", "#view-shared"),
            ("#tab-btn-profiles", "#view-profiles"),
            ("#tab-btn-requests", "#view-requests"),
            ("#tab-btn-sd", "#view-image-studio"),
            ("#tab-btn-docs", "#view-docs"),
        ]

        for tab_id, view_id in tabs_to_test:
            btn = page.locator(tab_id)
            assert btn.is_visible(), f"Tab button {tab_id} must be visible"
            btn.click()
            page.wait_for_selector(view_id, state="visible", timeout=3000)
            assert page.locator(view_id).is_visible(), f"Panel {view_id} must be visible after clicking {tab_id}"
            assert "active" in btn.get_attribute("class"), f"Tab button {tab_id} must have active class"

        browser.close()


def test_image_studio_sub_tabs():
    """Verify switching between all 6 mode tabs within Image Studio"""
    with sync_playwright() as p:
        browser, page = get_dashboard_page(p)

        # Open Image Studio tab
        page.locator("#tab-btn-sd").click()
        page.wait_for_selector("#view-image-studio", state="visible", timeout=3000)

        sub_tabs = [
            ("#sd-mode-tab-gen", "#sd-panel-gen"),
            ("#sd-mode-tab-flyer", "#sd-panel-flyer"),
            ("#sd-mode-tab-photo", "#sd-panel-photo"),
            ("#sd-mode-tab-canvas", "#sd-panel-canvas"),
            ("#sd-mode-tab-ocr", "#sd-panel-ocr"),
            ("#sd-mode-tab-promptgen", "#sd-panel-promptgen"),
        ]

        for tab_id, panel_id in sub_tabs:
            tab = page.locator(tab_id)
            assert tab.is_visible(), f"Sub-tab button {tab_id} must be visible"
            tab.click()
            page.wait_for_selector(panel_id, state="visible", timeout=3000)
            assert page.locator(panel_id).is_visible(), f"Sub-panel {panel_id} must be visible after clicking {tab_id}"

        browser.close()


def test_api_docs_tab_navigation():
    """Verify API Docs tab endpoint menu items click and activate doc panels"""
    with sync_playwright() as p:
        browser, page = get_dashboard_page(p)

        page.locator("#tab-btn-docs").click()
        page.wait_for_selector("#view-docs", state="visible", timeout=3000)

        menu_items = page.locator(".doc-menu-item")
        count = menu_items.count()
        assert count > 0, "API Docs tab should display endpoint menu items"

        # Click first 3 menu items and verify selection
        for i in range(min(count, 3)):
            item = menu_items.nth(i)
            item.click()
            assert "active" in item.get_attribute("class")

        browser.close()


def test_image_to_prompt_assistant_tab():
    """Verify Image-to-Prompt Assistant controls, prompt synthesis, and prompt transfer to Photo Editor"""
    with sync_playwright() as p:
        browser, page = get_dashboard_page(p)

        # Intercept route for prompt synthesis API to return deterministic response
        page.route(re.compile(r".*/api/vision/synthesize_edit_prompt"), lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"success", "master_prompt":"A futuristic city street at night, neon purple hovering car, cyan rain reflections, Cyberpunk Sci-Fi aesthetic, 8k resolution, raw photo"}'
        ))

        # Navigate to Image Studio and open Image-to-Prompt tab
        page.locator("#tab-btn-sd").click()
        page.wait_for_selector("#sd-mode-tab-promptgen", timeout=3000)
        page.locator("#sd-mode-tab-promptgen").click()

        # Check element visibilities
        assert page.locator("#sd-panel-promptgen").is_visible()
        assert page.locator("#sd-promptgen-dropzone").is_visible()
        assert page.locator("#sd-promptgen-desc").is_visible()
        assert page.locator("#sd-promptgen-changes").is_visible()
        assert page.locator("#sd-promptgen-preset").is_visible()
        assert page.locator("#sd-promptgen-synth-btn").is_visible()

        # Fill text inputs
        page.fill("#sd-promptgen-desc", "A high-resolution photograph of a futuristic city street at night")
        page.fill("#sd-promptgen-changes", "Add a neon purple hovering car and glowing cyan rain reflections")
        page.select_option("#sd-promptgen-preset", "Cyberpunk Sci-Fi")

        # Click Synthesize button
        page.locator("#sd-promptgen-synth-btn").click()

        # Wait for master prompt output container to update
        page.wait_for_selector("#sd-promptgen-result-prompt", timeout=5000)
        page.wait_for_function("document.getElementById('sd-promptgen-result-prompt').textContent.includes('neon purple')")
        result_text = page.locator("#sd-promptgen-result-prompt").text_content()
        assert "neon purple hovering car" in result_text

        # Dismiss dialog alert automatically when clicking send photo button
        page.on("dialog", lambda dialog: dialog.accept())

        # Click 'Send to Photo Editor' button and verify prompt transfer & tab switch
        page.locator("#sd-promptgen-send-photo-btn").click()
        assert page.locator("#sd-panel-photo").is_visible(), "Photo Editor panel must be active after prompt transfer"
        photo_input_val = page.locator("#sd-edit-prompt").input_value()
        assert "neon purple hovering car" in photo_input_val

        browser.close()


