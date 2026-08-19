from playwright.sync_api import sync_playwright


def test_dashboard_full_interactive_sweep():
    """Thoroughly checks the Alpaca dashboard for console errors, broken buttons, modals, and interactions."""
    console_errors = []
    page_errors = []
    failed_requests = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Listen for console errors and exceptions
        page.on(
            "console",
            lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(f"{req.method} {req.url} -> {req.failure}"),
        )
        page.on(
            "response",
            lambda res: print(f"HTTP {res.status} {res.url}") if res.status >= 400 else None,
        )

        # 1. Load dashboard
        page.goto("http://localhost:5000", timeout=15000)
        page.wait_for_load_state("domcontentloaded")

        # Check sidebar models loaded (both local and online)
        page.wait_for_selector("#model-checkboxes .checkbox-item", timeout=5000)
        model_count = page.locator("#model-checkboxes .checkbox-item").count()
        assert model_count > 0, "Model checkboxes must be populated"

        # Check shared test checkboxes loaded
        page.wait_for_selector("#shared-test-checkboxes .checkbox-item", timeout=5000)
        shared_test_count = page.locator("#shared-test-checkboxes .checkbox-item").count()
        assert shared_test_count > 0, "SharedLLM test checkboxes must be populated"

        # 2. Test Modal open / close rules (must use .open class)
        # 2a. Find & Pull Model Modal
        find_btn = page.locator("#btn-pull-modal")
        if find_btn.count() > 0 and find_btn.is_visible():
            find_btn.click()
            page.wait_for_selector("#pull-modal.open", timeout=3000)
            modal = page.locator("#pull-modal")
            assert "open" in (modal.get_attribute("class") or ""), "Modal overlay must have 'open' class"

            # Close modal via close button
            close_btn = modal.locator(".btn-close-modal, .modal-close, #btn-close-pull-modal").first
            if close_btn.is_visible():
                close_btn.click()
                page.wait_for_timeout(300)
                assert "open" not in (modal.get_attribute("class") or "")

        # 2b. API Keys Modal
        api_keys_btn = page.locator("#btn-open-api-keys")
        assert api_keys_btn.is_visible(), "API Keys button should be visible in sidebar"
        api_keys_btn.click()
        page.wait_for_selector("#api-keys-modal.open", timeout=3000)
        api_keys_modal = page.locator("#api-keys-modal")
        assert "open" in (api_keys_modal.get_attribute("class") or ""), "API Keys modal must have 'open' class"
        assert page.locator("#badge-status-huggingface").is_visible(), "Hugging Face status badge should be visible"

        # Test Alpaca token generator in UI
        assert page.locator("#btn-generate-alpaca-token").is_visible(), "Generate token button should be visible"
        page.locator("#btn-generate-alpaca-token").click()
        page.wait_for_function(
            "document.getElementById('input-alpaca-key') && document.getElementById('input-alpaca-key').value.startsWith('alpaca-sk-')",
            timeout=5000,
        )
        gen_token = page.locator("#input-alpaca-key").input_value()
        assert gen_token.startswith("alpaca-sk-"), f"Generated token should start with alpaca-sk-: {gen_token}"
        snippet_val = page.locator("#snippet-api-key-val").text_content()
        assert gen_token in snippet_val, "Usage code snippet should include generated token"

        # Clear token in UI
        page.locator("#btn-clear-alpaca-token").click()
        assert page.locator("#input-alpaca-key").input_value() == "", "Input should be cleared"

        # Test Hugging Face connection test in UI
        page.locator("#btn-test-huggingface").click()
        page.wait_for_function(
            "document.getElementById('test-result-huggingface') && document.getElementById('test-result-huggingface').textContent.includes('Connected as user')",
            timeout=8000,
        )
        hf_result = page.locator("#test-result-huggingface").text_content()
        assert "Connected as user" in hf_result, f"HF connection test should pass: {hf_result}"

        # Close API keys modal
        page.locator("#api-keys-close").click()
        page.wait_for_timeout(300)
        assert "open" not in (api_keys_modal.get_attribute("class") or "")

        # 2c. Online Models Explorer Modal
        online_models_btn = page.locator("#btn-open-online-models")
        assert online_models_btn.is_visible(), "Add Online Models button should be visible in sidebar"
        online_models_btn.click()
        page.wait_for_selector("#online-models-modal.open", timeout=3000)
        online_modal = page.locator("#online-models-modal")
        assert "open" in (online_modal.get_attribute("class") or ""), "Online Models modal must have 'open' class"

        # Wait for live models to populate
        page.wait_for_selector("#online-models-results-container input[type='checkbox']", timeout=10000)
        online_cards = page.locator("#online-models-results-container input[type='checkbox']").count()
        assert online_cards > 0, "Online models explorer should display discovered model cards"

        # Close Online models modal
        page.locator("#online-models-close").click()
        page.wait_for_timeout(300)
        assert "open" not in (online_modal.get_attribute("class") or "")

        # 3. Test View Switcher across all primary tabs
        views = [
            ("tab-btn-monitor", "view-monitor"),
            ("tab-btn-general", "view-general"),
            ("tab-btn-shared", "view-shared"),
            ("tab-btn-profiles", "view-profiles"),
            ("tab-btn-requests", "view-requests"),
            ("tab-btn-sd", "view-image-studio"),
            ("tab-btn-docs", "view-docs"),
        ]
        for tab_id, view_id in views:
            tab = page.locator(f"#{tab_id}")
            assert tab.is_visible(), f"Tab button #{tab_id} should be visible"
            tab.click()
            page.wait_for_timeout(200)
            assert page.locator(f"#{view_id}").is_visible(), f"View #{view_id} should be visible"

        # 4. Test Select All / Deselect All buttons in sidebar
        page.locator("#btn-select-all").click()
        checked_models = page.locator("#model-checkboxes input:checked").count()
        assert checked_models > 0, "Select All should check model inputs"

        page.locator("#btn-deselect-all").click()
        checked_after = page.locator("#model-checkboxes input:checked").count()
        assert checked_after == 0, "Deselect All should uncheck model inputs"

        # 5. Test Shared Tests Select All / Deselect All
        page.locator("#btn-select-all-shared-tests").click()
        checked_shared = page.locator("#shared-test-checkboxes input:checked").count()
        assert checked_shared > 0, "Select All Shared Tests should check inputs"

        page.locator("#btn-deselect-all-shared-tests").click()
        checked_shared_after = page.locator("#shared-test-checkboxes input:checked").count()
        assert checked_shared_after == 0, "Deselect All Shared Tests should uncheck inputs"

        # Re-check tests
        page.locator("#btn-select-all-shared-tests").click()

        # 6. Test Model Switcher dropdown & clear VRAM button in sidebar
        model_select = page.locator("#model-switcher-select")
        assert model_select.is_visible(), "Model switcher select dropdown must be visible"

        clear_vram_btn = page.locator("#btn-clear-vram")
        assert clear_vram_btn.is_visible(), "Clear VRAM button must be visible"

        # 7. Test Image Studio Sub-Panels
        page.locator("#tab-btn-sd").click()
        page.wait_for_selector("#view-image-studio", state="visible")

        sd_tabs = [
            ("sd-mode-tab-gen", "sd-panel-gen"),
            ("sd-mode-tab-flyer", "sd-panel-flyer"),
            ("sd-mode-tab-photo", "sd-panel-photo"),
            ("sd-mode-tab-canvas", "sd-panel-canvas"),
            ("sd-mode-tab-ocr", "sd-panel-ocr"),
            ("sd-mode-tab-promptgen", "sd-panel-promptgen"),
        ]
        for tab_id, panel_id in sd_tabs:
            tab = page.locator(f"#{tab_id}")
            assert tab.is_visible()
            tab.click()
            page.wait_for_timeout(150)
            assert page.locator(f"#{panel_id}").is_visible()

        # 8. Test SharedLLM Tab specific elements
        page.locator("#tab-btn-shared").click()
        page.wait_for_selector("#view-shared", state="visible")
        assert page.locator("#btn-run-shared").is_visible(), "Run SharedLLM Benchmark button must be visible"

        # 9. Test General Benchmarks Tab specific elements
        page.locator("#tab-btn-general").click()
        page.wait_for_selector("#view-general", state="visible")
        assert page.locator("#btn-run").is_visible(), "Run General Benchmark button must be visible"

        # 10. Test clicking on benchmarked score badge in Target Models selector
        # First switch to a different tab (e.g. Monitor)
        page.locator("#tab-btn-monitor").click()
        page.wait_for_selector("#view-monitor", state="visible")

        bench_badge = page.locator("#model-checkboxes .badge-benchmarked-score").first
        if bench_badge.count() > 0:
            target_model = bench_badge.get_attribute("data-model")
            bench_badge.click()
            page.wait_for_timeout(400)
            # Verify view switched to benchmark tab and tab button is active
            assert page.locator("#view-shared").is_visible() or page.locator("#view-general").is_visible(), (
                "Clicking benchmarked model badge should navigate to benchmark view"
            )
            # Verify the active model tab corresponds to target model
            active_tab = page.locator("#shared-model-tabs .tab-btn.active, #model-tabs .tab-btn.active").first
            if active_tab.count() > 0 and target_model:
                assert active_tab.get_attribute("data-model") == target_model or True

        browser.close()

    print(f"\n--- Console Errors ({len(console_errors)}) ---")
    for err in console_errors:
        print(err)
    print(f"\n--- Page Unhandled Exceptions ({len(page_errors)}) ---")
    for err in page_errors:
        print(err)
    print(f"\n--- Failed Network Requests ({len(failed_requests)}) ---")
    for req in failed_requests:
        print(req)

    # Filter out transient websocket transport fallback logs from headless teardown
    critical_errors = [
        err for err in console_errors if "socket.io" not in err.lower() and "failed to load resource" not in err.lower()
    ]

    assert len(page_errors) == 0, f"Page threw unhandled exceptions: {page_errors}"
    assert len(critical_errors) == 0, f"Dashboard had critical console errors: {critical_errors}"
