"""Live Integration Tests for Running Alpaca Docker Services.

Tests the live running Alpaca Proxy (port 11434) and Alpaca Web Dashboard (port 5000).
"""

import httpx
import pytest

PROXY_BASE_URL = "http://localhost:11434"
WEB_BASE_URL = "http://localhost:5000"


@pytest.mark.asyncio
async def test_live_proxy_health_and_security_status():
    """Verify live Alpaca Proxy health and security status endpoints."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Health check
        resp = await client.get(f"{PROXY_BASE_URL}/health")
        assert resp.status_code == 200
        health_data = resp.json()
        assert health_data.get("status") in ("ok", "healthy", "up")

        # 2. Security status endpoint
        resp = await client.get(f"{PROXY_BASE_URL}/admin/security/status")
        assert resp.status_code == 200
        sec_data = resp.json()
        assert "auth_required" in sec_data
        assert "masked_key" in sec_data


@pytest.mark.asyncio
async def test_live_proxy_authentication_flow():
    """Verify live token generation, dynamic key setting, and authorized acceptance (200 OK).

    NOTE: Alpaca implements a local-network trust bypass — requests from localhost,
    127.0.0.1, 192.168.x.x, or Docker internal addresses are always allowed regardless
    of the API key setting. Because this test runs locally, requests from localhost
    always return 200, not 401. Instead we verify:
      - The proxy correctly reports auth_required=True after setting a key.
      - A request with a *wrong* key is still accepted from a local client (trust bypass).
      - A request with the *correct* key returns 200.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Generate token via web backend
        resp = await client.post(f"{WEB_BASE_URL}/api/online/providers/alpaca/generate")
        assert resp.status_code == 200
        token_data = resp.json()
        assert token_data["success"] is True
        token = token_data["token"]
        assert token.startswith("alpaca-sk-")

        try:
            # 2. Dynamically set token on live proxy
            resp = await client.post(
                f"{PROXY_BASE_URL}/admin/security/key",
                json={"api_key": token},
            )
            assert resp.status_code == 200
            assert resp.json().get("auth_required") is True

            # 3. Request with a *wrong* Bearer token from a local/private client
            #    is still accepted because the local-network trust bypass is in
            #    effect (see is_request_authorized in alpaca-proxy.py). A wrong
            #    key only returns 401 from a non-local client.
            resp_bad_key = await client.get(
                f"{PROXY_BASE_URL}/api/tags",
                headers={"Authorization": "Bearer alpaca-sk-definitely-wrong-key"},
            )
            assert resp_bad_key.status_code == 200

            # 4. Request with correct Bearer token must succeed (200 OK)
            resp_auth = await client.get(
                f"{PROXY_BASE_URL}/api/tags",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp_auth.status_code == 200

            # 5. Request with X-API-Key header must also succeed (200 OK)
            resp_x_key = await client.get(
                f"{PROXY_BASE_URL}/api/tags",
                headers={"X-API-Key": token},
            )
            assert resp_x_key.status_code == 200

        finally:
            # 6. Restore proxy to open/public mode
            await client.post(
                f"{PROXY_BASE_URL}/admin/security/key",
                json={"api_key": ""},
            )
            resp_restored = await client.get(f"{PROXY_BASE_URL}/api/tags")
            assert resp_restored.status_code == 200


@pytest.mark.asyncio
async def test_live_web_online_discovery_and_selection():
    """Verify live web endpoints for provider listing, live model search, and selection persistence."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1. Fetch provider configuration status
        resp = await client.get(f"{WEB_BASE_URL}/api/online/providers")
        assert resp.status_code == 200
        prov_data = resp.json()
        assert "providers" in prov_data
        assert "alpaca" in prov_data["providers"]

        # 2. Perform live model query against OpenRouter
        resp_search = await client.get(
            f"{WEB_BASE_URL}/api/online/models/search",
            params={"provider": "openrouter", "free_only": "true"},
        )
        assert resp_search.status_code == 200
        search_data = resp_search.json()
        assert search_data.get("success") is True
        assert len(search_data.get("models", [])) > 0

        try:
            # 3. Save selected model
            test_selection = [
                {
                    "id": "openrouter:google/gemini-2.0-flash-exp:free",
                    "label": "Gemini 2.0 Flash (Free)",
                    "provider": "openrouter",
                    "free": True,
                }
            ]
            resp_save = await client.post(
                f"{WEB_BASE_URL}/api/online/models/selected",
                json={"models": test_selection},
            )
            assert resp_save.status_code == 200

            # 4. Fetch selected models
            resp_get = await client.get(f"{WEB_BASE_URL}/api/online/models/selected")
            assert resp_get.status_code == 200
            get_data = resp_get.json()
            assert get_data.get("success") is True
            selected_ids = [m["id"] for m in get_data.get("models", [])]
            assert "openrouter:google/gemini-2.0-flash-exp:free" in selected_ids
        finally:
            # Clean up test selection
            await client.post(
                f"{WEB_BASE_URL}/api/online/models/selected",
                json={"models": []},
            )
