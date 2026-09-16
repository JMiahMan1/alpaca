import ast
import contextlib
import os
import threading
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("ALPACA_AUDIO_TEST_IMAGE"), reason="Set ALPACA_AUDIO_TEST_IMAGE to a sandbox image"
)
def test_browser_audio_latency_and_recovery():
    import docker
    import httpx
    from flask import Flask, Response, jsonify
    from werkzeug.serving import make_server

    import sandbox_exec

    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[1]
    docker_client = docker.from_env()
    container = None
    server = None
    try:
        container = docker_client.containers.run(
            os.environ["ALPACA_AUDIO_TEST_IMAGE"],
            ["sleep", "infinity"],
            detach=True,
            ports={"8090/tcp": ("127.0.0.1", None)},
        )
        container.reload()
        port = container.ports["8090/tcp"][0]["HostPort"]
        result = container.exec_run(["bash", "-c", sandbox_exec._AUDIO_SETUP], user="sandbox")
        assert result.exit_code == 0, result.output
        result = container.exec_run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.25", "/tmp/beep.wav"]
        )
        assert result.exit_code == 0, result.output
        source = (root / "web/app.py").read_text()
        handler = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "sandbox_serve_audio"
        )
        route = ast.get_source_segment(source, handler)
        assert route is not None
        namespace = {
            "contextlib": contextlib,
            "httpx": httpx,
            "time": time,
            "jsonify": jsonify,
            "Response": Response,
            "ensure_audio_encoder": lambda _: container.exec_run(
                ["bash", "-c", sandbox_exec._AUDIO_FFMPEG_SH], user="sandbox", detach=True
            ),
            "_serve_container_host_port": lambda *args: port,
        }
        exec(route.replace("host.docker.internal", "127.0.0.1"), namespace)
        app = Flask(__name__)
        app.add_url_rule("/serve/audio/test", view_func=lambda: namespace["sandbox_serve_audio"]("test"))
        app.add_url_rule(
            "/",
            endpoint="index",
            view_func=lambda: '<audio id="game-audio" preload="none"></audio><button id="btn-sound">Sound</button>',
        )
        server = make_server("127.0.0.1", 0, app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        html = (root / "web/templates/ui_launcher.html").read_text()
        controls = html[html.index("        const audioEl =") : html.index("        btnSound.addEventListener('click'")]
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.evaluate("""async () => {
                    window.heard = []; window.events = [];
                    const audio = document.getElementById('game-audio');
                    const ctx = new AudioContext({latencyHint: 'interactive'});
                    const analyser = ctx.createAnalyser(); analyser.fftSize = 256;
                    ctx.createMediaElementSource(audio).connect(analyser);
                    analyser.connect(ctx.destination);
                    await ctx.resume();
                    const data = new Float32Array(256);
                    let last = -Infinity;
                    setInterval(() => {
                        analyser.getFloatTimeDomainData(data);
                        if (Math.max(...data.map(Math.abs)) > 0.02 && performance.now() - last > 500) {
                            last = performance.now(); heard.push(last);
                        }
                    }, 5);
                }""")
                page.evaluate(
                    "() => { const CONTAINER_ID = 'test'; const report = (...args) => events.push(args);"
                    "const setStatus = () => {};" + controls + "; window.setAudio = setAudio; setAudio(true); }"
                )

                def wait_live():
                    page.wait_for_function("events.some(e => e[0] === 'audio-playing')", timeout=15000)
                    page.wait_for_timeout(1000)

                def beep_latency():
                    page.evaluate("heard.length = 0")
                    trigger = page.evaluate("performance.now()")
                    result = container.exec_run(
                        [
                            "bash",
                            "-c",
                            "export XDG_RUNTIME_DIR=/tmp/pulse-$(id -u); paplay --latency-msec=20 /tmp/beep.wav",
                        ],
                        user="sandbox",
                    )
                    assert result.exit_code == 0, result.output
                    page.wait_for_function("heard.length > 0", timeout=4000)
                    latency = page.evaluate("heard[0]") - trigger
                    assert 0 < latency < 1000, f"Audio latency {latency:.0f} ms"
                    page.wait_for_timeout(600)
                    return round(latency)

                wait_live()
                latencies = [beep_latency(), beep_latency()]
                page.evaluate("setAudio(false); events.length = 0")
                page.wait_for_timeout(1000)
                page.evaluate("setAudio(true)")
                wait_live()
                latencies.append(beep_latency())
                page.evaluate("document.getElementById('game-audio').pause()")
                page.wait_for_timeout(7000)
                page.evaluate("document.getElementById('game-audio').play()")
                page.wait_for_function(
                    """() => {
                    const a = document.getElementById('game-audio');
                    return !a.seeking && a.buffered.length && a.buffered.end(a.buffered.length - 1) - a.currentTime < 0.5;
                }""",
                    timeout=3000,
                )
                latencies.append(beep_latency())
                page.wait_for_timeout(12000)
                latencies.append(beep_latency())
                assert page.evaluate("document.getElementById('game-audio').buffered.start(0)") > 0
                assert not errors
                assert not page.evaluate("events.filter(e => ['audio-error', 'audio-blocked'].includes(e[0]))")
                print(f"Trigger-to-decoded-audio latency (ms): {latencies}")
            finally:
                browser.close()
    finally:
        if server is not None:
            server.shutdown()
        if container is not None:
            container.remove(force=True)
        docker_client.close()
