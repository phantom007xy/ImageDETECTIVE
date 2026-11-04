"""Headless Chrome smoke checks against the real app with a CPU-only fake detector.

Run ``python -m src.eval.release_browser_check`` from the repository root. Chrome
and Node must already exist; this script installs nothing. Only synthetic files
are uploaded. Paid services, model downloads and CUDA initialization are denied.
The app uses an ephemeral loopback port. Browser state stays under .app_state.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import uuid

from src.eval.app_release_check import FakeDetector, image_bytes


@contextlib.contextmanager
def local_app(counters):
    """Run actual ASGI routes on localhost, without loading any learned model."""
    if "app.server" in sys.modules:
        raise RuntimeError("Run browser checks in a fresh Python interpreter")

    def denied(kind):
        def fail(*args, **kwargs):
            counters[kind] += 1
            raise AssertionError(f"Forbidden browser-check operation: {kind}")
        return fail

    original_connect = socket.socket.connect

    def loopback_connect(connection, address):
        host = address[0] if isinstance(address, tuple) else None
        try:
            loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            loopback = False
        if not loopback:
            return denied("external_network")(connection, address)
        return original_connect(connection, address)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            "HEADS_REPO": "", "MODEL_API_KEY": "", "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        }))
        stack.enter_context(patch.object(socket.socket, "connect", loopback_connect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", denied("external_network")))
        import torch
        import huggingface_hub
        import uvicorn

        stack.enter_context(patch.object(torch.cuda, "_lazy_init", denied("cuda")))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        stack.enter_context(patch.object(huggingface_hub, "snapshot_download", denied("download")))
        stack.enter_context(patch.object(huggingface_hub, "hf_hub_download", denied("download")))
        from src import inference

        stack.enter_context(patch.object(inference, "Detector", FakeDetector))
        app_server = importlib.import_module("app.server")
        from src.services import meta_api

        stack.enter_context(patch.object(meta_api, "api_key", return_value=None))
        stack.enter_context(patch.object(meta_api, "post", denied("paid_api")))
        stack.enter_context(patch.object(meta_api, "_record", denied("spending_file_write")))
        stack.enter_context(patch.object(app_server.investigator, "investigate", return_value=None))
        stack.enter_context(patch.object(app_server.sam_outline, "outline", return_value=None))
        stack.enter_context(patch.object(app_server.explain, "explain",
            return_value="No clear AI signs were found. This does not verify the photo's context."))
        stack.enter_context(patch.object(app_server.muse_image, "generate", return_value=None))

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(64)
        config = uvicorn.Config(app_server.app, host="127.0.0.1", port=port,
                                log_level="warning", access_log=False, lifespan="on")
        service = uvicorn.Server(config)
        thread = threading.Thread(target=service.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 12
            while not service.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not service.started:
                raise RuntimeError("Local ASGI app did not start within 12 seconds")
            yield f"http://127.0.0.1:{port}", app_server.detector
        finally:
            service.should_exit = True
            thread.join(10)
            listener.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/release_2026-11-04"))
    parser.add_argument("--chrome", type=Path,
                        default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "browser_checks.json"
    counters = {"external_network": 0, "download": 0, "cuda": 0,
                "paid_api": 0, "spending_file_write": 0}
    report = {"successful": False, "scope": "Real Chrome interactions; actual frontend/ASGI; synthetic uploads; fake CPU detector",
              "external_operations_attempted": counters, "checks": []}
    started = time.perf_counter()
    browser = None
    try:
        node = shutil.which("node")
        if not args.chrome.is_file() or node is None:
            report["limitation"] = "Existing Chrome or Node was unavailable; no tools were installed."
            return_code = 2
        else:
            state = Path(".app_state") / ("browser_release_" + uuid.uuid4().hex)
            profile = state.resolve() / "profile"
            profile.mkdir(parents=True)
            fixture = (state / "synthetic.png").resolve()
            corrupt = (state / "corrupt.png").resolve()
            fixture.write_bytes(image_bytes())
            corrupt.write_bytes(b"synthetic corrupt image; contains no private data")
            child_env = os.environ.copy()
            child_env.pop("MODEL_API_KEY", None)
            child_env["CUDA_VISIBLE_DEVICES"] = ""
            chrome_flags = [str(args.chrome), "--headless=new", "--disable-gpu",
                "--disable-background-networking", "--disable-component-update", "--disable-sync",
                "--disable-default-apps", "--disable-extensions", "--no-first-run", "--no-default-browser-check",
                "--metrics-recording-only", "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
                "--proxy-server=http://127.0.0.1:9", "--proxy-bypass-list=localhost;127.0.0.1;[::1]",
                "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1",
                "--user-data-dir=" + str(profile), "about:blank"]
            with local_app(counters) as (url, detector):
                with (state / "chrome-stderr.log").open("wb") as chrome_log:
                    browser = subprocess.Popen(chrome_flags, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=chrome_log, env=child_env,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    active = profile / "DevToolsActivePort"
                    deadline = time.monotonic() + 15
                    while not active.exists() and browser.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.1)
                    if not active.exists():
                        raise RuntimeError("Chrome CDP endpoint unavailable within 15 seconds")
                    lines = active.read_text(encoding="utf-8").splitlines()
                    websocket = "ws://127.0.0.1:" + lines[0] + lines[1]
                    helper = Path(__file__).with_name("release_browser_check.js")
                    run = subprocess.run([node, str(helper), websocket, url, str(fixture),
                        str(corrupt), str(output)], env=child_env, capture_output=True, text=True,
                        timeout=100, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if report_path.exists():
                        report.update(json.loads(report_path.read_text(encoding="utf-8")))
                    else:
                        raise RuntimeError("Browser helper did not produce its JSON report")
                    report["detector_calls"] = len(detector.calls)
                    report["external_operations_attempted"] = counters
                    report["successful"] = bool(report.get("successful")) and not any(counters.values())
                    return_code = 0 if report["successful"] and run.returncode == 0 else 1
    except Exception as error:
        report["limitation"] = type(error).__name__ + ": " + str(error)
        return_code = 2
    finally:
        if browser is not None and browser.poll() is None:
            browser.terminate()
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait(timeout=5)
        report["seconds"] = round(time.perf_counter() - started, 3)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"successful": report["successful"], "checks": len(report["checks"]),
                      "seconds": report["seconds"], "limitation": report.get("limitation")}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
