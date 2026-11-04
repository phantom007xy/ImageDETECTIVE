"""CPU-only behavioral release checks for the real FastAPI app.

Run from the repository root in a fresh interpreter::

    python -m src.eval.app_release_check

The detector is replaced *before* app.server is imported. Rendering, image decoding,
metadata parsing, routes, retention and explanation facts remain production code.
Paid services are mocked; network access, model downloads and CUDA initialization
are denied. All images are synthetic. Known defects fail normally, so the exit
status and JSON report remain useful release gates after fixes are made.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch


DEFAULT_OUTPUT = Path("results/release_2026-11-04/app_checks.json")


def fragmented_jpeg_declaration(field="digitalSourceType", drop_last=False, https=False, first_sequence=1, padding=70000):
    """Synthetic JPEG-XT manifest spanning APP11 packets, with repeated box headers."""
    import struct

    def box(kind, payload):
        return struct.pack(">I4s", len(payload) + 8, kind) + payload

    def text(value):
        raw = value.encode()
        return (bytes([0x60 + len(raw)]) if len(raw) < 24 else bytes([0x78, len(raw)])) + raw

    uri = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    if https:
        uri = uri.replace("http:", "https:", 1)
    description = bytes.fromhex("6332706100110010800000aa00389b71") + b"\x03c2pa\0"
    manifest = box(b"jumb", box(b"jumd", description) + box(b"free", b"\0" * padding)
                   + box(b"cbor", b"\xa1" + text(field) + text(uri)))
    parts = [manifest[pos:pos + 60000] for pos in range(0, len(manifest), 60000)]
    packets = []
    for index, part in enumerate(parts):
        payload = b"JP\0\x01" + struct.pack(">I", index + first_sequence) + (manifest[:8] if index else b"") + part
        packets.append(b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload)
    jpeg = image_bytes("JPEG")
    return jpeg[:2] + b"".join(packets[:-1] if drop_last else packets) + jpeg[2:]


def image_bytes(fmt="PNG", *, size=(128, 96), comment=None, orientation=None):
    from PIL import Image, PngImagePlugin

    image = Image.new("RGB", size, (38, 86, 114))
    options = {}
    if comment is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", comment)
        options["pnginfo"] = info
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


class FakeDetector:
    """Deterministic pixels only; actual app rendering and metadata stay active."""

    calibrated = True

    def __init__(self, *args, **kwargs):
        self.reset()

    def reset(self):
        self.mode = "authentic"
        self.faces = []
        self.calls = []
        self.called = threading.Event()

    def probability(self, value):
        return 1 / (1 + math.exp(-float(value)))

    def readiness(self):
        return {key: True for key in ("whole_image", "edit_check", "faces", "face_swap", "second_render_head")}

    def analyze(self, image, file_bytes=None):
        import numpy as np
        from src.services import metadata

        self.calls.append({"bytes": file_bytes, "size": image.size})
        self.called.set()
        meta = metadata.check_file(file_bytes) if file_bytes is not None else None
        verdict = {"authentic": "Likely authentic", "uncertain": "Uncertain",
                   "edited": "Likely AI-edited", "secondary": "Likely AI-generated",
                   "generated": "Likely AI-generated"}[self.mode]
        probability = 0.9 if self.mode == "generated" else 0.1
        result = {
            "score": -2.0, "probability_ai": probability,
            "verdict": verdict, "pixel_verdict": verdict, "verdict_reason": "pixels",
            "whole_ai": self.mode == "generated", "metadata": meta,
            "edit": {"flagged": np.bool_(self.mode == "edited"),
                     "flagged2": np.bool_(self.mode == "secondary")},
            "patch_logits": np.array([[-2.0, 2.0], [-1.0, 1.0]], dtype=np.float32),
            "faces": self.faces, "seconds": 0.125,
        }
        if meta and meta["ai_label"] == "generated":
            result.update(verdict="AI-generated", verdict_reason="metadata")
        return result


@contextlib.contextmanager
def offline_app(counters):
    if "app.server" in sys.modules:
        raise RuntimeError("Run this release check in a fresh Python interpreter.")

    def deny(kind):
        def blocked(*args, **kwargs):
            counters[kind] += 1
            raise AssertionError(f"Release check attempted forbidden {kind} operation")
        return blocked

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            "HEADS_REPO": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "", "MODEL_API_KEY": "offline-test-key",
        }))
        # Windows implements socketpair through a private loopback connection;
        # asyncio needs that pair even though TestClient itself is in-process.
        original_connect, original_socketpair = socket.socket.connect, socket.socketpair
        local = threading.local()

        def socketpair(*args, **kwargs):
            previous = getattr(local, "socketpair", False)
            local.socketpair = True
            try:
                return original_socketpair(*args, **kwargs)
            finally:
                local.socketpair = previous

        def connect(connection, address):
            if getattr(local, "socketpair", False):
                return original_connect(connection, address)
            return deny("network")(connection, address)

        stack.enter_context(patch.object(socket, "socketpair", socketpair))
        stack.enter_context(patch.object(socket.socket, "connect", connect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny("network")))
        stack.enter_context(patch.object(socket, "create_connection", deny("network")))
        import torch
        import huggingface_hub

        stack.enter_context(patch.object(torch.cuda, "_lazy_init", deny("cuda")))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        stack.enter_context(patch.object(huggingface_hub, "snapshot_download", deny("download")))
        stack.enter_context(patch.object(huggingface_hub, "hf_hub_download", deny("download")))
        from src import inference

        stack.enter_context(patch.object(inference, "Detector", FakeDetector))
        server = importlib.import_module("app.server")
        from src.services import meta_api

        stack.enter_context(patch.object(meta_api, "api_key", return_value="offline-test-key"))
        stack.enter_context(patch.object(meta_api, "post", deny("network")))
        stack.enter_context(patch.object(meta_api, "_record", deny("spending_file_write")))
        yield server


class AppReleaseChecks(unittest.TestCase):
    server = None

    def setUp(self):
        from fastapi.testclient import TestClient

        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.measurements = {}
        self.server.detector.reset()
        with self.server.memory_lock:
            self.server.memory.clear()
        self.investigator = self.stack.enter_context(
            patch.object(self.server.investigator, "investigate", return_value=None))
        self.outline = self.stack.enter_context(
            patch.object(self.server.sam_outline, "outline", return_value=None))
        self.explanation = self.stack.enter_context(
            patch.object(self.server.explain, "explain", return_value=None))
        self.generate = self.stack.enter_context(
            patch.object(self.server.muse_image, "generate", return_value=None))
        self.client = self.stack.enter_context(TestClient(self.server.app))

    def upload(self, data=None, *, name="synthetic.png", content_type="image/png"):
        if data is None:
            data = image_bytes()
        return self.client.post("/api/investigate", files={"file": (name, data, content_type)})

    def checked(self, mode="authentic"):
        self.server.detector.mode = mode
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_homepage_and_static_javascript(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Start investigation", response.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_supported_decodable_uploads(self):
        for fmt, suffix, mime in (("PNG", "png", "image/png"),
                                  ("JPEG", "jpg", "image/jpeg"),
                                  ("WEBP", "webp", "image/webp")):
            with self.subTest(format=fmt):
                response = self.upload(image_bytes(fmt), name="synthetic." + suffix, content_type=mime)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["verdict"], "Likely authentic")

    def test_health_checks_local_model_readiness(self):
        self.assertEqual(self.client.get("/healthz").json()["status"], "ready")
        with patch.object(self.server.detector, "readiness", return_value={"edit_check": False}):
            self.assertEqual(self.client.get("/healthz").status_code, 503)

    def test_sample_routes_are_whitelisted_and_keep_original_bytes(self):
        from pathlib import Path
        for name, filename in self.server.EXAMPLES.items():
            response = self.client.get("/examples/" + name)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, (Path("app/examples") / filename).read_bytes())
        self.assertEqual(self.client.get("/examples/unknown").status_code, 404)

    def test_retained_images_are_small_after_original_inference(self):
        report = self.upload(image_bytes(size=(3000, 2000))).json()
        self.assertEqual(self.server.detector.calls[-1]["size"], (3000, 2000))
        saved = self.server.recall(report["id"])
        self.assertLessEqual(max(saved["image"].size), self.server.DISPLAY_LONG_SIDE)
        self.assertEqual(saved["image"].size, saved["base"].size)

    def test_session_cache_evicts_by_count_and_bytes(self):
        with patch.object(self.server, "MAX_SESSIONS", 2):
            first = self.server.remember({"marker": "first"})
            self.server.remember({"marker": "second"})
            last = self.server.remember({"marker": "third"})
            self.assertIsNone(self.server.recall(first))
            self.assertIsNotNone(self.server.recall(last))
        with self.server.memory_lock:
            self.server.memory.clear()
        with patch.object(self.server, "MAX_SESSION_BYTES", 2000):
            first = self.server.remember({"padding": b"x" * 800})
            last = self.server.remember({"padding": b"x" * 800})
            self.assertIsNone(self.server.recall(first))
            self.assertIsNotNone(self.server.recall(last))

    def test_chunked_oversize_upload_is_bounded_before_detector(self):
        body = (b'--testbound\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n'
                b'Content-Type: image/png\r\n\r\n' + b'x' * 1500 + b'\r\n--testbound--\r\n')
        with patch.object(self.server, "MAX_BYTES", 1024):
            response = self.client.post("/api/investigate", content=(body[i:i+300] for i in range(0, len(body), 300)),
                headers={"content-type": "multipart/form-data; boundary=testbound"})
            self.assertEqual(response.status_code, 413)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_malformed_multipart_is_a_client_error(self):
        response = self.client.post("/api/investigate", content=b'not-a-boundary\r\n',
            headers={"content-type": "multipart/form-data; boundary=testbound"})
        self.assertEqual(response.status_code, 400)

    def test_structured_xmp_is_an_unverified_declaration(self):
        from PIL import Image, PngImagePlugin
        from src.services import metadata
        packet = ('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                  'xmlns:i="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">'
                  '<rdf:Description i:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
                  '</rdf:RDF>')
        info = PngImagePlugin.PngInfo(); info.add_itxt("XML:com.adobe.xmp", packet)
        buffer = io.BytesIO(); Image.new("RGB", (64, 64)).save(buffer, format="PNG", pnginfo=info)
        result = metadata.check_file(buffer.getvalue())
        self.assertEqual(result["ai_label"], "generated")
        self.assertFalse(result["verified"])

    def test_empty_upload_rejected(self):
        self.assertEqual(self.upload(b"").status_code, 400)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_corrupt_upload_rejected(self):
        self.assertEqual(self.upload(b"not an image").status_code, 400)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_unsupported_decodable_format_rejected(self):
        response = self.upload(image_bytes("GIF"), name="synthetic.gif", content_type="image/gif")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_missing_upload_rejected(self):
        self.assertEqual(self.client.post("/api/investigate").status_code, 422)

    def test_decoded_format_used_instead_of_filename(self):
        self.assertEqual(self.upload(image_bytes("PNG"), name="synthetic.jpg",
                                     content_type="image/jpeg").status_code, 200)

    def test_file_size_enforced_before_detector(self):
        with patch.object(self.server, "MAX_BYTES", 1024):
            self.assertEqual(self.upload(b"x" * 1025).status_code, 413)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_oversized_content_length_rejected(self):
        response = self.client.post("/api/investigate", content=b"",
            headers={"content-length": str(self.server.MAX_BYTES + 1024 * 1024 + 1)})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_original_bytes_reach_metadata_check_unchanged(self):
        data = image_bytes(comment="synthetic original-byte retention marker")
        response = self.upload(data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.server.detector.calls[-1]["bytes"], data)
        self.assertTrue(response.json()["metadata"]["read"])

    def test_exif_orientation_applied_without_reencoding_original(self):
        data = image_bytes("JPEG", size=(12, 20), orientation=6)
        self.assertEqual(self.upload(data, name="synthetic.jpg").status_code, 200)
        self.assertEqual(self.server.detector.calls[-1]["size"], (20, 12))
        self.assertEqual(self.server.detector.calls[-1]["bytes"], data)

    def test_report_serialization_and_evidence_decode(self):
        import numpy as np
        from PIL import Image

        self.server.detector.faces = [
            {"box": [10, 30, 65, 85], "too_small": np.bool_(False), "flagged": True,
             "level": "ai", "kind": "swapped", "probability_ai": 0.98},
            {"box": [80, 35, 100, 55], "too_small": np.bool_(True)},
        ]
        report = self.checked()
        self.assertEqual(report["faces_checked"], 1)
        self.assertEqual(report["faces_flagged"], 1)
        self.assertEqual([row["n"] for row in report["faces"]], [1, 2])
        self.assertIs(report["edit"]["flagged"], False)
        self.assertIs(report["faces"][1]["too_small"], True)
        self.assertEqual(report["probability_ai"], 0.1)
        self.assertTrue(report["calibrated"])
        self.assertEqual(report["tone"], "green")
        evidence = base64.b64decode(report["evidence"].split(",", 1)[1])
        with Image.open(io.BytesIO(evidence)) as decoded:
            self.assertEqual(decoded.size, (128, 96))
            self.assertEqual(decoded.format, "JPEG")

    def test_unknown_followup_session_rejected(self):
        response = self.client.post("/api/investigate/unknown-session/explain")
        self.assertEqual(response.status_code, 410)
        self.investigator.assert_not_called()
        self.explanation.assert_not_called()

    def test_expired_session_rejected_and_cleanup_preserves_fresh(self):
        report = self.checked()
        expired = report["id"]
        fresh = self.server.remember({"marker": "synthetic fresh session"})
        with self.server.memory_lock:
            _, item = self.server.memory[expired]
            self.server.memory[expired] = (time.time() - self.server.KEEP_SECONDS - 1, item)
        self.assertIsNone(self.server.recall(expired))
        self.assertEqual(self.client.post(f"/api/investigate/{expired}/explain").status_code, 410)
        self.server.forget_old()
        self.assertNotIn(expired, self.server.memory)
        self.assertIsNotNone(self.server.recall(fresh))

    def test_shutdown_clears_retained_sessions(self):
        from fastapi.testclient import TestClient

        with TestClient(self.server.app):
            self.server.remember({"marker": "synthetic shutdown session"})
            self.assertTrue(self.server.memory)
        self.assertEqual(self.server.memory, {})

    def test_authentic_photo_skips_picture_apis(self):
        self.explanation.return_value = "No clear AI signs; the context is not verified."
        report = self.checked()
        response = self.client.post(f"/api/investigate/{report['id']}/explain")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["investigator"], "not run")
        self.assertEqual(response.json()["explanation"], self.explanation.return_value)
        self.investigator.assert_not_called()
        self.outline.assert_not_called()

    def test_optional_services_unavailable_preserve_verdict(self):
        report = self.checked("uncertain")
        response = self.client.post(f"/api/investigate/{report['id']}/explain")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["investigator"], "unavailable")
        self.assertIsNone(response.json()["explanation"])
        self.assertEqual(self.server.recall(report["id"])["result"]["verdict"], "Uncertain")
        self.investigator.assert_called_once()

    def test_named_investigation_renders_real_outline(self):
        import numpy as np

        mask = np.zeros((96, 128), dtype=bool)
        mask[25:70, 35:90] = True
        self.investigator.return_value = {"guess": "bicycle", "mask": mask, "confidence": 80}
        report = self.checked("edited")
        response = self.client.post(f"/api/investigate/{report['id']}/explain")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outline"], "named object")
        self.assertEqual(response.json()["named"], "bicycle")
        self.assertTrue(response.json()["evidence"].startswith("data:image/jpeg;base64,"))

    def test_ai_photo_uses_strongest_object_outline(self):
        import numpy as np

        mask = np.zeros((96, 128), dtype=bool)
        mask[:, 64:] = True
        self.outline.return_value = {"mask": mask}
        report = self.checked("generated")
        response = self.client.post(f"/api/investigate/{report['id']}/explain")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outline"], "strongest object")
        self.investigator.assert_not_called()
        self.assertIsNotNone(self.explanation.call_args.args[2])

    def test_generation_validation_and_unavailable_fallback(self):
        self.assertEqual(self.client.post("/api/fool", json={"prompt": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/api/fool", json={"prompt": "a famous actor"}).status_code, 400)
        self.generate.assert_not_called()
        response = self.client.post("/api/fool", json={"prompt": "a fictional bicycle"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(self.server.detector.calls), 0)

    def test_generated_image_crop_flow_and_expiry(self):
        self.generate.return_value = image_bytes()
        response = self.client.post("/api/fool", json={"prompt": "a fictional bicycle"})
        self.assertEqual(response.status_code, 200)
        key = response.json()["id"]
        cropped = self.client.post(f"/api/fool/{key}/crop")
        self.assertEqual(cropped.status_code, 200)
        self.assertIsNone(cropped.json()["report"]["metadata"]["ai_label"])
        self.assertEqual(self.server.detector.calls[-1]["size"], (90, 67))
        self.assertEqual(self.client.post("/api/fool/unknown-session/crop").status_code, 410)

    def test_secondary_head_facts_describe_positive_detection(self):
        report = self.checked("secondary")
        result = self.server.recall(report["id"])["result"]
        facts = self.server.explain.facts(result)
        self.measurements["facts"] = facts
        self.assertNotEqual(facts.get("ai_edit_check"), "no clear signs",
            "Secondary PE head triggered the verdict but explanation facts say no clear signs")

    def test_optional_api_does_not_block_another_detector_verdict(self):
        report = self.checked("uncertain")
        entered, release, verdict_done = threading.Event(), threading.Event(), threading.Event()
        statuses, errors = {}, []

        def blocked_investigation(image):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Synthetic API release signal timed out")
            return None

        self.investigator.side_effect = blocked_investigation

        def followup():
            try:
                statuses["followup"] = self.client.post(f"/api/investigate/{report['id']}/explain").status_code
            except Exception as error:
                errors.append(type(error).__name__)

        def next_verdict():
            try:
                statuses["verdict"] = self.upload().status_code
            except Exception as error:
                errors.append(type(error).__name__)
            finally:
                verdict_done.set()

        first = threading.Thread(target=followup, daemon=True)
        second = threading.Thread(target=next_verdict, daemon=True)
        first.start()
        try:
            self.assertTrue(entered.wait(3), "Synthetic API was not entered")
            second.start()
            independent = verdict_done.wait(0.75)
            self.measurements["verdict_completed_while_api_blocked"] = independent
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive(), "Release-check workers failed to exit")
        self.assertEqual(errors, [])
        self.assertEqual(statuses, {"followup": 200, "verdict": 200})
        self.assertTrue(independent, "A blocked optional API call held up the next basic detector verdict")

    def test_memory_only_upload_does_not_spool_to_disk(self):
        import starlette.formparsers

        original = starlette.formparsers.SpooledTemporaryFile
        temporary_files = []

        def capture(*args, **kwargs):
            file = original(*args, **kwargs)
            temporary_files.append(file)
            return file

        # Synthetic padding makes a valid image exceed the ordinary 1 MiB spool threshold.
        payload = image_bytes() + b"\0" * (1024 * 1024 + 64)
        with patch.object(starlette.formparsers, "SpooledTemporaryFile", side_effect=capture):
            response = self.upload(payload)
        self.assertEqual(response.status_code, 200)
        rolled = sum(bool(getattr(file, "_rolled", False)) for file in temporary_files)
        self.measurements["temporary_files_rolled_to_disk"] = rolled
        self.assertEqual(rolled, 0, "Synthetic upload reached temporary disk storage despite memory-only promise")

    def test_comment_mention_is_not_an_actual_ai_source_declaration(self):
        from src.services import metadata

        data = image_bytes(comment="A discussion of trainedAlgorithmicMedia, not an AI source declaration")
        found = metadata.check_file(data)
        self.measurements["metadata_result"] = found
        self.assertIsNone(found["ai_label"], "Ordinary PNG Comment text was mistaken for an AI source field")

    def test_fragmented_jpeg_source_declaration_is_read(self):
        from src.services import metadata
        found = metadata.check_file(fragmented_jpeg_declaration())
        self.assertEqual(found["ai_label"], "generated")
        self.assertTrue(found["c2pa"])
        self.assertFalse(found["verified"])

    def test_fragmented_jpeg_comment_is_not_a_source_declaration(self):
        from src.services import metadata
        self.assertIsNone(metadata.check_file(fragmented_jpeg_declaration(field="comment"))["ai_label"])

    def test_https_iptc_source_declaration_is_read(self):
        from src.services import metadata
        self.assertEqual(metadata.check_file(fragmented_jpeg_declaration(https=True))["ai_label"], "generated")

    def test_zero_indexed_jpeg_source_declaration_is_read(self):
        from src.services import metadata
        for padding in (0, 70000):
            with self.subTest(padding=padding):
                found = metadata.check_file(fragmented_jpeg_declaration(first_sequence=0, padding=padding))
                self.assertEqual(found["ai_label"], "generated")

    def test_incomplete_jpeg_manifest_does_not_declare_ai(self):
        from src.services import metadata
        self.assertIsNone(metadata.check_file(fragmented_jpeg_declaration(drop_last=True))["ai_label"])

    def test_pe_evidence_marks_only_the_analyzed_region(self):
        import numpy as np
        from PIL import Image
        result = self.server.detector.analyze(Image.new("RGB", (128, 96)))
        result.update(evidence_source="edit", evidence_bounds=(.25, 0, .75, 1),
                      evidence_logits=np.full((24, 24), 10, dtype=np.float32))
        image = Image.new("RGB", (128, 96), (40, 80, 120))
        body, base = self.server.report(image, result)
        self.assertEqual(base.getpixel((5, 48)), image.getpixel((5, 48)))
        self.assertNotEqual(base.getpixel((64, 48)), image.getpixel((64, 48)))
        self.assertIn("Perception Encoder", body["evidence_source"])
        self.assertIn("outer area is not scored", body["evidence_note"])

    def test_failed_paid_stage_keeps_its_budget_reservation(self):
        from src.services import meta_api
        charged = []
        with contextlib.ExitStack() as local:
            local.enter_context(patch.object(meta_api, "_cache", {}))
            local.enter_context(patch.object(meta_api, "spent_today", side_effect=lambda: sum(charged)))
            local.enter_context(patch.object(meta_api, "daily_cap", return_value=.02))
            local.enter_context(patch.object(meta_api, "_record", side_effect=charged.append))
            failed = unittest.mock.Mock(side_effect=RuntimeError("synthetic second-stage failure"))
            self.assertIsNone(meta_api.paid_call("failed-stage", .02, failed))
            self.assertAlmostEqual(sum(charged), .02)
            another = unittest.mock.Mock(return_value=("unused", .01))
            self.assertIsNone(meta_api.paid_call("another", .01, another))
            another.assert_not_called()

    def test_busy_api_returns_without_waiting_for_a_worker(self):
        from src.services import meta_api
        held = threading.Lock(); held.acquire()
        finished = threading.Event()
        results = []
        def worker():
            results.append(meta_api.paid_call("busy-test", .01, unittest.mock.Mock()))
            finished.set()
        with patch.object(meta_api, "_lock", held), patch.object(meta_api, "_cache", {}):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                independent = finished.wait(.25)
            finally:
                held.release(); thread.join(3)
        self.assertTrue(independent)
        self.assertEqual(results, [None])

    def test_success_refunds_unused_reservation_and_reuses_cache(self):
        from src.services import meta_api
        charged = []
        with contextlib.ExitStack() as local:
            local.enter_context(patch.object(meta_api, "_cache", {}))
            local.enter_context(patch.object(meta_api, "spent_today", side_effect=lambda: sum(charged)))
            local.enter_context(patch.object(meta_api, "daily_cap", return_value=.02))
            local.enter_context(patch.object(meta_api, "_record", side_effect=charged.append))
            callback = unittest.mock.Mock(return_value=("synthetic cached", .004))
            self.assertEqual(meta_api.paid_call("refund", .02, callback), "synthetic cached")
            self.assertEqual(meta_api.paid_call("refund", .02, callback), "synthetic cached")
            self.assertAlmostEqual(sum(charged), .004)
            callback.assert_called_once()

    def test_budget_reserves_inflight_cost_and_deduplicates_same_request(self):
        from src.services import meta_api

        entered, release = threading.Event(), threading.Event()
        charged, results, errors = [], [], []

        def fake_call(key):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Synthetic budget-call release signal timed out")
            return "synthetic cached result", 0.01

        def worker():
            try:
                results.append(meta_api.paid_call("synthetic-same-request", 0.01, fake_call))
            except Exception as error:
                errors.append(type(error).__name__)

        with contextlib.ExitStack() as local:
            local.enter_context(patch.object(meta_api, "_cache", {}))
            local.enter_context(patch.object(meta_api, "_lock", threading.Lock()))
            local.enter_context(patch.object(meta_api, "spent_today", side_effect=lambda: sum(charged)))
            local.enter_context(patch.object(meta_api, "daily_cap", return_value=0.01))
            local.enter_context(patch.object(meta_api, "_record", side_effect=charged.append))
            first = threading.Thread(target=worker, daemon=True)
            second = threading.Thread(target=worker, daemon=True)
            first.start()
            try:
                self.assertTrue(entered.wait(3), "Synthetic paid callback was not entered")
                second.start()
                # First call remains in flight; a fixed implementation may reject or wait.
                second.join(0.25)
            finally:
                release.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
            self.assertFalse(first.is_alive() or second.is_alive(), "Budget-check workers failed to exit")
        self.measurements.update(simulated_charges=len(charged), simulated_spend=sum(charged), cap=0.01)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertLessEqual(sum(charged), 0.01,
            "Concurrent identical requests charged more than the remaining simulated daily cap")


class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checks = []
        self.started = {}

    def startTest(self, test):
        self.started[test.id()] = time.perf_counter()
        super().startTest(test)

    def record(self, test, status, error=None):
        item = {"id": test.id().split(".")[-1], "status": status,
                "seconds": round(time.perf_counter() - self.started.get(test.id(), time.perf_counter()), 4)}
        if error:
            item["error_type"] = error[0].__name__
            item["message"] = str(error[1])
        if getattr(test, "measurements", None):
            item["measurements"] = test.measurements
        self.checks.append(item)

    def addSuccess(self, test):
        self.record(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test, error):
        self.record(test, "failed", error)
        super().addFailure(test, error)

    def addError(self, test, error):
        self.record(test, "error", error)
        super().addError(test, error)

    def addSubTest(self, test, subtest, error):
        if error:
            self.record(test, "failed", error)
        super().addSubTest(test, subtest, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counters = {"network": 0, "download": 0, "cuda": 0, "spending_file_write": 0}
    started = time.perf_counter()
    report = {"scope": "CPU-only FastAPI behavior with synthetic images and mocked detector/paid services",
              "production_code_changed": False, "external_operations_attempted": counters, "checks": []}
    try:
        with offline_app(counters) as server:
            AppReleaseChecks.server = server
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(AppReleaseChecks)
            result = unittest.TextTestRunner(verbosity=2, resultclass=JsonResult).run(suite)
        report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                      successful=result.wasSuccessful() and not any(counters.values()), checks=result.checks)
    except Exception as error:
        report.update(tests_run=0, failures=0, errors=1, successful=False,
                      harness_error={"type": type(error).__name__, "message": str(error)})
    report["seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("tests_run", "failures", "errors", "successful", "seconds")}))
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
