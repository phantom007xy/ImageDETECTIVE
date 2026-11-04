"""One public calibration-boundary photo: CPU fp32, then CPU bf16 DINO only.

No SAM, PE, private inputs, network, APIs, GPU, training, or recalibration.
The fp32 test precedes bf16 conversion so fp32 weights are never rounded first.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.data import preprocess
from src.models.backbone import Backbone, cpu_has_amx_bf16
from src.models.linear_head import LinearHead
from src.inference import Detector

OUT = Path("results/release_2026-11-04/cross_device_boundary.json")
HEAD = Path("weights/google_v1/head_G3tt.pt")
EDIT_CAL = Path("weights/edit_check_v3/head_EC3pf_calibration.json")
CACHE = Path("features/sid_reals_0_9_unseen_app.npz")


def level(score, whole_line, real_line, strong_line):
    detector = object.__new__(Detector)
    detector.calibration = {"ai_logit": whole_line, "real_logit": real_line, "strong_ai_logit": strong_line}
    return detector.verdict(score, [], None)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    torch.set_num_threads(4)
    calibration = json.loads(HEAD.with_name(HEAD.stem + "_calibration.json").read_text())
    edit_calibration = json.loads(EDIT_CAL.read_text())
    if abs(edit_calibration["whole_ai_logit_calibrated"] - calibration["ai_logit"]) > 1e-6:
        raise RuntimeError("The current joint calibration does not match the whole-image calibration")
    whole_line = edit_calibration["whole_ai_logit"]
    real_line, strong_line = calibration["real_logit"], calibration["strong_ai_logit"]
    head = LinearHead.load(HEAD)
    with np.load(CACHE, allow_pickle=False) as data:
        keep = np.flatnonzero(data["labels"] == 0)
        scores = head.logits(torch.from_numpy(data["features"][keep].astype(np.float32))).numpy()
        local_index = int(np.argmax(scores))
        row_index = int(keep[local_index])
        cached_score = float(scores[local_index])
        public_path = Path(str(data["img_ids"][row_index]))
    normalized = str(public_path).replace("\\", "/").lower()
    if "/sid_set/" not in normalized or any(token in normalized for token in ("/private", "/selfedits/")):
        raise RuntimeError("Selected input is not the expected public SID-Set source")
    if abs(cached_score - strong_line) > 1e-6:
        raise RuntimeError("Selected public cached maximum is no longer the strong-AI boundary row")
    image_digest = hashlib.sha256(public_path.read_bytes()).hexdigest()
    with Image.open(public_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    view = preprocess.inference_view(image)
    pixels = preprocess.to_tensor(view).unsqueeze(0)
    gpu_record = None
    live_path = OUT.parent / "live_predictions.json"
    if live_path.exists():
        live = json.loads(live_path.read_text()).get("records", {})
        record = live.get(image_digest)
        if record:
            gpu_record = {"score": record["score"], "pixel_verdict": record["pixel_verdict"],
                          "margin_to_strong": record["score"] - strong_line,
                          "source": "existing full live detector record, laptop GPU bf16"}
    report = {"status": "running/incomplete", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "ONE public OpenImages calibration maximum; DINO whole-image branch only; CPU fp32 followed by CPU bf16",
              "public_image_sha256": image_digest, "public_file": public_path.name,
              "cached_whole_score": cached_score, "strong_ai_logit": strong_line,
              "joint_whole_ai_logit": whole_line, "real_logit": real_line,
              "existing_gpu_record": gpu_record, "cpu_threads": torch.get_num_threads(),
              "cpu_has_amx_bf16_probe": cpu_has_amx_bf16(), "measurements": [],
              "limitations": [
                  "The CPU bf16 run is on this laptop, not the hosted AMX CPU. It does not establish hosted-server arithmetic behavior.",
                  "Only one known boundary image is tested. These results do not estimate any dataset-wide error rate.",
                  "Verdict levels here use whole-image thresholds only; SAM, PE, faces and metadata are not run.",
                  "No thresholds, weights, source data, or production code are modified."]}

    def save():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(OUT)

    save()
    backbone = None
    try:
        print("Loading CPU DINO only, fp32 first, four threads", flush=True)
        backbone = Backbone(device="cpu")
        if backbone.dtype != torch.float32 or next(backbone.model.parameters()).dtype != torch.float32:
            raise RuntimeError("The first CPU pass must use original unrounded fp32 weights")
        for dtype in (torch.float32, torch.bfloat16):
            if dtype == torch.bfloat16:
                backbone.model.to(dtype=dtype)
                backbone.dtype = dtype
                gc.collect()
            started = time.perf_counter()
            with torch.no_grad():
                tokens = backbone.patch_tokens(pixels)[0].float().cpu()
                patch_logits = tokens @ head.weight + head.bias
                score = float(patch_logits.mean())
            measurement = {"device": "cpu", "dtype": str(dtype), "score": score,
                           "whole_image_verdict_level": level(score, whole_line, real_line, strong_line),
                           "above_strong_ai_line": score > strong_line,
                           "margin_to_strong": score - strong_line,
                           "delta_from_cached_score": score - cached_score,
                           "delta_from_gpu_score": score - gpu_record["score"] if gpu_record else None,
                           "seconds": time.perf_counter() - started}
            report["measurements"].append(measurement)
            save()
            print(json.dumps(measurement), flush=True)
            del tokens, patch_logits
        report["status"] = "complete"
        report["any_tested_cpu_dtype_above_strong_line"] = any(item["above_strong_ai_line"] for item in report["measurements"])
        report["any_confident_pixel_false_alarm"] = any(item["whole_image_verdict_level"] == "AI-generated"
                                                       for item in report["measurements"])
    finally:
        del backbone, head, pixels, image, view
        gc.collect()
        report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
