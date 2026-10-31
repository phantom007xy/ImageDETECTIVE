"""Calibrate a head on the dev sets: a temperature and the two verdict thresholds (plan Section 5.6).

Order when goals conflict (fixed in the plan):
  1. "Likely AI-generated" only strictly above the AI threshold: at most 5% of the real photos in
     EVERY real dev pool may be above it (the strictest pool decides).
  2. "Likely authentic" only below the authentic threshold: as low as possible (so few fakes are called
     authentic), but the "Uncertain" band between the two thresholds may hold at most 15% of a fixed
     mix of real and fake dev images (both classes weighted equally).
  3. Missed fakes: whatever results, reported honestly.
The temperature T makes the probability sigmoid(logit / T) honest: it is chosen so that these
probabilities fit the dev mix best (lowest log loss). It does not move the thresholds, which live
on the logit.

The clear level "AI-generated" (without "likely") is used only above the highest score of any real
dev photo. There is no clear "real" level on purpose: a generator the head has never seen can score
very low (30 Oct: 1 in 10 SD 3.5 fakes did), and that can't be measured in advance.

Faces (optional, --real-faces): the same head scores every face crop (the app crop). A face only
changes the verdict above a very strict face threshold: at most 0.5% of the real dev faces in
every pool may be above it. "Shows some AI signs" (shown, never deciding): at most 5% of real faces.
With --fake-faces, a face probability is fitted by Platt scaling on real and AI dev faces. Why so strict: all checks share ONE false-alarm budget, and on fully
generated images a face rule adds almost nothing (src/eval/shared_budget.py); its job is a clearly
AI-made face in an otherwise real-looking photo.

All feature files must be made with the app's preprocessing (src.features.extract --mode app).
A real pool may be private (data/private/...): only saved feature vectors are read, never photos,
and only totals are printed.

Usage (from the repo root):
  python -m src.calibrate.calibrate --head weights/commons_v1/head_E.pt
      --real "phone:data/private/phone_v1_app.npz" --real "polished dev:features/commons_dev_app.npz"
      --fake "Muse test v1:features/muse_image_test_v1_app.npz" --out weights/commons_v1/head_E_calibration.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.eval.fair_compare import load, threshold_for
from src.models.linear_head import LinearHead


def fit_temperature(logits, is_ai, weights):
    """The temperature with the lowest weighted log loss (a simple search over 400 values)."""
    best = None
    for t in np.exp(np.linspace(math.log(0.05), math.log(50), 400)):
        p = 1 / (1 + np.exp(-logits / t))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = -np.sum(weights * np.where(is_ai, np.log(p), np.log(1 - p)))
        if best is None or loss < best[0]:
            best = (loss, float(t))
    return best[1]


def authentic_threshold(logits, weights, ai_logit, max_uncertain):
    """The lowest threshold below ai_logit whose "Uncertain" band holds at most max_uncertain of the mix."""
    best = ai_logit
    for t in np.sort(logits[logits < ai_logit])[::-1]:
        band = weights[(logits > t) & (logits < ai_logit)].sum()
        if band > max_uncertain:
            break
        best = float(t)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--real", action="append", required=True, help='"NAME:feature file[@source]"')
    parser.add_argument("--fake", action="append", required=True, help='"NAME:feature file[@source]"')
    parser.add_argument("--max-false-alarms", type=float, default=0.05)
    parser.add_argument("--max-uncertain", type=float, default=0.15)
    parser.add_argument("--real-faces", action="append", default=[], help='"NAME:face feature file" (real faces)')
    parser.add_argument("--max-face-false-alarms", type=float, default=0.005)
    parser.add_argument("--fake-faces", action="append", default=[], help='"NAME:face feature file" (AI faces)')
    parser.add_argument("--out", type=Path, help="JSON file for the app")
    args = parser.parse_args()

    head = LinearHead.load(args.head)
    split = lambda item: item.split(":", 1)  # noqa: E731
    reals = {name: head.logits(load(spec, 0)).numpy() for name, spec in map(split, args.real)}
    fakes = {name: head.logits(load(spec, 1)).numpy() for name, spec in map(split, args.fake)}

    ai_logit = max(threshold_for(s, args.max_false_alarms) for s in reals.values())
    real_all, fake_all = np.concatenate(list(reals.values())), np.concatenate(list(fakes.values()))
    logits = np.concatenate([real_all, fake_all])
    is_ai = np.concatenate([np.zeros(len(real_all), bool), np.ones(len(fake_all), bool)])
    weights = np.where(is_ai, 0.5 / len(fake_all), 0.5 / len(real_all))  # both classes count equally
    real_logit = authentic_threshold(logits, weights, ai_logit, args.max_uncertain)
    temperature = fit_temperature(logits, is_ai, weights)
    prob = lambda z: 1 / (1 + math.exp(-z / temperature))  # noqa: E731
    # The clear level "AI-generated": above every real dev photo (none of them reaches it).
    strong_ai_logit = float(max(s.max() for s in reals.values()))

    print(f"head {args.head.name}: temperature {temperature:.2f}")
    print(f"  AI-generated (clear) if logit >  {strong_ai_logit:+.2f}  (calibrated AI probability >  "
          f"{prob(strong_ai_logit):.3f}; no real dev photo is above it)")
    print(f"  Likely AI-generated  if logit >  {ai_logit:+.2f}  (calibrated AI probability >  {prob(ai_logit):.3f})")
    print(f"  Likely authentic     if logit <= {real_logit:+.2f}  (calibrated AI probability <= {prob(real_logit):.3f})")
    print(f"  Uncertain in between: {weights[(logits > real_logit) & (logits < ai_logit)].sum():.1%} of the mix "
          f"(cap {args.max_uncertain:.0%})\n")
    print(f"  {'set':<18}{'n':>6} | {'AI (clear)':>10} {'Likely AI':>10} {'Uncertain':>10} {'Authentic':>10}")
    for title, groups in (("REAL", reals), ("AI", fakes)):
        for name, s in groups.items():
            ai, auth = (s > ai_logit).mean(), (s <= real_logit).mean()  # AI = strictly above, as in fair_compare
            clear = (s > strong_ai_logit).mean()
            print(f"  {title + ' ' + name:<18}{len(s):>6} | {clear:>10.1%} {ai - clear:>10.1%} {1 - ai - auth:>10.1%} "
                  f"{auth:>10.1%}")

    face_ai_logit = face_leans_logit = face_platt = None
    if args.real_faces:
        real_faces = {name: head.logits(load(spec, 0)).numpy() for name, spec in map(split, args.real_faces)}
        face_ai_logit = max(threshold_for(s, args.max_face_false_alarms) for s in real_faces.values())
        face_leans_logit = max(threshold_for(s, 0.05) for s in real_faces.values())
        print(f"\n  A face looks AI-made if its logit > {face_ai_logit:+.2f} "
              f"(at most {args.max_face_false_alarms:.1%} of the real dev faces in every pool; changes the verdict)")
        print(f"  A face shows some AI signs if its logit > {face_leans_logit:+.2f} (at most 5% of the real dev faces)")
        for name, s in real_faces.items():
            print(f"  REAL faces {name:<12}{len(s):>6} | looks AI-made: {(s > face_ai_logit).mean():.1%} | "
                  f"some AI signs: {(s > face_leans_logit).mean():.1%}")
        if args.fake_faces:
            # Platt scaling: a 1-number logistic regression from the face logit to an honest face probability
            # (the whole-image temperature doesn't fit face crops, whose scores sit lower).
            fake_faces = {name: head.logits(load(spec, 1)).numpy() for name, spec in map(split, args.fake_faces)}
            x = np.concatenate(list(real_faces.values()) + list(fake_faces.values()))
            y = np.concatenate([np.zeros(sum(map(len, real_faces.values()))), np.ones(sum(map(len, fake_faces.values())))])
            platt = LinearHead().fit(torch.from_numpy(x[:, None]).float(), torch.from_numpy(y).float())
            face_platt = [float(platt.weight[0]), float(platt.bias)]
            for name, s in fake_faces.items():
                print(f"  AI faces {name:<14}{len(s):>6} | looks AI-made: {(s > face_ai_logit).mean():.1%} | "
                      f"some AI signs: {(s > face_leans_logit).mean():.1%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "head": str(args.head), "temperature": temperature, "ai_logit": ai_logit, "real_logit": real_logit,
            "strong_ai_logit": strong_ai_logit, "face_ai_logit": face_ai_logit, "face_leans_logit": face_leans_logit,
            "face_platt": face_platt,
            "max_false_alarms": args.max_false_alarms, "max_uncertain": args.max_uncertain,
            "real_pools": {n: len(s) for n, s in reals.items()}, "fake_sets": {n: len(s) for n, s in fakes.items()},
        }, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
