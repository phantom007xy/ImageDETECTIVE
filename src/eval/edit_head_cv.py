"""Head-only retraining of the edit check on the frozen fine-tuned PE (EC3), with Muse edits of a person added; tested
fairly on my own selfies. Totals only.

Why: on my selfies the frozen EC3 features separate Muse edits from their originals almost perfectly (AUC 0.99), but
the head scores every photo with a person far lower, so the edits stay under the line (results/self_edits_v1.txt).
A new linear head on the same frozen features, trained with person edits too, is the cheapest fix: the PE stays as it
is, so its private calibration features stay valid.

Fair test: the 30 selfies are split into 3 folds by ORIGINAL photo; each fold's selfies (originals and every edit of
them) are held out once while the head trains on everything else. The line is then set exactly as in the app
(src/calibrate/calibrate_edit_check.py: jointly with head F on the public real pools, the joint trade, the edit check
alone at most 0.5% of every pool), and the held-out selfies are scored at it.

Inputs (src/features/extract_pe.py with --pe-weights weights/edit_check_v3/pe_EC3.pt):
  --train   public training features (train views), globs
  --person  my selfie pairs' training views (data/private/...), globs; their groups come from the edit manifests
  --person-eval  the same pairs in the app's view (data/private/...), globs
Usage (from the repo root):
  python -m src.eval.edit_head_cv --train "features/pe_ec3_train/**/*_train.npz" --person "data/private/pe_ec3_train/*.npz"
      --person-eval "data/private/pe_ec3/*.npz" --repeat 0 5 20
"""

import argparse
import csv
import glob
import zlib
from pathlib import Path

import numpy as np
import torch

from src.calibrate.calibrate_edit_check import edit_line, paired, traded_line
from src.inference import load_calibration
from src.models.linear_head import LinearHead

HEAD_F = "weights/open_gen_v1/head_F.pt"
EC3 = "weights/edit_check_v3/head_EC3.pt"
POOLS = {"polished dev": "commons_dev", "OpenImages val": "muse_pairs_val", "originals v1": "muse_edits_v1",
         "originals v2": "muse_edits_test_v2", "OpenImages unseen": "sid_reals_0_9_unseen", "Commons v3 dev": "commons_v3_dev"}
FAKES = {"edits v1 local": ("muse_edits_v1", "muse_edits/local"), "edits v1 whole": ("muse_edits_v1", "muse_edits/whole"),
         "edits v2 local": ("muse_edits_test_v2", "muse_edits/local"), "edits v2 whole": ("muse_edits_test_v2", "muse_edits/whole"),
         "Muse test v1": ("muse_image_test_v1", ""), "Muse pairs val": ("muse_pairs_val", "")}
MANIFESTS = [Path(r"D:\Datasets\SelfEdits\self_edits_v1"), Path(r"D:\private_real2\self_train_v1")]


def load(paths):
    parts = [np.load(p) for p in paths]
    x = np.concatenate([p["features"].reshape(len(p["features"]), -1) for p in parts]).astype(np.float32)
    y = np.concatenate([(p["labels"] > 0).astype(np.float32) for p in parts])
    ids = np.concatenate([p["img_ids"].astype(str) for p in parts])
    return x, y, ids


def fit_near(x, y, start, pull, max_iter=200):
    """Logistic regression (each class counts half) that starts at the EC3 head and is pulled back towards it
    (an L2 pull to the start instead of to zero), so the new data adjusts the head instead of replacing it."""
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    mean, std = x.mean(dim=0), x.std(dim=0) + 1e-6
    z = (x - mean) / std
    n_ai = y.sum()
    weight = torch.where(y == 1, len(y) / (2 * n_ai), len(y) / (2 * (len(y) - n_ai)))
    w0 = (start.weight.float() * std).detach()  # the start head in standardised units
    b0 = (start.bias.float() + (start.weight.float() * mean).sum()).reshape(1).detach()
    w, b = w0.clone().requires_grad_(), b0.clone().requires_grad_()
    optimizer = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y, weight=weight)
        loss = loss + pull * ((w - w0) ** 2).sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return LinearHead(w / std, (b - (w * mean / std).sum()).squeeze())


def group_of():
    """Edit id per folder -> the original selfie it came from (from the edit manifests)."""
    groups = {}
    for folder in MANIFESTS:
        for r in csv.DictReader(open(folder / "manifest.csv", newline="")):
            groups[str(folder / r["id"])] = r["source"]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--person", nargs="+", required=True)
    parser.add_argument("--person-eval", nargs="+", required=True)
    parser.add_argument("--repeat", type=int, nargs="+", default=[0, 5, 20], help="selfie pairs shown K times")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--near", type=float, nargs="*", default=[],
                        help="also: start at the EC3 head with this pull towards it (one variant per value and --repeat)")
    parser.add_argument("--skip-scratch", action="store_true", help="skip the heads trained from zero")
    parser.add_argument("--save-final", type=Path, help="then fit the head from EC3 with --final-pull and --final-repeat "
                        "on ALL selfies and save it here (no held-out selfies left: test on new ones)")
    parser.add_argument("--final-pull", type=float, default=0.1)
    parser.add_argument("--start", default=EC3, help="the head the final fit starts from and is pulled back to "
                        "(head_EC3p for the Gemini head: it keeps the person edits it already knows)")
    parser.add_argument("--final-repeat", type=int, default=5)
    parser.add_argument("--extra", action="append", default=[], help='"GLOB:K": more training features for the final '
                        "head, shown K times (e.g. Google's pictures: features/pe_ec3_train/google/*_train.npz:5)")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    expand = lambda patterns: sorted({f for p in patterns for f in glob.glob(p, recursive=True)})  # noqa: E731
    x_pub, y_pub, _ = load(expand(args.train))
    x_per, y_per, id_per = load(expand(args.person))
    x_ev, y_ev, id_ev = load(expand(args.person_eval))
    kinds = np.array(["original" if not y else ("whole" if "whole" in s else "local")
                      for y, s in zip(y_ev, np.concatenate([np.load(p)["sources"].astype(str) for p in expand(args.person_eval)]))])
    groups = group_of()
    selfie = lambda path: groups[str(Path(path).parent / Path(path).stem.split("_original")[0].split("_edited")[0])]  # noqa: E731
    fold_of = lambda path: zlib.crc32(selfie(path).encode()) % args.folds  # noqa: E731
    f_per, f_ev = np.array([fold_of(p) for p in id_per]), np.array([fold_of(p) for p in id_ev])
    print(f"public training views {len(x_pub)} | selfie training views {len(x_per)} | selfie test pictures {len(x_ev)} "
          f"({(kinds == 'original').sum()} originals, {(kinds == 'local').sum()} local, {(kinds == 'whole').sum()} whole) "
          f"| folds by original selfie: {np.bincount(f_ev, minlength=args.folds).tolist()} pictures")

    whole, cal = LinearHead.load(HEAD_F), load_calibration(HEAD_F)

    def judge(head):
        """The app's joint line for this head on the public pools, and what it catches there."""
        pools = {n: paired(whole, head, f"features/{s}_app.npz:features/pe_ec3/{s}_eval.npz", True) for n, s in POOLS.items()}
        w_line, _ = traded_line(pools, cal["ai_logit"], 1)
        line = edit_line(pools, w_line, 0.005)
        real = {n: float(((w > w_line) | (e > line)).mean()) for n, (w, e, _) in pools.items()}
        fakes = {}
        for n, (stem, prefix) in FAKES.items():
            w, e, _ = paired(whole, head, f"features/{stem}_app.npz:features/pe_ec3/{stem}_eval.npz" + (f"@{prefix}" if prefix else ""), False)
            fakes[n] = float(((w > w_line) | (e > line)).mean())
        return line, real, fakes

    rows = []
    variants = [("EC3 head (now)", None, None)]
    variants += [] if args.skip_scratch else [(f"retrained, selfies x{k}", k, None) for k in args.repeat]
    variants += [(f"from EC3, pull {p:g}, selfies x{k}", k, p) for p in args.near for k in args.repeat if k]
    ec3 = LinearHead.load(EC3)
    for name, k, pull in variants:
        caught = {"original": [], "local": [], "whole": []}
        lines, reals, fakes = [], [], []
        for fold in range(args.folds):
            if k is None:
                head = LinearHead.load(EC3)
            elif k == 0 and fold > 0:
                pass  # no selfies in training: the same head for every fold
            else:
                keep = f_per != fold
                x = np.concatenate([x_pub] + [x_per[keep]] * k) if k else x_pub
                y = np.concatenate([y_pub] + [y_per[keep]] * k) if k else y_pub
                if pull is None:
                    head = LinearHead().fit(torch.from_numpy(x), torch.from_numpy(y), weight_decay=args.weight_decay)
                else:
                    head = fit_near(x, y, ec3, pull)
            line, real, fake = judge(head)
            lines.append(line), reals.append(real), fakes.append(fake)
            held = f_ev == fold
            s = head.logits(torch.from_numpy(x_ev[held])).numpy()
            for kind in caught:
                caught[kind] += list(s[kinds[held] == kind] > line)
            if k is None:
                break  # the EC3 head does not depend on the fold: one pass scores every selfie
        if k is None:  # score every selfie once with the EC3 head
            s = LinearHead.load(EC3).logits(torch.from_numpy(x_ev)).numpy()
            caught = {kind: list(s[kinds == kind] > lines[0]) for kind in caught}
        mean = lambda dicts: {n: np.mean([d[n] for d in dicts]) for n in dicts[0]}  # noqa: E731
        rows.append((name, np.mean(lines), mean(reals), mean(fakes), {kd: np.mean(v) for kd, v in caught.items()}))

    print("\nEdit check (+ head F at its traded line), lines set on the public real pools as in the app; averaged over folds")
    for name, line, real, fake, sel in rows:
        print(f"\n== {name}: edit line {line:+.2f}")
        print("  real called AI or edited: " + " | ".join(f"{n} {v:.1%}" for n, v in real.items()))
        print("  caught: " + " | ".join(f"{n} {v:.1%}" for n, v in fake.items()))
        print(f"  my held-out selfies (edit check alone): originals flagged {sel['original']:.0%} | local edits caught "
              f"{sel['local']:.0%} | whole edits caught {sel['whole']:.0%}")

    if args.save_final:
        x_parts, y_parts = [x_pub] + [x_per] * args.final_repeat, [y_pub] + [y_per] * args.final_repeat
        for spec in args.extra:
            pattern, k = spec.rsplit(":", 1)
            x_e, y_e, _ = load(expand([pattern]))
            x_parts += [x_e] * int(k)
            y_parts += [y_e] * int(k)
        x, y = np.concatenate(x_parts), np.concatenate(y_parts)
        fit_near(x, y, LinearHead.load(args.start), args.final_pull).save(args.save_final)
        print(f"\nsaved {args.save_final}: from {Path(args.start).name}, pull {args.final_pull:g}, all selfie pairs "
              f"x{args.final_repeat}" + "".join(f", {e}" for e in args.extra))


if __name__ == "__main__":
    main()
