"""Writes notebooks/colab_face_swap.ipynb: the thin Colab launcher for the face-swap training (night 2-3 Nov).
The notebook only clones the repo and calls scripts (logic stays in src/ and scripts/). Run: python scripts/make_colab_notebook.py
"""

import json
from pathlib import Path

DRIVE = "/content/drive/MyDrive/ImageDETECTIVE"
TRAIN_SETS = ("of_val_train of_train_train of_test_challenge_train dfdc_train rtfs_train real_sid real_commons "
              "real_fairface real_openimages real_faceid550_train real_indian")
HELD = ("pool_commons_dev pool_muse_pairs_val pool_sid_reals_0_9_unseen pool_real_face_crops_dev_all pool_commons_v4_dev "
        "pool_sid_val pool_faceid550_dev pool_commons_v3_dev of_val_dev of_test_dev dfdc_p3_dev pool_muse_image_test_v1 "
        "pool_sd_faces pool_tpdne_test rtfs_dev rtfs_test")
# swaps 40% of a batch (photo swaps, older video swaps, today's swap tools), each with its own real faces; 6 more real sets
SHARES = ('--share "openforensics/fake=0.15" --share "dfdc/fake=0.10" --share "rtfs/fake=0.15" '
          '--share "openforensics/real=0.12" --share "dfdc/real=0.08" --share "rtfs/real=0.10" --share "sid_set/real=0.05" '
          '--share "commons/real=0.05" --share "fairface/real=0.05" --share "openimages/real=0.05" '
          '--share "faceid550/real=0.05" --share "indian/real=0.05"')
COMMON = (f"--crops /content/FaceCrops --train-sets {TRAIN_SETS} {SHARES} --steps 20000 --eval-every 2000 "
          "--lr 2e-5 --p-damage 0.5 --p-phone 0.4 --dev-max 1500 --workers 10")

cells = [
    ("markdown", f"""# ImageDETECTIVE: face-swap model training (Colab, A100)
Runs everything in order and **disconnects the runtime at the end** (so no compute units are wasted).
Every step skips work that is already done, so after a disconnect you can simply run all cells again.

Before you start:
1. Runtime > Change runtime type > **A100 GPU**.
2. Secrets (key icon on the left), each with notebook access on: **HF_TOKEN** = your Hugging Face token (SAM 3 is
   gated); **KAGGLE_USERNAME** and **KAGGLE_KEY** = the two values in your kaggle.json (for RTFS-10k; without them the
   run goes on without it)
3. `My Drive/ImageDETECTIVE/FaceCrops_colab.zip` (or the unzipped `FaceCrops` folder) uploaded.
4. Keep this tab open and the laptop awake (Google AI Pro has no background execution).

Logs, checkpoints and results go to `{DRIVE}/` (private). DFDC data and the trained weights must never be shared."""),
    ("code", f"""# 1. GPU, Drive, settings
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess, time
DRIVE = '{DRIVE}'
os.makedirs(f'{{DRIVE}}/logs', exist_ok=True)
os.makedirs(f'{{DRIVE}}/runs', exist_ok=True)
import torch
gpu_mem = torch.cuda.get_device_properties(0).total_memory
# A100 (40-80 GB): batch 32 without gradient checkpointing (fastest); a smaller GPU: batch 16 with it (fits)
FAST = '--batch-size 32 --no-checkpointing' if gpu_mem > 35e9 else '--batch-size 16'
print(torch.cuda.get_device_name(0), f'{{gpu_mem / 1e9:.0f}} GB ->', FAST)
def run(cmd, log):
    \"\"\"Runs a shell command, prints its last lines, keeps the full log on Drive.\"\"\"
    t = time.time()
    with open(f'{{DRIVE}}/logs/{{log}}.txt', 'a') as f:
        f.write(f'\\n$ {{cmd}}\\n')
        f.flush()
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd='/content/ImageDETECTIVE')
    tail = open(f'{{DRIVE}}/logs/{{log}}.txt').read().splitlines()[-6:]
    print(f'[{{log}}] exit {{p.returncode}} after {{(time.time() - t) / 60:.0f}} min'); print('\\n'.join(tail))
    return p.returncode"""),
    ("code", """# 2. Code (public repo) and the same library versions as the laptop
!rm -rf /content/ImageDETECTIVE && git clone -q https://github.com/phantom007xy/ImageDETECTIVE /content/ImageDETECTIVE
!pip install -q transformers==5.21.0 huggingface_hub==2.3.0 timm==1.0.30 opencv-python-headless==5.0.0.93 httpx==0.28.1 pyarrow gdown
!apt-get -qq install -y unrar p7zip-full ffmpeg > /dev/null
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
    try:
        os.environ[key] = userdata.get(key)
    except Exception:
        print(f'no {key} secret: RTFS-10k is skipped')
from huggingface_hub import login
login(os.environ['HF_TOKEN'])
!cd /content/ImageDETECTIVE && git log --oneline -1"""),
    ("code", """# 3. Our existing face crops (DFDC, OpenForensics Val, SID-Set, Commons, dev and test pools): from FaceCrops_colab.zip
#    or the uploaded FaceCrops folder in Drive; paths made Colab paths, rows whose file is missing dropped
run(f'python scripts/colab_facecrops.py --drive {DRIVE} --out /content/FaceCrops', 'facecrops')
!tail -20 $DRIVE/logs/facecrops.txt"""),
    ("code", """# 4. Downloads (Colab's fast network): OpenForensics Train + Test-Challenge (CC BY 4.0), RTFS-10k (CC BY-SA 4.0)
#    and the new real-face sets (CC BY)
OF = '/content/data/OpenForensics'
os.makedirs(OF, exist_ok=True)
for split in ('Train', 'Test-Challenge'):
    if not os.path.exists(f'{OF}/{split}_poly.json'):
        !curl -s -L -o {OF}/{split}_poly.json "https://zenodo.org/records/5528418/files/{split}_poly.json?download=1"
# one split after the other, deleting its zips after unpacking (disk space)
of_dl = subprocess.Popen('for s in Train Test-Challenge; do python scripts/download_openforensics.py --split $s --n 0 --workers 12 '
                         '--data-dir /content/data/OpenForensics && rm -f /content/data/OpenForensics/${s}_part_*.zip; done'
                         f' > {DRIVE}/logs/download_openforensics.txt 2>&1', shell=True, cwd='/content/ImageDETECTIVE')
rtfs_dl = (subprocess.Popen(f'python scripts/download_rtfs.py --out /content/data/rtfs > {DRIVE}/logs/download_rtfs.txt 2>&1',
                            shell=True, cwd='/content/ImageDETECTIVE') if os.environ.get('KAGGLE_KEY') else None)
run('python scripts/download_real_faces.py fairface openimages faceid550 indian --fairface-n 30000 --openimages-n 10000 '
    '--out /content/data/real', 'download_real_faces')
of_dl.wait()
print('OpenForensics Train:', of_dl.returncode); print(open(f'{DRIVE}/logs/download_openforensics.txt').read()[-600:])
if rtfs_dl:
    rtfs_dl.wait()
    print('RTFS-10k:', rtfs_dl.returncode); print(open(f'{DRIVE}/logs/download_rtfs.txt').read()[-600:])
!rm -f /content/data/OpenForensics/Train_part_*.zip /content/data/real/*.zip
!df -h /content | tail -1"""),
    ("code", """# 5. Face crops with SAM 3 (the app's settings and crop rule), several processes at once on the GPU.
#    RTFS-10k: 4 processes (re-encoding its real clips like the swaps takes CPU time), then joined + the coding check
jobs = []
of_cmds = [f'python -m src.faces.make_crops of --split {s} --data-dir /content/data/OpenForensics --out /content/FaceCrops'
           for s, m in (('Train', 'of_train_train'), ('Test-Challenge', 'of_test_challenge_train'))
           if not os.path.exists(f'/content/FaceCrops/{m}.csv')]
if of_cmds:
    jobs.append(subprocess.Popen(' && '.join(of_cmds) + f' > {DRIVE}/logs/crops_openforensics_train.txt 2>&1', shell=True,
                                 cwd='/content/ImageDETECTIVE'))
real = [('real_fairface', 'wide'), ('real_openimages', 'wide'), ('real_faceid550_train', 'wide'), ('real_indian', 'wide')]
cmds = [f'python -m src.faces.make_crops list features/lists/{n}.csv --kind {k} --name {n} --out /content/FaceCrops'
        for n, k in real if os.path.exists(f'/content/ImageDETECTIVE/features/lists/{n}.csv') and not os.path.exists(f'/content/FaceCrops/{n}.csv')]
if os.path.exists('/content/ImageDETECTIVE/features/lists/real_faceid550_dev.csv') and not os.path.exists('/content/FaceCrops/pool_faceid550_dev.csv'):
    cmds.append('python -m src.faces.make_crops list features/lists/real_faceid550_dev.csv --kind app --name pool_faceid550_dev --out /content/FaceCrops')
if cmds:
    jobs.append(subprocess.Popen(' && '.join(cmds) + f' > {DRIVE}/logs/crops_real.txt 2>&1', shell=True, cwd='/content/ImageDETECTIVE'))
rtfs = os.path.exists('/content/data/rtfs/inswapper.csv') and not os.path.exists('/content/FaceCrops/rtfs_dev.csv')
if rtfs:
    for i in range(4):
        jobs.append(subprocess.Popen(f'python -m src.faces.make_crops rtfs --data-dir /content/data/rtfs --shard {i}/4 '
                                     f'--out /content/FaceCrops > {DRIVE}/logs/crops_rtfs_{i}.txt 2>&1', shell=True,
                                     cwd='/content/ImageDETECTIVE'))
for j in jobs:
    j.wait()
if rtfs:
    run('python -m src.faces.make_crops rtfs --merge 4 --out /content/FaceCrops', 'crops_rtfs_merge')
    !rm -rf /content/data/rtfs/original_videos /content/data/rtfs/inswapper /content/data/rtfs/uniface
!tail -2 $DRIVE/logs/crops_openforensics_train.txt; tail -8 $DRIVE/logs/crops_real.txt
!wc -l /content/FaceCrops/*.csv | tail -25"""),
    ("code", f"""# 6. Near-duplicate check: training faces that (almost) repeat a dev/test face are removed (OpenForensics reuses photos)
run('python scripts/dedupe_crops.py --crops /content/FaceCrops --train {TRAIN_SETS} --held {HELD}', 'dedupe')
# a copy of all crop lists on Drive (small), for the laptop's records
!mkdir -p $DRIVE/crop_lists && cp /content/FaceCrops/*.csv $DRIVE/crop_lists/
!cp /content/ImageDETECTIVE/data/attribution/openimages_faces_v1.csv $DRIVE/ 2>/dev/null; cp /content/ImageDETECTIVE/features/lists/real_*.csv $DRIVE/crop_lists/ 2>/dev/null"""),
    ("code", f"""# 7. Run A: WITHOUT self-blended fakes (from the original Perception Encoder)
if not os.path.exists(f'{{DRIVE}}/runs/A_noblend/pe_step20000.pt'):
    run(f'python -m src.models.tune_face_pe {COMMON} {{FAST}} --p-blend 0 --out-dir {{DRIVE}}/runs/A_noblend', 'run_A_noblend')"""),
    ("code", f"""# 8. Run B: WITH self-blended fakes (everything else the same)
if not os.path.exists(f'{{DRIVE}}/runs/B_blend/pe_step20000.pt'):
    run(f'python -m src.models.tune_face_pe {COMMON} {{FAST}} --p-blend 0.5 --out-dir {{DRIVE}}/runs/B_blend', 'run_B_blend')"""),
    ("code", """# 9. Test both runs at several steps (dev and test crops here; WildDeepfake and my private photos on the laptop)
for r in ('A_noblend', 'B_blend'):
    for s in (6000, 10000, 14000, 20000):
        if os.path.exists(f'{DRIVE}/runs/{r}/pe_step{s}.pt'):
            run(f'python -m src.eval.face_pe_eval --crops /content/FaceCrops --pe {DRIVE}/runs/{r}/pe_step{s}.pt '
                f'--head {DRIVE}/runs/{r}/head_step{s}.pt', f'eval_{r}')
!grep -h "strict line\\|caught\\|REAL faces" $DRIVE/logs/eval_*.txt | head -60"""),
    ("code", """# 10. Done: disconnect the runtime so no compute units are used while nobody is here
from google.colab import runtime
runtime.unassign()"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if kind == "code" else {})} for kind, src in cells]}
out = Path("notebooks/colab_face_swap.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
