#!/bin/sh
# Night run (1-2 Nov): face crops for the face-swap fine-tune, then the fine-tune itself (src/models/tune_face_pe.py).
# Stream A (photos) and stream B (DFDC videos) make the crops in parallel; training starts when both are done AND the
# GPU is free (the app must be closed). Logs: D:/Datasets/FaceCrops/logs/.
cd "D:/deepfake analyzer" || exit 1
PY=.venv/Scripts/python.exe
L=features/lists
OUT=D:/Datasets/FaceCrops
LOG=$OUT/logs
mkdir -p "$LOG"
SID=D:/Datasets/SID_Set/data
EXCL="$L/commons_dev.csv $L/commons_v4_dev.csv $L/real_face_crops_dev_all.csv $L/sid_reals_0_9_unseen.csv $L/muse_pairs_val.csv"

stream_a() {
  # dev pools in the app's view (they set the strict line during training)
  $PY -m src.faces.make_crops list $L/commons_dev.csv --kind app --name pool_commons_dev --out $OUT
  $PY -m src.faces.make_crops list $L/muse_pairs_val.csv --kind app --name pool_muse_pairs_val --out $OUT
  $PY -m src.faces.make_crops list $L/sid_reals_0_9_unseen.csv --kind app --name pool_sid_reals_0_9_unseen --out $OUT
  $PY -m src.faces.make_crops list $L/real_face_crops_dev_all.csv --kind app --name pool_real_face_crops_dev_all --out $OUT
  $PY -m src.faces.make_crops list $L/commons_v4_dev.csv --kind app --name pool_commons_v4_dev --out $OUT
  $PY -m src.faces.make_crops list $SID/validation-00000-of-00034.parquet --kind app --name pool_sid_val --out $OUT
  # OpenForensics Val: 85% of the photos train (wide crops), 15% dev (app crops)
  $PY -m src.faces.make_crops of --split Val --out $OUT
  # more real faces for training: SID-Set real photos (train files 10-19) and Commons training photos
  $PY -m src.faces.make_crops list $SID/train-000[1][0-9]-of-00249.parquet --kind wide --name real_sid --exclude $EXCL --out $OUT
  $PY -m src.faces.make_crops list $L/commons_train.csv $L/commons_v2_train.csv $L/commons_v3_train.csv $L/commons_v4_train.csv \
      --kind wide --name real_commons --exclude $EXCL --out $OUT
  # OpenForensics Test-Dev sample (the untouched test, app crops)
  $PY -m src.faces.make_crops of --split Test-Dev --out $OUT
  echo done > "$LOG/stream_a.done"
}

stream_b() {
  $PY -m src.faces.make_crops dfdc D:/Datasets/DFDC/dfdc_train_part_3 --kind app --name dfdc_p3_dev --out $OUT
  $PY -m src.faces.make_crops dfdc D:/Datasets/DFDC/dfdc_train_part_0 D:/Datasets/DFDC/dfdc_train_part_1 \
      D:/Datasets/DFDC/dfdc_train_part_2 --kind wide --name dfdc_train --out $OUT
  echo done > "$LOG/stream_b.done"
}

rm -f "$LOG/stream_a.done" "$LOG/stream_b.done"
stream_a > "$LOG/stream_a.txt" 2>&1 &
stream_b > "$LOG/stream_b.txt" 2>&1 &
wait
echo "crops done: $(date)" >> "$LOG/chain.txt"

# training needs the whole 8 GB GPU: wait until nothing else uses it (e.g. the app)
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 1500 ]; do
  echo "waiting for a free GPU: $(date)" >> "$LOG/chain.txt"
  sleep 60
done
echo "training starts: $(date)" >> "$LOG/chain.txt"
$PY -m src.models.tune_face_pe --crops $OUT --out-dir weights/face_pe_v1 --steps 5000 --eval-every 500 > "$LOG/train.txt" 2>&1
echo "training done: $(date)" >> "$LOG/chain.txt"
