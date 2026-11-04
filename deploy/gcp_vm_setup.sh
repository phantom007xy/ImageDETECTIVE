#!/usr/bin/env bash
# Sets up (or updates) the ImageDETECTIVE web app on a Google Cloud VM (Ubuntu 24.04, CPU only):
#   - the code from the public GitHub repo in /opt/ImageDETECTIVE, CPU PyTorch + requirements in /opt/venv
#   - the app (python -m app.server) as the systemd service "imagedetective" on 127.0.0.1:7860
#   - Caddy in front of it on ports 80/443, with an automatic HTTPS certificate for <ip-with-dashes>.sslip.io
# Secrets are NOT in this file or in git: /etc/imagedetective.env (root only) must hold HF_TOKEN (gated DINOv3 and
# SAM 3, and the private heads repo), HEADS_REPO (the private Hugging Face repo with our trained heads and fine-tuned
# encoders, see scripts/upload_heads.py), MODEL_API_KEY (Meta Model API) and API_DAILY_CAP_USD.
# Run as root on the VM: sudo bash gcp_vm_setup.sh   (again after a git push, to update the code and restart)
set -euo pipefail
REPO=https://github.com/phantom007xy/ImageDETECTIVE
DIR=/opt/ImageDETECTIVE
[ -f /etc/imagedetective.env ] || { echo "missing /etc/imagedetective.env (secrets)"; exit 1; }

apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git curl gpg debian-keyring debian-archive-keyring apt-transport-https >/dev/null
if ! command -v caddy >/dev/null; then  # Caddy's official package repository
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy >/dev/null
fi

id imagedetective >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/imagedetective imagedetective
if [ -d $DIR/.git ]; then git config --global --add safe.directory $DIR; git -C $DIR pull --ff-only -q; else git clone -q $REPO $DIR; fi
chown -R imagedetective $DIR  # the app downloads the private heads into $DIR/weights at its first start
[ -d /opt/venv ] || python3 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
/opt/venv/bin/pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
/opt/venv/bin/pip install -q -r $DIR/requirements.txt
/opt/venv/bin/pip check

cat > /etc/systemd/system/imagedetective.service <<EOF
[Unit]
Description=ImageDETECTIVE web app
After=network-online.target
Wants=network-online.target

[Service]
User=imagedetective
WorkingDirectory=$DIR
EnvironmentFile=/etc/imagedetective.env
Environment=HF_HOME=/var/lib/imagedetective/hf-cache HOST=127.0.0.1 PORT=7860 PYTHONUNBUFFERED=1
ExecStart=/opt/venv/bin/python -m app.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

IP=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
HOSTNAME_PUBLIC="${IP//./-}.sslip.io"
cat > /etc/caddy/Caddyfile <<EOF
$HOSTNAME_PUBLIC {
    request_body {
        max_size 25MB
    }
    reverse_proxy 127.0.0.1:7860
}
EOF

systemctl daemon-reload
systemctl enable -q imagedetective caddy
systemctl restart imagedetective caddy
for attempt in $(seq 1 120); do
    if curl --fail --silent --max-time 5 http://127.0.0.1:7860/healthz >/dev/null; then
        echo "detector ready: https://$HOSTNAME_PUBLIC"
        exit 0
    fi
    sleep 5
done
echo "Detector readiness did not pass. Check: journalctl -u imagedetective" >&2
echo "Initial model downloads can take longer; verify /healthz before sharing the demo." >&2
exit 1
