#!/bin/bash
# Runs a command with this project's .venv python, which sees both the
# container's torch (via --system-site-packages) and the editable-installed
# llm_from_scratch package. Needed because `module load pytorch/...` only
# wraps the bare "python" name, not .venv/bin/python.
SIF=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Host RHEL SSL_CERT_FILE/DIR leak into the container and point at paths that
# don't exist there, breaking HTTPS (e.g. downloading TinyShakespeare).
# Override to the container's own valid cert directory.
exec apptainer exec --nv --env SSL_CERT_DIR=/etc/ssl/certs "$SIF" "$DIR/.venv/bin/python" "$@"
