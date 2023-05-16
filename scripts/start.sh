#!/bin/bash

# Check if running on a GCE VM
if curl -s -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance" >/dev/null 2>&1; then
  # Running on a GCE VM
  docker run --gpus all --restart unless-stopped -p 80:7860 --log-driver=gcplogs -v "$(pwd)/.cache:/home/user/.cache" -dt scb10x/thaillm
else
  # Not running on a GCE VM
  docker run --gpus all --restart unless-stopped -p 80:7860 -v "$(pwd)/.cache:/home/user/.cache" -dt scb10x/thaillm
fi
