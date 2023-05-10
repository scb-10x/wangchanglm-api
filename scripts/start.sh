#!/bin/bash
docker run --gpus all --restart unless-stopped -p 80:7860 -v `pwd`/.cache:/home/user/.cache -dt scb10x/thaillm
