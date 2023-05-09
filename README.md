---
title: Xglm 564m
emoji: 🦀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Thai XGLM api

# build docker

    docker build -t scb10x/thaillm .

# run docker

    mkdir .cache #optional
    chmod 777 .cache #optional
    # on cpu
    docker run --restart unless-stopped -p 80:7860 -v `pwd`/.cache:/home/user/.cache -dt scb10x/thaillm
    # on gpu
    docker run --gpus all --restart unless-stopped -p 80:7860 -v `pwd`/.cache:/home/user/.cache -dt scb10x/thaillm

# try with curl

    curl --request POST \
      --url http://localhost:80/generate \
      --header 'Content-Type: application/json' \
      --cookie session-space-cookie=730e8af02a9acbbdb0e941d63f05d41e \
      --data '{
      "input": "hello"
    }'

# how to deploy on hugging face
1. create a new space on hugging face https://huggingface.co/new-space
2. add ssh public key to hugging face via https://huggingface.co/settings/keys
3. add hf as remote repo (change the repo path to your hugging face space)

    git remote add huggingface git@hf.co:spaces/steerapi/xglm-564m

4. push to hf

    git push huggingface main
