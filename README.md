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

    docker build -t steerapi/xglm .

# run docker
    
    docker run -p 7860:7860 -v `pwd`/.cache:/home/user/.cache -t steerapi/xglm

# how to deploy on hugging face
1. create a new space on hugging face
2. add ssh public key to hugging face via https://huggingface.co/settings/keys
3. add hf as remote repo (change the repo path to your hugging face space)

    git remote add huggingface git@hf.co:spaces/steerapi/xglm-564m

4. push to hf

    git push huggingface main