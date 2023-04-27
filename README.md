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

# try with curl

    curl --request POST \
      --url http://localhost:7860/generate \
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