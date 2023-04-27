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

