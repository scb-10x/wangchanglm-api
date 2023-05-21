# WangChanGLM API

API Service for the [WangChanGLM model](https://github.com/PyThaiNLP/WangChanGLM). More information about the model can be found [here](https://medium.com/airesearch-in-th/wangchanglm-the-thai-turned-multilingual-instruction-following-model-7aa9a0f51f5f).

Try out the public demo at https://wangchanglm.in.th

## Build docker

    docker build -t scb10x/thaillm .

## Run docker

    mkdir .cache #optional
    chmod 777 .cache #optional
    
    # on cpu
    docker run --restart unless-stopped -p 80:7860 -v `pwd`/.cache:/home/user/.cache -dt scb10x/thaillm

    # on gpu
    docker run --gpus all --restart unless-stopped -p 80:7860 -v `pwd`/.cache:/home/user/.cache -dt scb10x/thaillm

## Request with cURL

    curl --request POST \
      --url http://localhost:80/generate \
      --header 'Content-Type: application/json' \
      --data '{"instruction": "เขียนบทความเกี่ยวกับ ประโยชน์ของการออกกำลังกาย"}'

## Deploy on GCP

### Create a VM
1. Compute Engine > Create an instance
2. GPUs > NVIDIA T4
3. Machine Type > n1-standard-8
4. Boot disk > Switch Image > Debian 10 based Deep Learning VM with M108 > 250GB
5. Firewall > Allow HTTP traffic > Allow HTTPS traffic
6. Create

### Install git-lfs on the VM

    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs

### SSH into the VM and build

1. clone this repo
2. build docker: `./scripts/build.sh`
3. run docker: `./scripts/start.sh`
