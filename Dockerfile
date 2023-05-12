# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM wallies/python-cuda:3.10-cuda11.6-runtime

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

#RUN pip install bitsandbytes-cuda117

#RUN python3 -c 'import torch; from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained("pythainlp/wangchanglm-7.5B-sft-enth",return_dict=False,load_in_8bit=True,device_map="auto",torch_dtype=torch.float16)'

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# RUN pip install tensorflow_text

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
