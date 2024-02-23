FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

COPY pretrained/ /pretrained/

RUN pip install tensorflow_text tensorflow tensorflow_hub

WORKDIR /app

COPY main.py /app/main.py
COPY protector /app/protector/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
