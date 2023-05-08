from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

import torch

app = FastAPI()

# pipe = pipeline("text-generation", model="facebook/xglm-7.5B", revision="sharded")
# pipe = pipeline("text-generation",
#                 model="pythainlp/wangchanglm-7.5B-sft-enth", load_in_8bit=True,
#                 offload_folder="./",
#                 low_cpu_mem_usage=True,)
model_name = "pythainlp/wangchanglm-7.5B-sft-enth"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=False,
    # load_in_8bit=True,
    # device_map="auto",
    # load_in_8bit_fp32_cpu_offload=True,
    torch_dtype=torch.float32,
    offload_folder="./",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class Input(BaseModel):
    input: str
    max_length: int = 64


@app.post("/generate")
def generate(input: Input):
    batch = tokenizer(input.input, return_tensors="pt")
    output_tokens = model.generate(
        input_ids=batch["input_ids"],
        max_new_tokens=input.max_length,  # 512
        begin_suppress_tokens=[],
        no_repeat_ngram_size=2,

        # oasst k50
        top_k=50,
        top_p=0.95,  # 0.95
        typical_p=1.,
        temperature=0.9,  # 0.9

        # #oasst typical3
        # typical_p = 0.3,
        # temperature = 0.8,
        # repetition_penalty = 1.2,
    )
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # output = pipe(f"{input.input}", max_length=input.max_length)
    return {"output": output}
