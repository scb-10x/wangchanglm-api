from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

import torch

PROMPT_FORMATS = {
    "with_context": (
        "<context>: {context}\n<human>: {instruction}\n<bot>: "
    ),
    "no_context": (
        "<human>: {instruction}\n<bot>: "
    ),
}

app = FastAPI()

# pipe = pipeline("text-generation", model="facebook/xglm-7.5B", revision="sharded")
# pipe = pipeline("text-generation",
#                 model="pythainlp/wangchanglm-7.5B-sft-enth", load_in_8bit=True,
#                 offload_folder="./",
#                 low_cpu_mem_usage=True,)
model_name = "pythainlp/wangchanglm-7.5B-sft-enth"
# model_name = "facebook/xglm-564M"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=False,
    load_in_8bit=True,
    device_map="auto",
    # load_in_8bit_fp32_cpu_offload=True,
    torch_dtype=torch.float16,
    # offload_folder="./",
    # low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class GenerateParams(BaseModel):
    instruction: str
    context: str = ""
    max_length: int = 64
    no_repeat_ngram_size: int = 2
    top_k: int = 50
    top_p: float = 0.95
    typical_p: float = 1.
    temperature: float = 0.9

def format_prompt(params: GenerateParams):
    """
    Generate a formatted prompt string based on the given parameters.

    Args:
        params (GenerateParams): A named tuple containing at least the following fields:
            - instruction (str): The instruction to display in the prompt.
            - context (str): The context to display in the prompt, if any.

    Returns:
        str: The formatted prompt string.
    """
    if params.context == 'none' or len(params.context) < 2:
        prompt = PROMPT_FORMATS['no_context'].format_map(
            {'instruction': params.instruction, 'context': ''})
    else:
        prompt = PROMPT_FORMATS['prompt_input'].format_map(
            {'instruction': params.instruction, 'context': params.context})
    return prompt


### ROUTES ###

@app.post("/generate")
def generate(params: GenerateParams):
    prompt = format_prompt(params)
    # print(f"Prompt: {prompt}")

    batch = tokenizer(prompt, return_tensors="pt")
    input_ids = batch["input_ids"].to('cuda')
    with torch.cuda.amp.autocast(): 
        output_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=params.max_length,  # 512
            begin_suppress_tokens=[],
            no_repeat_ngram_size=params.no_repeat_ngram_size,

            # oasst k50
            top_k=params.top_k,
            top_p=params.top_p,  # 0.95
            typical_p=params.typical_p,
            temperature=params.temperature,  # 0.9

            # #oasst typical3
            # typical_p = 0.3,
            # temperature = 0.8,
            # repetition_penalty = 1.2,
        )
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # output = pipe(f"{input.input}", max_length=input.max_length)
    return {"output": output}
