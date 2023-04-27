from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# pipe = pipeline("text-generation", model="facebook/xglm-7.5B", revision="sharded")
pipe = pipeline("text-generation", model="facebook/xglm-564M")


class Input(BaseModel):
    input: str
    max_length: int = 512

@app.post("/generate")
def generate(input: Input):
    output = pipe(f"{input.input}", max_length=input.max_length)
    return {"output": output[0]["generated_text"]}
