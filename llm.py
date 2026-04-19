from gpt4all import GPT4All
from config import *
 
print("Loading LLM...")
 
llm = GPT4All(
    model_name=LLM_MODEL,
    model_path=LLM_MODEL_PATH,  
    allow_download=False,
    device="cpu",
    n_threads=6
)
 
 
def generate_answer(prompt):
    response = llm.generate(
        prompt,
        max_tokens=200,
        temp=0.2
    )
    return response
 
 
def generate_stream(prompt):
    for token in llm.generate(
        prompt,
        max_tokens=200,
        temp=0.2,
        streaming=True
    ):
        yield token
 