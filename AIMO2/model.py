import os
import polars as pl
# import kaggle_evaluation.aimo_2_inference_server
import gc
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
import math
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm
import re
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # "0,1,2,3"
import vllm
llm = vllm.LLM(
    "Qwen/Qwen2.5-Math-7B-Instruct",
    tensor_parallel_size=1, # 4 
    gpu_memory_utilization=0.96, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len = 1000
)
tokenizer = llm.get_tokenizer()
def generate_text_vllm(requests, tokenizer, model):
    sampling_params = vllm.SamplingParams(
      temperature=0.00,
      max_tokens=1024
    )
    responses = model.generate(requests, sampling_params=sampling_params, use_tqdm=False)
    response_text_list = []
    for response in responses:
        # total_tokens += len(response.outputs[0].token_ids)
        response_text_list.append(response.outputs[0].text)
    return response_text_list
def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)
tool_instruction = '\nPlease solve the problem above, and put your final answer within \\boxed{}. Use math competition strategies to solve these problems and use a proof based problem solving method. From that method, create a python program to solve the problem and execute that to find the solution.'
# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: str, question: str) -> pl.DataFrame:
    """Make a prediction."""
    prompt = question + tool_instruction
    generate_text = generate_text_vllm([prompt], tokenizer, llm)[0]
    print(f"Generated text for ID {id_}: {generate_text}")
    answer = 2
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', generate_text)
        if len(result_output) > 0:
            no = naive_parse(result_output[0])
            if len(no) > 0:
                answer = int(no) % 1000
            print(answer)
    except:
        print('error')
    return pl.DataFrame({'id': [id_], 'answer': [answer]})
# Load the reference data