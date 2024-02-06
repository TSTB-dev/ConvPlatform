import base64
import io
import math
import os
from typing import Optional, Union

import openai
import tiktoken
from openai import AsyncOpenAI, OpenAI

from .model_config import MODEL_CONFIG

MODEL_ZOO = set(MODEL_CONFIG.keys())

def get_client(
    api_key_env_name: str,
    time_out: int = 60,
    max_retries: int = 5,
):
    return OpenAI(
        api_key=os.environ[api_key_env_name],
        max_retries=max_retries,
        timeout=time_out,
    )
    
def get_async_client(
    api_key_env_name: str,
    time_out: int = 60,
    max_retries: int = 5,
):
    return AsyncOpenAI(
        api_key=os.environ[api_key_env_name],
        max_retries=max_retries,
        timeout=time_out,
    )

def get_n_tokens(message, model_name: str):
    """Return the number of tokens in the message.

    Args:
        message (dict[str, str  |  list[dict[str, str | dict[str, str]]]]): message
        model_name (str): model name
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    n_content_tokens = 0
    n_other_tokens = 0
    
    for key, value in message.items():
        if key == "content":
            n_content_tokens += len(encoding.encode(value))
        elif key == "name":
            n_other_tokens += len(encoding.encode(value))
        else:
            n_other_tokens += len(encoding.encode(value))
    
    total_tokens = n_content_tokens + n_other_tokens
    return {"total": total_tokens, "content": n_content_tokens, "other": n_other_tokens}

def get_token_limit(model_name: str):
    if model_name in MODEL_ZOO:
        return MODEL_CONFIG[model_name]["max_tokens"]
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")
    
def make_batch(
    iterable,
    batch_size: int = 1
):
    length = len(iterable)
    for ndx in range(0, length//batch_size, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]