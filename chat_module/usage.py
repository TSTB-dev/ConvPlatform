# Used to track the usage of the chat module.
from __future__ import annotations

from openai.types import CompletionUsage
from pydantic import BaseModel

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def __add__(self, other: __class__ | dict | CompletionUsage):
        if isinstance(other, dict):
            other = Usage(**other)
        if hasattr(other, 'prompt_tokens'):
            prompt_tokens = self.prompt_tokens + other.prompt_tokens
        else:
            prompt_tokens = self.prompt_tokens 
        if hasattr(other, 'completion_tokens'):
            completion_tokens = self.completion_tokens + other.completion_tokens
        else:
            completion_tokens = self.completion_tokens
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
    
    def __sub__(self, other: __class__ | dict | CompletionUsage):
        if isinstance(other, dict):
            other = Usage(**other)
            
        if hasattr(other, 'prompt_tokens'):
            prompt_tokens = self.prompt_tokens - other.prompt_tokens
        else:
            prompt_tokens = self.prompt_tokens
        if hasattr(other, 'completion_tokens'):
            completion_tokens = self.completion_tokens - other.completion_tokens
        else:
            completion_tokens = self.completion_tokens
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens
    
    # @field_validator('prompt_tokens')
    # def check_prompt_tokens(cls, v):
    #     if v < 0:
    #         raise ValueError("prompt_tokens must be a positive integer.")
    #     return v

    # @field_validator('completion_tokens')
    # def check_completion_tokens(cls, v):
    #     if v < 0:
    #         raise ValueError("completion_tokens must be a positive integer.")
    #     return v
    
    def __repr__(self):
        return f"Usage(prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens})"