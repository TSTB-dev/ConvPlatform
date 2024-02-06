from typing import Any

import tiktoken

from .base import BaseConversationLengthAdjuster
from .model_config import MODEL_CONFIG, MODEL_POINT
from .chat_utils import get_n_tokens

import logging
from logging import config

class OldConversationTruncationModule(BaseConversationLengthAdjuster):
    """Adjust the length of the conversation.

    Args:
        BaseConversationLengthAdjuster (_type_): _description_
    """
    
    def __init__(
        self,
        model_name: str,
        context_length: int,
    ):
        if model_name in MODEL_POINT.keys():
            model_name = MODEL_POINT[model_name]
        
        assert (
            model_name in MODEL_CONFIG.keys()
        ), f"model_name must be in {MODEL_CONFIG.keys()}"
        
        self.model_name = model_name
        self.context_length = context_length
    
    def run(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Adjust the length of the conversation.

        Args:
            messages (list[dict[str, str]]): conversation history

        Returns:
            list[dict[str, str]]: adjusted conversation history
        """
        adjusted_messsages: list[dict[str, str]] = []
        total_n_tokens = 0
        print(f"Message: {messages}")
        
        for message in messages[::-1]:
            if total_n_tokens <= self.context_length:
                message, total_n_tokens = self.adjust_message_length_and_update_total_tokens(
                    message, total_n_tokens
                )
                
                if message is not None:
                    adjusted_messsages.append(message)
            
        return adjusted_messsages[::-1]
    
    
    def adjust_message_length_and_update_total_tokens(
        self, message: dict[str, str], total_n_tokens: int = 0,
    ) -> str:
        n_tokens = get_n_tokens(message, self.model_name)
        if total_n_tokens + n_tokens["total"] <= self.context_length:
            total_n_tokens += n_tokens["total"]
            
            return message, total_n_tokens
        else:
            available_n_tokens = max(
                self.context_length - total_n_tokens - n_tokens["other"], 0
            )
            if available_n_tokens > 0:
                if isinstance(message["content"], str):
                    message["content"] = self.truncate(message["content"], available_n_tokens)
                    total_n_tokens += available_n_tokens + n_tokens["other"]
                    logger.debug(f"Truncated message: {message}")
                    return message, total_n_tokens
                else:
                    raise ValueError(
                        f"message['content'] must be a str. Got {type(message['content'])}"
                    )
            else: 
                return None, total_n_tokens
                    
    
    def truncate(self, text: str, n_tokens: int) -> str:
        try:
            TOKENIZER = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            print(f"Model name: {self.model_name} not found. Use cl100k_base instead.")
            TOKENIZER = tiktoken.get_encoding("cl100k_base")
        
        if n_tokens > 0:
            return TOKENIZER.decode(TOKENIZER.encode(text)[-n_tokens:])
        else:
            return ""
        
                    
                    
        