import json
import os
import asyncio

from .base import BaseConversationMemory

COMPRESS_PROMPT = """
会話履歴から，今後の会話に必要な情報のみを抽出します．これまでの会話の記憶は以下の通りです．
記憶：{}\n
この記憶を参照して，以下の1ターンの会話から必要な情報があれば，記憶を更新してください．
会話履歴：{}\n

更新後の記憶: 
"""

class ConversationMemory(BaseConversationMemory):
    def __init__(self, save_path: str, memory_length: int, \
        compress_model, exists_ok: bool = False, prompt: str = COMPRESS_PROMPT):
        self.memory = ""
        self.save_path = save_path
        self.memory_length = memory_length
        self.compress_model = compress_model
        self.compress_prompt = prompt

    def store(self):
        with open(self.save_path, "w") as f:
            f.write("\n".join(self.memory))
            
    def load(self) -> str:
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                return f.read()
        else:
            return ""
    
    def update(self, messages: list[dict[str, str]]):
        pass
    
    async def aupdate(self, messages: list[dict[str, str]]):
        prompt = self.compress_prompt.format(self.memory, messages)
        compressed_memory = await self.compress_model.arun(
            prompt
        )
        self.memory = compressed_memory
        

# WARN: Duplucated 
class JSONConversationMemory(BaseConversationMemory):
    
    def __init__(self, path: str = "conversation_history.json", exists_ok: bool = True):
        self.path = path
        if not os.path.exists(path) and not exists_ok:
            os.remove(path)
    
    def store(self, conversation_history: list[dict[str, str]]):
        with open(self.path, "w") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=4)
    
    def load(self) -> list[dict[str, str]]:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        else:
            return []
        
    