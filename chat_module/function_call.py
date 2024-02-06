from typing import Dict, Callable
from pydantic import BaseModel

class CallableFunction(BaseModel):
    function: Callable
    name: str
    description: dict
    
