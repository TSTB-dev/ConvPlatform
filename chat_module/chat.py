import asyncio
import json
from copy import deepcopy
from typing import Any, Optional, List, Dict, Tuple

from openai import OpenAI

from .base import BaseConversationLengthAdjuster, BaseConversationMemory, BaseFilter, BaseModule
from .result import CompletionResults, EmbeddingResults, FunctionCallingResults, RetrievalResult, ToolOutput
from .usage import Usage
from .chat_utils import get_client, get_async_client, get_n_tokens, get_token_limit, make_batch
from .message import Message
from .truncate import OldConversationTruncationModule
from .function_call import CallableFunction
from .memory import ConversationMemory

import logging
from logging import config

LOG_CONFIG_PATH = "./logging_config.ini"
config.fileConfig(LOG_CONFIG_PATH)
logger = logging.getLogger()

TASK_COMPRESS_PROMPT = """
以下のタスク情報を埋めるために必要な情報のみを残してください．
タスク情報: {} \n

現在の会話履歴: {} \n

維持する会話履歴: {} \n
"""


def completion(
    client: OpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    stop: Optional[List[str]] = None,
    tools: List[dict] = None,
    tool_choice: str = "auto",
    **kwargs
):
    """Simply calls the OpenAI Chat completion API with the given parameters."""
    params = dict(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        stream=stream,
        **kwargs,
    )
    return client.chat.completions.create(**params)

async def acompletion(
    client: OpenAI,
    model_name: str,
    messages: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stream: bool,
    stop: Optional[List[str]] = None,
    tools: List[dict] = None,
    tool_choice: str = "auto",
    **kwargs
):
    """Simply calls the OpenAI Chat completion API with the given parameters. Asynchronous version."""
    params = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    )
    
    return await client.chat.completions.create(**params)


class BaseChatModule(BaseModule):
    def __init__(
        self, 
        api_key_env_name: str,
        model_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop=None, 
        timeout: int = 60,
        max_retries: int = 3,
        seed: Optional[int] = None,
        tools: Optional[List[dict[str, str]]] = None,
        tool_choice: str = "auto",
        response_format: Optional[dict[str, str]] = None,
    ) -> None:
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed
        self.response_format = response_format
        self.tools = tools
        self.tool_choice = tool_choice
        
    def run(self, messages: list[dict[str, str]], enable_functioncall: bool = True) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")
        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")
        
        client = get_client(
            api_key_env_name=self.api_key_env_name,
            max_retries=self.max_retries,
            time_out=self.timeout,
        )
        
        response = completion(
            client,
            self.model_name,
            messages,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.frequency_penalty,
            self.presence_penalty,
            False,
            stop=self.stop,
            tools=self.tools,
            tool_choice=self.tool_choice if enable_functioncall else "none",
        )
        # logger.debug(f"Response: {response}")
        
        usage = Usage()
        usage += response.usage  # Update usage
        response_message = response.choices[0].message.content.strip("\n")
        logger.info("response_message: {}".format(response_message))
        return CompletionResults(
            usage=usage,
            message={"role": "assistant", "content": response_message},
            prompt=deepcopy(messages),
            tool_call=response.choices[0].message.tool_calls,
        )
        
    async def arun(self, messages: list[dict[str, str]], enable_functioncall: bool = True) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")
        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")
        
        client = get_async_client(
            api_key_env_name=self.api_key_env_name,
            max_retries=self.max_retries,
            time_out=self.timeout,
        )
        
        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            stream=False,
            tools=self.tools,
            tool_choice=self.tool_choice if enable_functioncall else "none",
        )
        
        usage = Usage()
        usage += response.usage  # Update usage
        response_message = response.choices[0].message.content.strip("\n")
        logger.info("response_message: {}".format(response_message))
        if response.choices[0].message.tool_calls:
            content = ''
        else:
            content = response_message
            
        return CompletionResults(
            usage=usage,
            message={"role": "assistant", "content": content},
            prompt=deepcopy(messages),
            tool_call=response.choices[0].message.tool_calls,
        )
    
    def stream(self, messages: list[dict[str, str]]) -> CompletionResults:
        
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")
        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")
        
        client = get_client(
            api_key_env_name=self.api_key_env_name,
            max_retries=self.max_retries,
            time_out=self.timeout,
        )
        
        response = completion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            stream=True,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )     
        
        response_message = {"role": "assistant", "content": ""}
        for r in response:
            if len(r.choices) > 0:
                delta = r.choices[0].delta
                if delta is not None:
                    chunk = delta.content
                    if chunk is not None and chunk != "":
                        response_message["content"] += chunk
                        yield chunk
        
        usage = Usage(
            prompt_tokens=sum([get_n_tokens(m, self.model_name)["total"] for m in messages]),
            completion_tokens=get_n_tokens(response_message, self.model_name)["total"],
        )
        logger.info("response_message: {}".format(response_message))
        
        if r.choices[0].delta.tool_calls:
            response_message['content'] = ''
        else:
            response_message['content'] = response_message
        yield CompletionResults(
            usage=usage,
            message=response_message,
            prompt=deepcopy(messages),
            tool_call=response.choices[0].message.tool_calls,
        )
        
    async def astream(self, messages: list[dict[str, str]]) -> CompletionResults:
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")
        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")   

        client = get_async_client(
            api_key_env_name=self.api_key_env_name,
            max_retries=self.max_retries,
            time_out=self.timeout,
        )
        
        response = await acompletion(
            client=client,
            model_name=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            stream=True,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )
        
        response_message = {"role": "assistant", "content": ""}
        tool_call_response = ""
        tool_call_dict = {}
        function_name = ""
        call_id = None
        
        async for r in response:
            if r.choices[0].delta.tool_calls:
                # logger.debug(f"Tool calls: {r.choices[0].delta.tool_calls}")
                function_name = r.choices[0].delta.tool_calls[0].function.name if function_name == "" else function_name
                call_id = r.choices[0].delta.tool_calls[0].id if call_id is None else call_id
                tool_call_response += r.choices[0].delta.tool_calls[0].function.arguments
            elif len(r.choices) > 0:
                delta = r.choices[0].delta
                if delta is not None:
                    chunk = delta.content
                    if chunk is not None and chunk != "":
                        response_message["content"] += chunk
                        yield chunk
        
        logger.debug(f"tool_call_response: {tool_call_response}")           
        # logger.debug(f"messages: {messages}")
        # logger.debug(f"response_message: {response_message}")
        usage = Usage(
            prompt_tokens=sum([get_n_tokens(m, self.model_name)["total"] for m in messages]),
            completion_tokens=get_n_tokens(response_message, self.model_name)["total"],
        )
        # logger.info("response_message: {}".format(response_message))
        
        if tool_call_response != "":
            tool_call_response = json.loads(tool_call_response) if tool_call_response else None
            tool_call_dict["args"] = tool_call_response
            tool_call_dict["name"] = function_name
            tool_call_dict["id"] = call_id
        
        if r.choices[0].delta.tool_calls:
            response_message = tool_call_dict
            response_message.role = "assistant"
            # logger.debug(f"Response message: {response_message}")
            
        yield CompletionResults(
            usage=usage,
            message=response_message,
            prompt=deepcopy(messages),
            tool_call=tool_call_dict,
        )
        
class OpenAIChatModule(BaseModule):
    def __init__(
        self, 
        api_key_env_name: str,
        model_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop=None, 
        timeout: int = 60,
        max_retries: int = 3,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
        context_length: Optional[int] = None,
        conversation_memory: Optional[BaseConversationMemory] = None,
        content_filter: Optional[BaseFilter] = None,
        conversation_length_adjuster: Optional[BaseConversationLengthAdjuster] = None,
        tool_choice: str = "auto",
        callable_functions: Optional[List[CallableFunction]] = None,
    ) -> None:
        
        self.api_key_env_name = api_key_env_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed
        self.response_format = response_format
        self.context_length = context_length
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter
        self.conversation_length_adjuster = conversation_length_adjuster
        self.tool_choice = tool_choice
        self.collable_functions = callable_functions
        
        # Function calling用の設定
        self.tools = [collable_func.description for collable_func in callable_functions]
        self.task_conversetion_memory = ConversationMemory(
            save_path = "task_cache_memory.txt",
            memory_length = 2048,
            compress_model = self,
            prompt = TASK_COMPRESS_PROMPT
        )
        self.task_conversation_mode = False
        
        token_limit = get_token_limit(self.model_name)
        max_tokens = max_tokens if max_tokens else int(token_limit / 2)
        
        context_length = token_limit - max_tokens if context_length is None else context_length
        
        assert (
            token_limit >= max_tokens + context_length
        ), f"max_tokens + context_length must be less than or equal to {token_limit}."
        
        
        self.chat_model = BaseChatModule(
            api_key_env_name=self.api_key_env_name,
            model_name=self.model_name,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            timeout=self.timeout,
            max_retries=self.max_retries,
            seed=self.seed,
            response_format=self.response_format,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )
        
        self.conversation_length_adjuster = (
            OldConversationTruncationModule(model_name=model_name, context_length=context_length)
            if conversation_length_adjuster is None
            else conversation_length_adjuster
        )
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter
        
    def function_call(self, tool_call: Dict[str, str], 
                      messages: list = [],
                      second_response: bool = True,
                      response_mode: str = "run"
                      ) -> list[FunctionCallingResults]:
        """Call functions based on tool_calls."""
        
        logger.debug(f"Tool calls: {tool_call}")
        logger.debug("type(tool_calls): {}".format(type(tool_call)))
            
        func_name = tool_call["name"]
        collable_func_names = [collable_func.name for collable_func in self.collable_functions]
        
        assert func_name in collable_func_names, f"Called function name: {func_name} is not in callable functions."
        for callable_func in self.collable_functions:
            if func_name == callable_func.name:
                logger.debug(f"Callable func: {callable_func}")
                func = callable_func.function
                func_args = tool_call["args"]
                function_response = func(**func_args)
                logger.debug(f"Function response: {function_response}")
                break
            
            messages.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": function_response
                }
            )
        if second_response:
            logger.debug(f"Generating Second response...  Mode: {response_mode}")
            logger.debug(f"Messages: {messages[-5:]}")
            if response_mode == "run":
                return self.chat_model.run(
                    messages=messages,
                    enable_functioncall=False,
                )
        else:
            result = CompletionResults(
                message={
                    "role": "assistant",
                    "content": function_response,
                },
                usage=Usage(),
                prompt=deepcopy(messages),
            )
            return result
                
    def run(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []
        
        if isinstance(init_conversation, list) and len(messages) == 0:
            messages = init_conversation + messages
        
        messages.append(
            Message(
                content=prompt,
            ).as_user
        )
        
        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)
        
        messages = self.conversation_length_adjuster(messages)    
        response = self.chat_model(messages)
        
        if response.tool_calls:
            tool_calls = response.tool_calls
            logger.debug(f"Tool calls: {tool_calls}")
            messages.append(response.message)
            response = self.function_call(tool_calls, messages)
        
        if self.content_filter is not None:
            response.messages = self.content_filter.restore([response.message])[0]
        
        messages.append(response.message)
        
        if self.conversation_memory is not None:
            self.conversation_memory.store(messages)

        return response

    def stream(
        self, 
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []
        
        if isinstance(init_conversation, list) and len(messages) == 0:
            messages = init_conversation + messages
        
        messages.append(
            Message(
                content=prompt,
            ).as_user
        )
        
        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)
        
        messages = self.conversation_length_adjuster(messages)
        logger.debug(f"Adujsted messages: {messages}")
            
        response = self.chat_model.stream(messages)
        
        response_message_stream = {"role": "assistant", "content": ""}
        for chunk in response:
            if isinstance(chunk, str):
                response_message_stream["content"] += chunk
                
                if self.content_filter is not None:
                    response_message_stream = self.content_filter.restore(
                        [response_message_stream]
                    )[0]  
                yield response_message_stream["content"]
                
            elif isinstance(chunk, CompletionResults):  # Finish streaming
                if self.content_filter is not None:
                    chunk.message = self.content_filter.restore([chunk.message])[0]

                if chunk.tool_call:
                    logger.debug(f"Tool calls: {chunk.tool_call}")
                    chunk = self.function_call(chunk.tool_call)
                    
                messages.append(chunk.message)
                
                if self.conversation_memory is not None:
                    self.conversation_memory.store(messages)
                    
                yield chunk
            else:
                raise AssertionError("Invalid type of chunk.")
    
    async def astream(
      self,
      prompt: str,
      init_conversation: Optional[list[dict[str, str]]] = None,  
    ):
        
        # Load Conversation Memory if exists
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []
        if self.task_conversation_mode:
            messages = self.task_conversetion_memory.load()
        else:
            messages = []
        
        if isinstance(init_conversation, list) and len(messages) == 0:
            messages = init_conversation + messages
        
        messages.append(
            Message(
                content=prompt,
            ).as_user
        )
        
        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)
            
        messages = self.conversation_length_adjuster(messages)
        response = self.chat_model.astream(messages)
        
        response_message_stream = {"role": "assistant", "content": ""}
        async for chunk in response:
            if isinstance(chunk, str):
                response_message_stream["content"] += chunk
                
                if self.content_filter is not None:
                    response_message_stream = self.content_filter.restore(
                        [response_message_stream]
                    )[0]  
                yield response_message_stream["content"]
                
            elif isinstance(chunk, CompletionResults): # overall response
                if chunk.tool_call:
                    tool_call = chunk.tool_call
                    logger.debug(f"Tool calls: {tool_call}")
                    logger.debug(f"GPT Messages: {chunk.message}")
                    messages.append(chunk.message)
                    chunk = self.function_call(tool_call, messages, second_response=False)
                    
                if self.content_filter is not None:
                    chunk.message = self.content_filter.restore([chunk.message])[0]
                
                messages.append(chunk.message)
                
                # Update conversation memory
                if self.conversation_memory is not None:
                    self.conversation_memory.store(messages)
                if self.task_conversation_mode:
                    self.task_conversetion_memory.store(messages)
                
                yield chunk
            else:
                raise AssertionError("Invalid type of chunk.")
    
    async def arun(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
    ) -> CompletionResults:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []
        
        if isinstance(init_conversation, list) and len(messages) == 0:
            messages = init_conversation + messages
        
        messages.append(
            Message(
                content=prompt,
            ).as_user
        )
        
        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)
        
        messages = self.conversation_length_adjuster(messages)    
        response = await self.chat_model.arun(messages)
        
        if response.tool_call:
            tool_calls = response.tool_call
            logger.debug(f"Tool calls: {tool_calls}")
            for tool_call in tool_calls:
                function_call_result = self.function_call(tool_call)
        
        if self.content_filter is not None:
            response.messages = self.content_filter.restore([response.message])[0]
        
        messages.append(response.message)
        
        if self.conversation_memory is not None:
            self.conversation_memory.store(messages)

        return response
    
    
    async def abatch_run(
        self,
        prompts: list[str],
        init_conversation: Optional[list[dict[str, str]]] = None,
        batch_size: int = 4,
    ) -> list[CompletionResults]:
        if init_conversation is None:
            init_conversation = [None] * len(prompts)
        
        assert (
            len(prompts) == len(init_conversation)
        ), "prompts and init_conversation must have the same length."
        
        z = zip(prompts, init_conversation)
        batches = make_batch(list(z), batch_size)
        results = []
        for batch in batches:
            async_process = [
                self.arun(
                    prompt=prompt,
                    init_conversation=init_conversation,
                ) for prompt, init_conversation in batch
            ]
            results.extend(await asyncio.gather(*async_process))
        return results