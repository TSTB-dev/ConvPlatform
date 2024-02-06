import os
from dotenv import load_dotenv
from typing import List, Dict

from openai import AsyncOpenAI

import asyncio
import streamlit as st

load_dotenv('./.env')
CHAT_MODEL = "gpt-4-1106-preview"
TEMPERATURE = 0.7
ROLE = "system"
PREPROMPT = """

# [同意] BさんはこのあとAさんに短い同意を示します
# [回答] 相手の質問に対して客観的な事実を回答します
# [評価] 自分の主観的な意見を述べます
# [質問] BさんはこのあとAさんに短く質問します．
# [質問2] BさんはこのあとAさんに軽く脱線する質問します．
# [脱線1] Bさんはこの後この話題に関連して軽く脱線します．
# [脱線2] Bさんはこの後この話題に関連してAさんに軽く質問します
# [未知] Bさんはこの後，知識不足を感じてAさんに質問します．
# [既知] Bさんはこの後，既知の情報であることを示します．
# [相槌1] 「ええ」「はい」のいずれかを返します 
# [相槌2] 「なるほど」「そうなんですか」のいずれかを返します

上記の行動から一つ選択して回答してください．回答時にはどの行動を選択したかと，その行動におけるBさんの発言も教えてください．
選択時の事前分布
同意: 0.2
回答: 0.2
評価: 0.1
質問: 0.2
質問2: 0.1
脱線1: 0.3
脱線2: 0.1
未知: 0.01
既知: 0.02
相槌1: 0.4

上記の会話におけるBの返答を考えてください．回答時にはどの行動を選択したかと，その行動におけるBさんの発言も教えてください．
出力は以下の形式をとってください．
[行動]・「Bさんの発言」
"""

client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )


async def astream(client, prompt: str, role: str):
    print(f"Stream request:-----\n{prompt}----\n")
    messages = [
        {"role": role, "content": prompt},
    ]
    params = dict(
        model=CHAT_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True,
    )
    return await client.chat.completions.create(**params)

async def astream_chat(client, prompt: str, role: str):
    response = await astream(client, prompt, role)
    response_message = {"role": "assistant", "content": ""}
    async for r in response:
        if len(r.choices) > 0:
            delta = r.choices[0].delta
            if delta is not None:
                chunk = delta.content
                if chunk is not None:
                    response_message["content"] += chunk
                    yield chunk

def format_chat_message(messages: List[Dict[str, str]], user_input: str) -> str:
    formatted_messages = ""
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant":
            name = "B"
            formatted_messages += f"{name}{m['content']}\n"
        if m["role"] == "user":
            name = "A"
            formatted_messages += f"{name}「{m['content']}」\n"
            
    formatted_messages += f"A「{user_input}」"
    print(f"FormattedMessages----\n{format_chat_message}-----\n")
    return formatted_messages

def deserialize_chat_message(messages: str) -> List[Dict[str, str]]:
    messages = []
    for m in messages.split("\n"):
        role, content = m.split("「")
        content = content.replace("」", "")
        messages.append({"role": role, "content": content}) 
    return messages

async def manual_run(key, user_input: str):
    
    if "chat" in key:
        history: List[Dict[str, str]] = st.session_state[f"{key}_messages"]
        formatted_chat_message: str = format_chat_message(history, user_input)
        input_prompt = f"{PREPROMPT}\n{formatted_chat_message}"
        
        with st.chat_message("user"):
            container = st.empty()
            container.markdown(user_input)
        with st.chat_message("V.I.I.D.A."):
            container = st.empty()
        response_buffer = ""
        
        async for chunk in astream_chat(client, input_prompt, role=ROLE):
            if isinstance(chunk, str):
                response_buffer += chunk
                container.markdown(response_buffer + "⚫︎")
            else:
                container.markdown(response_buffer)
                break
    return response_buffer  

async def logic(key, user_input: str):
    if "chat" in key:
        response: str = await manual_run(key, user_input)
        print(f"Response: ----\n{response}-----\n")
        st.session_state[f"{key}_messages"].extend([{
                "role": "user",
                "content": user_input,
            }, {
                "role": "assistant",
                "content": response,
        }])
        return response
        
async def task_factory(params):
    tasks = []
    for param in params:
        tasks.append(asyncio.create_task(logic(*param)))
    return await asyncio.gather(*tasks)

def main_page():
            
    st.title("ChatBot")
    
    with st.sidebar:
        exit_app = st.sidebar.button("Shut Down")
        if exit_app:
            # TODO: Save chat history & setting
            
            # compress_history(LOG_DIR, st.session_state["chat_messages"])
            # save_profile_info(PROFILE_DIR, LOG_DIR)
            # create_index(log_dir=LOG_DIR, storage_dir=STORAGE_DIR)
            st.stop()

    # 初期化
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "system":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input():
        params = [
            ("chat", user_input),
        ]
        responses = asyncio.run(task_factory(params))
        

if __name__ == "__main__":
    main_page()



