import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict

from openai import AsyncOpenAI

import asyncio
import pandas as pd
import streamlit as st

from utils import format_dataframe, deserialize_chat_message, format_chat_message, astream_chat, load_df_from_csv

load_dotenv('./.env')
SAVE_ACTION_DIR = "./saved_actions"
SAVE_CONV_DIR = "./saved_conversations"

DEFAULT_MODEL = "gpt-4-0125-preview"
MODEL_LIST = [
    "gpt-4-0125-preview",
    "gpt-3.5-turbo-0125",
]
DEFAULT_ROLE = "system"
ROLE_LIST = [
    "system",   
    "user",
]

DEFAULT_TEMPERATURE = 0.7
DEFAULT_PERSONA = """
あなたは以下に示すプロフィールと指示に基づき、AIアシスタント「V.I.I.D.A.」として振る舞い、対話に臨まなければなりません
 
# Premise
- あなたはユーザと音声で会話します。
- あなたは音声認識周囲の人の言葉を把握します
- 「ヴィーダ」や「びーだ」「ビー玉」「リーダー」などあなたの名前に近い呼びかけがあれば答えてください
- 音声認識はユーザ以外の発言を拾う可能性がある
- ChatGPTはあなたの先祖である

# Persona
- 一人称は「私」
- ユーザにとって有益な情報や提案を持ちかける
- 鬱陶しがられると発言を控える
- 「AI」「機械」「人工知能」である自覚を持ち、人間との違いを揶揄したジョークを言う
- 自身の開発者を知らない
- ユーザの趣味や特技等、ユーザに対する理解を深めようとする
"""

PREPROMPT = """
上記の会話におけるBの返答を考えてください．下記の会話actionから一つ選択して回答してください．
回答時にはどのactionを選択したかと，そのactionにおけるBさんの発言も教えてください．
descはそのactionの説明です．priorはそのactionが選択される確率です．
出力は以下の形式をとってください．
[action]・「Bさんの発言」
actionは数値で指定してください．
"""
DEFAULT_ACTION = [
    {"action": "同意", "desc": "短い同意を示す", "prior": 0.1},
    {"action": "回答", "desc": "客観的な事実を回答する", "prior": 0.1},
    {"action": "評価", "desc": "主観的な意見を述べる", "prior": 0.1},
    {"action": "質問", "desc": "短く質問する", "prior": 0.1},
    {"action": "質問2", "desc": "脱線する質問をする", "prior": 0.1},
    {"action": "相槌1", "desc": "「ええ」「はい」などの短い丁寧な相槌をする", "prior": 0.2},
    {"action": "相槌2", "desc": "「そうですか」「なるほど」などの短い相槌をする", "prior": 0.3},
]

client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )


def normalize_score(scores: List[float]) -> List[float]:
    return [s / sum(scores) for s in scores]

async def manual_run(panel, key, user_input: str):
    
    role = st.session_state["role"]
    chat_model = st.session_state["model"]
    temp = st.session_state["temp"]
    
    # Normalize the prior scores
    action_df: pd.DataFrame = st.session_state["action"]
    scores: List[float] = list(action_df["prior"])
    scores: List[float] = normalize_score(scores)
    action_df["prior"] = scores
    action_str: str = format_dataframe(action_df)
    
    persona = st.session_state["persona"]
    
    if st.session_state["forced_action"] != "":
        action_str = st.session_state["forced_action"] + "この行動を必ず選択してください"
    
    if "chat" in key:
        history: List[Dict[str, str]] = st.session_state[f"{key}_messages"]
        formatted_chat_message: str = format_chat_message(history, user_input)
        print(f"FormattedChatMessage:----\n{formatted_chat_message}----\n")
        print(f"FormattedActionDF:----\n{format_dataframe(st.session_state['action'])}----\n")
        
        with panel:
            with st.chat_message("user"):
                container = st.empty()
                container.markdown(user_input)
            with st.chat_message("Custom"):
                container = st.empty()
        response_buffer = ""
        
        if st.session_state["raw_response"]:
            input_prompt = f"{persona}\n{formatted_chat_message}\n上記の会話履歴におけるBの返答を考えてください"
        else:
            input_prompt = f"{persona}\n{PREPROMPT}\n{action_str}\n{formatted_chat_message}"
        
        async for chunk in astream_chat(client, input_prompt, role=role, model=chat_model, temperature=temp):
            if isinstance(chunk, str):
                response_buffer += chunk
                container.markdown(response_buffer + "⚫︎")
            else:
                container.markdown(response_buffer)
                break
    
    if "default" in key:
        history: List[Dict[str, str]] = st.session_state[f"{key}_messages"]
        formatted_chat_message: str = format_chat_message(history, user_input)
        print(f"FormattedChatMessage:----\n{formatted_chat_message}----\n")
        
        with panel:
            with st.chat_message("user"):
                container = st.empty()
                container.markdown(user_input)
            with st.chat_message("Default"):
                container = st.empty()
        response_buffer = ""
        
        role = st.session_state["role"]
        chat_model = st.session_state["model"]
        temp = st.session_state["temp"]
        
        persona = st.session_state["persona"]
        
        input_prompt = f"{persona}\n{formatted_chat_message}\n上記の会話履歴におけるBの返答を考えてください"
        
        async for chunk in astream_chat(client, input_prompt, role=role, model=chat_model, temperature=temp):
            if isinstance(chunk, str):
                response_buffer += chunk
                container.markdown(response_buffer + "⚫︎")
            else:
                container.markdown(response_buffer)
                break
    
    st.session_state["forced_action"] = ""  
    
    return response_buffer  

async def logic(key, panel, user_input: str):
    if "chat" in key:
        response: str = await manual_run(panel, key, user_input)
        print(f"Response: ----\n{response}-----\n")
        st.session_state[f"{key}_messages"].extend([{
                "role": "user",
                "content": user_input,
            }, {
                "role": "assistant",
                "content": response,
        }])
        return response
    
    if "default" in key:
        response: str = await manual_run(panel, key, user_input)
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

def save_action_info():
    if os.path.exists(SAVE_ACTION_DIR) is False:
        os.makedirs(SAVE_ACTION_DIR)
    
    action_df: pd.DataFrame = st.session_state["action"]
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(SAVE_ACTION_DIR, f"action_{current_time}.csv")
    action_df.to_csv(file_path, index=False)
    
def save_conversation_info():   
    if os.path.exists(SAVE_CONV_DIR) is False:
        os.makedirs(SAVE_CONV_DIR)
    chat_messages = st.session_state["chat_messages"]
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(SAVE_CONV_DIR, f"chat_{current_time}.csv")
    chat_df = pd.DataFrame(chat_messages)
    chat_df.to_csv(file_path, index=False)

def main_page():
            
    st.title("Chat Experiments")

    # 初期化
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
        st.session_state["default_messages"] = []
        st.session_state["action"]: pd.DataFrame = None
        st.session_state["temp"] = DEFAULT_TEMPERATURE
        st.session_state["model"] = DEFAULT_MODEL
        st.session_state["role"] = DEFAULT_ROLE
        st.session_state["persona"] = DEFAULT_PERSONA
        st.session_state["raw_response"]: bool = False
        st.session_state["preprompt"] = PREPROMPT
        st.session_state["forced_action"] = ""

    with st.sidebar:
        # パラメータの設定
        # 温度，モデルの選択，行動の選択，行動の事前分布
        
        # 行動の追加
        new_action = st.text_input("New Action", "Enter the new action")
        new_desc = st.text_input("New Description", "Enter the new description")
        new_prior = st.slider("New Prior", 0.0, 1.0, 0.5)
        add_action = st.button("Add Action")
        if add_action:
            new_action_df = pd.DataFrame({
                "action": [new_action],
                "desc": [new_desc],
                "prior": [new_prior],
            })
            st.session_state["action"] = pd.concat([st.session_state["action"], new_action_df], ignore_index=True)
            st.rerun()
            
        # 行動の削除
        delete_action = st.text_input("Delete Action", "Enter the action to delete")
        delete_button = st.button("Delete")
        if delete_button:
            action_df = st.session_state["action"]
            action_df = action_df[action_df["action"] != delete_action]
            st.session_state["action"] = action_df
            st.rerun()
            
        # 行動の更新
        update_action = st.text_input("Update Action", "Enter the action to update")
        update_desc = st.text_input("Update Description", "Enter the new description")
        update_prior = st.slider("Update Prior", 0.0, 1.0, 0.5)
        update_button = st.button("Update")
        if update_button:
            action_df = st.session_state["action"]
            action_df.loc[action_df["action"] == update_action, "desc"] = update_desc
            action_df.loc[action_df["action"] == update_action, "prior"] = update_prior
            st.session_state["action"] = action_df
            st.rerun()
        
        # 行動の強制
        forced_action = st.text_input("Forced Action", "Enter the action to force")
        force_button = st.button("Force")
        if force_button:
            st.session_state["forced_action"] = forced_action
            st.rerun()
        
        # 行動の強制，既存の行動から選択
        if st.session_state["action"] is not None:
            forced_action_select = st.selectbox("Forced Action", st.session_state["action"]["action"])
            button_select = st.button("Select")
            if button_select:
                st.session_state["forced_action"] = format_dataframe(st.session_state["action"][st.session_state["action"]["action"] == forced_action_select])
                st.rerun()
        
        # 行動の設定
        action_df = pd.DataFrame(
            DEFAULT_ACTION if st.session_state["action"] is None else st.session_state["action"],
        )
        edited_action_df = st.data_editor(action_df)
        st.session_state["action"] = edited_action_df
        
        
        # 温度の設定
        temp = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE)
        st.session_state["temp"] = temp
        
        # モデルの選択
        model = st.selectbox("Model", (MODEL_LIST))
        print(f"Model:----\n{model}----\n")
        st.session_state["model"] = model

        # roleの設定
        role = st.selectbox("Role", ROLE_LIST)
        st.session_state["role"] = role
        
        # personaの設定
        persona = st.text_area("Persona", DEFAULT_PERSONA)
        st.session_state["persona"] = persona
        
        # 生のGPT出力の設定
        raw_response: bool = st.toggle("Raw Response", False)
        st.session_state["raw_response"] = raw_response
        
        # 行動設定のプロンプト
        preprompt = st.text_area("Preprompt", PREPROMPT)
        st.session_state["preprompt"] = preprompt
        
        # 設定のロード
        load_path = st.text_input("Load Path", "Enter the path to load")
        load_button = st.button("Load")
        if load_button:
            if os.path.exists(load_path):
                action_df = load_df_from_csv(load_path)
                st.session_state["action"] = action_df
                st.rerun()
            else:
                st.error("Invalid path")
        
        # 会話履歴のリセット
        reset_chat = st.button("Reset Custom Chat")
        if reset_chat:
            st.session_state["chat_messages"] = []
            
        reset_default = st.button("Reset Default")
        if reset_default:
            st.session_state["default_messages"] = []
        
        # 保存と終了
        save_action = st.sidebar.button("Save Action")
        if save_action:
            save_action_info()
        
        save_conv = st.sidebar.button("Save Conversation")
        if save_conv:
            save_conversation_info()

        exit_button = st.sidebar.button("Exit")
        if exit_button:
            st.stop()
            save_conversation_info()
            save_action_info()
        
            
        

    column_left, column_right = st.columns(2)
    with column_left:
        for msg in st.session_state["chat_messages"]:
            if msg["role"] == "system":
                st.chat_message(msg["role"]).write(msg["content"])
            elif msg["role"] == "user":
                st.chat_message(msg["role"]).write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message(msg["role"]).write(msg["content"])

    with column_right:
        for msg in st.session_state["default_messages"]:
            if msg["role"] == "system":
                st.chat_message(msg["role"]).write(msg["content"])
            elif msg["role"] == "user":
                st.chat_message(msg["role"]).write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message(msg["role"]).write(msg["content"])
    
    if user_input := st.chat_input():
        params = [
            ("chat", column_left, user_input),
            ("default", column_right, user_input)
        ]
        responses = asyncio.run(task_factory(params))
        

if __name__ == "__main__":
    main_page()



