import pandas as pd
from typing import List, Dict

def load_df_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"---LoadCSV---\n{df}\n-----\n")
    return df

def format_dataframe(df: pd.DataFrame) -> str:
    print(f"---ToMarkdown---\n{df.to_markdown()}\n-----\n")
    return df.to_markdown()

def deserialize_chat_message(messages: str) -> List[Dict[str, str]]:
    messages = []
    for m in messages.split("\n"):
        role, content = m.split("「")
        content = content.replace("」", "")
        messages.append({"role": role, "content": content}) 
    return messages

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

async def astream(client, prompt: str, role: str, model: str, temperature: float):
    print(f"Stream request:-----\n{prompt}----\n")
    messages = [
        {"role": role, "content": prompt},
    ]
    params = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    return await client.chat.completions.create(**params)

async def astream_chat(client, prompt: str, role: str, model: str, temperature: float):
    response = await astream(client, prompt, role, model, temperature)
    response_message = {"role": "assistant", "content": ""}
    async for r in response:
        if len(r.choices) > 0:
            delta = r.choices[0].delta
            if delta is not None:
                chunk = delta.content
                if chunk is not None:
                    response_message["content"] += chunk
                    yield chunk