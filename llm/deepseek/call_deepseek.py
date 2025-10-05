#!/usr/bin/env python
# coding: utf-8

"""
Simple DeepSeek API client.
Reads API key from key.conf (format: 'api_key: <KEY>') and calls chat API.

Usage:
  python localllm/deepseek/call_deepseek.py --prompt "你好，给我一个示例" [--model deepseek-chat] [--stream false]

Notes:
  - 'deepseek-chat' 对应 DeepSeek-V3.2-Exp 的非思考模式。
  - 'deepseek-reasoner' 对应 DeepSeek-V3.2-Exp 的思考模式。
  - 默认非流式输出。
"""

import argparse
import json
from pathlib import Path
import sys
import requests


BASE_URL = "https://api.deepseek.com/v1"
KEY_CONF = Path(r"D:\workspace\xiaoyao\llm\deepseek\key.conf")


def read_api_key(conf_path: Path) -> str:
    if not conf_path.exists():
        raise FileNotFoundError(f"key.conf not found: {conf_path}")
    txt = conf_path.read_text(encoding="utf-8").strip()
    for line in txt.splitlines():
        if line.strip().lower().startswith("api_key:"):
            return line.split(":", 1)[1].strip()
    raise ValueError("api_key not found in key.conf")


def call_chat(api_key: str, messages: list, model: str = "deepseek-chat", stream: bool = False):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="用户输入")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="模型：deepseek-chat 或 deepseek-reasoner")
    parser.add_argument("--stream", type=str, default="false", help="是否流式输出：true/false")
    args = parser.parse_args()

    api_key = read_api_key(KEY_CONF)
    stream_flag = args.stream.strip().lower() == "true"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    try:
        data = call_chat(api_key, messages, model=args.model, stream=stream_flag)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        if e.response is not None:
            print(e.response.text)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 非流式：打印首条回复内容
    try:
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        print("== DeepSeek Response ==")
        print(content)
    except Exception:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()