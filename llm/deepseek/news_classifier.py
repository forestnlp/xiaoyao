#!/usr/bin/env python
# coding: utf-8

"""
News sentiment classification (bullish/bearish/neutral) via DeepSeek Chat API.
Reads API key from key.conf and returns structured JSON.

Usage:
  python llm/deepseek/news_classifier.py --text "公司发布业绩预增公告，净利润增长80%"
  python llm/deepseek/news_classifier.py --text "监管加码、公司高管被调查" --model deepseek-chat
"""

import argparse
import json
import os
import re
from pathlib import Path
import requests


BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"  # 非思考模式；可改为 deepseek-reasoner
KEY_PATHS = [
    Path(r"D:\workspace\xiaoyao\localllm\deepseek\key.conf"),
    Path(r"D:\workspace\xiaoyao\llm\deepseek\key.conf"),
]


def read_api_key() -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key.strip()
    for p in KEY_PATHS:
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            for line in txt.splitlines():
                if line.strip().lower().startswith("api_key:"):
                    return line.split(":", 1)[1].strip()
    raise FileNotFoundError("DeepSeek API key not found. Set DEEPSEEK_API_KEY or create key.conf with 'api_key: <KEY>'.")


def call_chat(api_key: str, messages: list, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> dict:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": False,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_json(text: str) -> dict:
    """Try to parse JSON from LLM output; fallback by extracting first {...} block."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"raw": text}


def classify_news(text: str, model: str = DEFAULT_MODEL) -> dict:
    """Classify news into bullish/bearish/neutral with confidence and brief rationale.

    Returns:
        dict with keys: label (bullish|bearish|neutral), confidence (0-1), rationale (str)
    """
    api_key = read_api_key()
    system_prompt = (
        "你是一名证券市场分析助手。请基于给定的中文新闻或公告，判断其对股票市场的影响是利好、利空还是中性。"
        "严格返回 JSON，字段为：label（bullish/bearish/neutral），confidence（0-1），rationale（不超过40字的中文理由）。"
        "注意：若信息不完整或影响不明确，标注为 neutral 并降低 confidence。"
    )
    user_prompt = f"待判定文本：\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    data = call_chat(api_key, messages, model=model)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = extract_json(content)
    # basic normalization
    label = str(parsed.get("label", "neutral")).lower().strip()
    if label not in {"bullish", "bearish", "neutral"}:
        label = "neutral"
    try:
        conf = float(parsed.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    rationale = str(parsed.get("rationale", parsed.get("reason", "")))[:80]
    return {"label": label, "confidence": max(0.0, min(1.0, conf)), "rationale": rationale}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="新闻或公告文本")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="deepseek-chat 或 deepseek-reasoner")
    args = parser.parse_args()

    res = classify_news(args.text, model=args.model)
    print("== Classification ==")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()