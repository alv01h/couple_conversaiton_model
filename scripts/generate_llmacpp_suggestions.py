#!/usr/bin/env python3
"""使用 llama-cpp-python 載入 GGUF 模型進行推論，並在計算完畢後立即釋放 RAM/VRAM 記憶體。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def find_gguf_model(user_path: str | None = None) -> Path | None:
    """尋找本機可用的 GGUF 模型檔案"""
    if user_path and Path(user_path).exists():
        return Path(user_path)

    # 搜尋預設模型目錄與 Ollama blobs 快取
    possible_paths = [
        Path("models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        Path("models/model.gguf"),
    ]

    ollama_blobs = Path.home() / ".ollama" / "models" / "blobs"
    if ollama_blobs.exists():
        for blob in ollama_blobs.glob("sha256-*"):
            # GGUF 檔案通常大於 200MB
            if blob.stat().st_size > 200 * 1024 * 1024:
                possible_paths.append(blob)

    for p in possible_paths:
        if p.exists():
            return p

    return None


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """建立符合 llama-cpp-python chat completion 格式的對話 Prompt"""
    history_str = "\n".join(item.get("history", [])) if item.get("history") else "(無)"
    
    system_msg = (
        "你是一位專業的人際關係與非暴力溝通 (NVC) 相處教練。\n"
        "請針對使用者發送的特定對話訊息進行分析，並給出專屬、具體的溫和改寫建議。\n"
        "請務必使用【繁體中文】回答，且直接輸出純 JSON 格式（不要包含 MarkDown 格式說明或無關字詞）。"
    )

    user_msg = f"""請針對這句對話訊息進行分析，並輸出【專屬修復改寫對白】：

【背景情境】{item.get('scenario', '')}
【對話前文】
{history_str}

【當前發送的原句】「{item.get('current_message', '')}」
【AI 風險預測】分數: {item.get('predicted_risk_score_1_4', 0.0)} 分 ({item.get('risk_level', '').upper()})

---
請務必依據「原句」真實含義，直接替換寫出 3 種實際機能講出口的溫和修復句子：

請精準輸出以下 JSON 結構：
{{
  "analysis": "簡短分析此句背後隱含的情緒與需求 (1-2句)",
  "suggestions": {{
    "empathic": "（表達同理心與感受的實際對白）",
    "softened": "（軟化語氣、直白表達需求的實際對白）",
    "solution_oriented": "（提出雙方共同解決方案的實際對白）"
  }}
}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_json_from_llm(response_text: str) -> Dict[str, Any]:
    """解析 LLM 輸出的 JSON 文字"""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    result = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            try:
                result = json.loads(text[start_idx : end_idx + 1])
            except Exception:
                pass

    if not isinstance(result, dict):
        return {
            "analysis": "LLM 輸出非標準 JSON 格式",
            "raw_response": response_text,
            "suggestions": {
                "empathic": response_text,
                "softened": response_text,
                "solution_oriented": response_text,
            },
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="使用 llama-cpp-python 進行推論並在算完後即時釋放記憶體")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/sample_predictions.json",
        help="輸入之預測結果 JSON 檔案",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/sample_suggestions.json",
        help="輸出之含修復建議 JSON 檔案",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="GGUF 模型檔案路徑 (.gguf)",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="載入至 GPU/Metal 的圖層數 (-1 表示全卸載至 Metal GPU)",
    )
    args = parser.parse_args()

    if Llama is None:
        print("❌ [錯誤] 未安裝 llama-cpp-python 套件！請先執行: python3 -m pip install llama-cpp-python")
        sys.exit(1)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ [錯誤] 找不到輸入檔案 {input_path}！請先執行 python3 scripts/predict_risk.py。")
        sys.exit(1)

    model_path = find_gguf_model(args.model_path)
    if not model_path:
        print("❌ [錯誤] 找不到可用的 .gguf 模型檔！請指定 --model_path /path/to/model.gguf")
        sys.exit(1)

    print(f"📖 [資料讀取] 載入 {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🚀 [模型載入] 使用 llama-cpp-python 載入 GGUF 模型: {model_path}")
    print(f"⚡ [GPU 加速] Metal/GPU 卸載層數: {args.n_gpu_layers} ...")

    # 1. 初始化並載入 Llama 模型至記憶體 / Metal 顯存
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=2048,
        verbose=False,
    )

    updated_data = []
    try:
        for item in data:
            print(f"\n💬 處理對話 [{item['id']}] - 當前訊息: 「{item['current_message']}」")
            messages = build_messages(item)

            # 進行推論
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )

            raw_text = response["choices"][0]["message"]["content"]
            parsed_json = parse_json_from_llm(raw_text)

            item["llm_repair_analysis"] = parsed_json.get("analysis", raw_text)
            item["llm_repair_suggestions"] = parsed_json.get("suggestions", {})
            print("  ✨ 成功生成修復改寫建議！")

            updated_data.append(item)

    finally:
        # 2. 算完後【立即手動刪除模型實例並執行垃圾回收】，釋放 RAM/VRAM！
        print("\n🧹 [記憶體釋放] 刪除 Llama 模型物件並觸發垃圾回收 (Garbage Collection)...")
        del llm
        gc.collect()
        print("✅ [記憶體釋放完成] RAM & Metal VRAM 已成功清空釋放！")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 [流程完成] 成果已寫入 {output_path}")


if __name__ == "__main__":
    main()
