#!/usr/bin/env python3
"""讀取 sample_predictions.json 並使用 Ollama (Qwen) 產生獨特且專屬的對話修復改寫建議。"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any

try:
    import ollama
except ImportError:
    ollama = None


def check_ollama_server() -> str | None:
    """檢查 Ollama 服務是否正在背景運行，回傳可用的 host URL"""
    hosts = ["http://127.0.0.1:11434", "http://localhost:11434"]
    for host in hosts:
        try:
            req = urllib.request.urlopen(f"{host}/api/tags", timeout=3)
            if req.status == 200:
                return host
        except Exception:
            pass
    return None


def build_system_prompt() -> str:
    return (
        "You are an expert relationship communication and Nonviolent Communication (NVC) coach.\n"
        "Your task is to analyze conflict messages and provide tailored, empathetic, repair-oriented rewrites in English.\n"
        "Output pure valid JSON strictly without markdown formatting or introductory text."
    )


def build_user_prompt(item: Dict[str, Any]) -> str:
    history_str = "\n".join(item.get("history", [])) if item.get("history") else "(None)"
    
    prompt = f"""You are an NVC relationship coach. Write ACTUAL FIRST-PERSON SPOKEN REWRITE QUOTES (e.g. "I feel...", "Can we...") to replace the aggressive message.
IMPORTANT: Do NOT write meta-descriptions like "The author is expressing...". Write the EXACT SPOKEN DIALOGUE QUOTES!

### EXAMPLE INPUT:
Original Message: "You never help with anything around the house!"
### EXAMPLE OUTPUT JSON:
{{
  "analysis": "The speaker feels overwhelmed with chores and needs shared support.",
  "suggestions": {{
    "empathic": "I'm feeling really overwhelmed with household chores right now and could really use your help.",
    "softened": "I get frustrated when I have to clean up alone. Can we talk about sharing these tasks?",
    "solution_oriented": "Let's list the daily chores together and split them so we both have time to relax."
  }}
}}

---

### YOUR TARGET TASK:
[Scenario Context] {item.get('scenario', '')}
[Dialogue History]
{history_str}

[Original Current Message] "{item.get('current_message', '')}"

Now write actual first-person spoken dialogue quotes for the target message above in JSON:
{{
  "analysis": "Short 1-2 sentence emotion & need analysis in English",
  "suggestions": {{
    "empathic": "<Write actual spoken first-person English quote>",
    "softened": "<Write actual spoken first-person English quote>",
    "solution_oriented": "<Write actual spoken first-person English quote>"
  }}
}}"""
    return prompt.strip()


def parse_json_from_llm(response_text: str) -> Dict[str, Any]:
    """從 LLM 輸出文字中精確提取並解析 JSON 物件"""
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

    # 如果 suggestions 本身是字串內含 JSON，進行自動解包
    if isinstance(result.get("suggestions"), str):
        s_str = result["suggestions"].strip()
        if s_str.startswith("{") and s_str.endswith("}"):
            try:
                result["suggestions"] = json.loads(s_str)
            except Exception:
                pass

    return result


def call_ollama_http(host: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """使用 HTTP API 呼叫 Ollama /api/chat 取得高質量專屬回應"""
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        return res_data.get("message", {}).get("content", "")


def main():
    parser = argparse.ArgumentParser(description="呼叫 Ollama LLM 生成專屬對話修復與改寫建議")
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
        "--model",
        type=str,
        default="qwen2.5:0.5b",
        help="Ollama 模型名稱 (例如: qwen2.5:0.5b, qwen2.5:1.5b)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ [錯誤] 找不到輸入檔案 {input_path}！請先執行 python3 scripts/predict_risk.py 產生預測結果。")
        sys.exit(1)

    print(f"📖 [資料讀取] 載入 {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 檢查 Ollama 服務主機
    ollama_host = check_ollama_server()
    if not ollama_host:
        print("\n⚠️ [警告] 未偵測到正在運行的 Ollama 服務 (http://127.0.0.1:11434)！")
        print("💡 請確保您已啟動 Ollama 服務：")
        print("   1. 開啟 Terminal 執行 `ollama serve` (或 brew services start ollama)")
        print(f"   2. 下載模型：`ollama pull {args.model}`")
        sys.exit(1)

    print(f"✅ [服務連接] 成功連線 Ollama ({ollama_host}) | 使用模型: {args.model}")

    updated_data = []
    for item in data:
        print(f"\n💬 處理對話 [{item['id']}] - 當前訊息: 「{item['current_message']}」")
        print(f"   風險分數: {item['predicted_risk_score_1_4']} ({item['risk_level']})")

        sys_prompt = build_system_prompt()
        user_prompt = build_user_prompt(item)

        try:
            print(f"  🤖 發送專屬 Prompt 至 Ollama API...")
            raw_response = call_ollama_http(ollama_host, args.model, sys_prompt, user_prompt)
            parsed_json = parse_json_from_llm(raw_response)

            item["llm_repair_analysis"] = parsed_json.get("analysis", raw_response)
            item["llm_repair_suggestions"] = parsed_json.get("suggestions", {})
            print("  ✨ 成功生成專屬情緒分析與 3 種溫和改寫建議！")
        except Exception as e:
            print(f"  ❌ 呼叫 Ollama 時出錯: {e}")
            item["llm_repair_analysis"] = f"呼叫失敗: {e}"
            item["llm_repair_suggestions"] = {"error": str(e)}

        updated_data.append(item)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 [流程完成] 已將包含專屬修復建議的 JSON 儲存至 {output_path}")


if __name__ == "__main__":
    main()
