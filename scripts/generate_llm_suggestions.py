#!/usr/bin/env python3
"""讀取 sample_predictions.json 並使用 Ollama (Qwen) 產生衝突分析與非暴力修復建議改寫。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import ollama
except ImportError:
    ollama = None

import urllib.request
import urllib.error


def check_ollama_server(host: str = "http://localhost:11434") -> bool:
    """檢查 Ollama 服務是否正在背景運行"""
    try:
        req = urllib.request.urlopen(f"{host}/api/tags", timeout=3)
        return req.status == 200
    except Exception:
        return False


def build_llm_prompt(item: Dict[str, Any]) -> str:
    """構建傳給 LLM 的最佳提示詞 (Prompt)"""
    history_str = "\n".join(item.get("history", [])) if item.get("history") else "(無)"
    
    prompt = f"""你是一位專業的人際關係與非暴力溝通 (NVC) 相處教練。
請分析以下對話中的衝突訊息，並給出分析與修復建議。

【對話背景】
- 關係類型：{item.get('relationship', 'couple')}
- 前因後果情境：{item.get('scenario', '')}
- 歷史對話紀錄：
{history_str}

【當前發送的訊息】
「{item.get('current_message', '')}」

【AI 模型風險預測】
- 風險分數 (1-4分)：{item.get('predicted_risk_score_1_4', 0.0)} 分
- 風險等級：{item.get('risk_level', 'low').upper()}

---
請以【繁體中文】輸出繁體 JSON 格式（只輸出純 JSON，不要包含 Markdown 註解或無關文字），欄位包含：
1. "analysis": 簡短分析這句話背後隱含的情緒與未被滿足的需求 (約 1-2 句話)。
2. "suggestions": 包含 3 種不同語氣的「溫和改寫建議 (Repair Suggestions)」，幫助溝通順利：
   - "empathic": 展現同理心與表達需求的說法
   - "softened": 軟化語氣、直白表達感覺的說法
   - "solution_oriented": 著重於提出共同解決方案的說法

JSON 範例格式：
{{
  "analysis": "...",
  "suggestions": {{
    "empathic": "...",
    "softened": "...",
    "solution_oriented": "..."
  }}
}}
"""
    return prompt.strip()
def parse_json_from_llm(response_text: str) -> Dict[str, Any]:
    """嘗試從 LLM 輸出文字中解析出 JSON 物件"""
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
            "analysis": "解析 LLM 輸出時遇到格式問題",
            "raw_response": response_text,
            "suggestions": {
                "empathic": response_text,
                "softened": response_text,
                "solution_oriented": response_text,
            },
        }

    # 如果 suggestions 本身是字串且內含 JSON，進行自動解包
    if isinstance(result.get("suggestions"), str):
        s_str = result["suggestions"].strip()
        if s_str.startswith("{") and s_str.endswith("}"):
            try:
                result["suggestions"] = json.loads(s_str)
            except Exception:
                pass

    return result


def main():
    parser = argparse.ArgumentParser(description="呼叫 Ollama LLM (Qwen) 生成對話改寫與修復建議")
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
        help="Ollama 模型名稱 (例如: qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:7b 等)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ [錯誤] 找不到輸入檔案 {input_path}！請先執行 python3 scripts/predict_risk.py 產生預測結果。")
        sys.exit(1)

    print(f"📖 [資料讀取] 載入 {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 檢查 Ollama 服務狀態
    server_online = check_ollama_server()
    if not server_online:
        print("\n⚠️ [警告] 未偵測到正在運行的 Ollama 服務 (http://localhost:11434)！")
        print("💡 請確保您已安裝並啟動 Ollama：")
        print("   1. 啟動 Ollama 應用程式或執行 `ollama serve`")
        print(f"   2. 下載指定模型：`ollama pull {args.model}`")
        print("\n🔧 目前為您展示 LLM Prompt 生成架構與備用 Mock 邏輯...")

    print(f"\n🚀 [開始呼叫 Ollama LLM] 模型: {args.model} ...")

    updated_data = []
    for item in data:
        print(f"\n💬 處理項目 [{item['id']}] - 風險分數: {item['predicted_risk_score_1_4']} ({item['risk_level']})")
        prompt = build_llm_prompt(item)

        if server_online and ollama is not None:
            try:
                print(f"  🤖 發送 Prompt 至 Ollama ({args.model})...")
                response = ollama.generate(model=args.model, prompt=prompt)
                llm_output_text = response.get("response", "")
                parsed_json = parse_json_from_llm(llm_output_text)

                item["llm_repair_analysis"] = parsed_json.get("analysis", "")
                item["llm_repair_suggestions"] = parsed_json.get("suggestions", {})
                print("  ✅ 成功接收 LLM 產出的情緒分析與 3 種溫和改寫建議！")
            except Exception as e:
                print(f"  ⚠️ 呼叫 Ollama 時出錯: {e}")
                item["llm_repair_analysis"] = f"無法呼叫 Ollama ({args.model})"
                item["llm_repair_suggestions"] = {"error": str(e)}
        else:
            # 當 Ollama 服務未開啟時提供示意回應
            item["llm_repair_analysis"] = f"【示範模式】此處將由 Ollama ({args.model}) 分析此處隱含的情緒與需求。"
            item["llm_repair_suggestions"] = {
                "empathic": f"【示範同理說法】當你說「{item['current_message']}」時，我感受到些許壓力，我們可以聊聊嗎？",
                "softened": f"【示範軟化說法】我今天有點累，希望能被理解，而不是指責。",
                "solution_oriented": f"【示範解決方案】我們一起找時間討論如何分配，好嗎？",
            }
            print("  ℹ️ 已完成提示詞構建與示範修復結構備份。")

        updated_data.append(item)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 [流程完成] 包含 LLM 修復建議的 JSON 已儲存至 {output_path}")


if __name__ == "__main__":
    main()
