#!/usr/bin/env python3
"""使用訓練好的 no_speakers Transformer 模型推論對話風險分數，並輸出 JSON 檔。"""

from __future__ import annotations

import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_device() -> torch.device:
    """自動取得硬體加速裝置 (CUDA / Apple Silicon MPS / CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def risk_class(score_1_4: float) -> str:
    """判斷風險等級標籤"""
    if score_1_4 >= 3.0:
        return "high"
    if score_1_4 >= 2.0:
        return "medium"
    return "low"


def main():
    parser = argparse.ArgumentParser(description="預測對話訊息之衝突風險分數")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/all_no_speakers",
        help="已訓練好之 no_speakers 模型目錄",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/sample_predictions.json",
        help="預測結果 JSON 輸出檔名",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"📌 [運算裝置] 使用裝置: {device}")
    print(f"📖 [模型載入] 載入模型權重: {args.model_dir} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # 定義測試的 3 句代表性文字 (完全去除人名，符合真實通訊軟體無標籤輸入)
    sample_dialogues = [
        {
            "id": "sample_01",
            "relationship": "couple",
            "scenario": "One partner comes home exhausted after work while the other relaxes on the couch.",
            "history": ["Hey love. How was your day?"],
            "current_message": "Same old, honey. I got you your favorite dessert on the way home!",
            "label": "無衝突 / 正向對話 (Non-conflict)",
        },
        {
            "id": "sample_02",
            "relationship": "couple",
            "scenario": "Partners discussing household chores and spending weekend time.",
            "history": [
                "Hey, did you remember to take out the trash?",
                "I've been busy with work all day, I'll do it later."
            ],
            "current_message": "You always say 'later' and then I end up having to do it myself every single time.",
            "label": "中度衝突 / 抱怨抱怨 (Medium Risk)",
        },
        {
            "id": "sample_03",
            "relationship": "couple",
            "scenario": "A heated argument after one partner forgot their anniversary dinner reservation.",
            "history": [
                "I waited at the restaurant for 45 minutes alone.",
                "I told you my team meeting ran late, it's not a big deal."
            ],
            "current_message": "You're completely selfish and never care about my feelings! I am sick of your excuses!",
            "label": "高風險衝突 / 激烈人格攻擊 (High Risk)",
        },
    ]

    results: List[Dict[str, Any]] = []

    print("\n🔍 [開始預測風險分數...]")
    with torch.no_grad():
        for item in sample_dialogues:
            history_str = "\n".join(item["history"]) if item["history"] else ""
            input_text = (
                f"relationship: {item['relationship']}\n"
                f"scenario: {item['scenario']}\n"
                f"history:\n{history_str}\n"
                f"current_message:\n{item['current_message']}"
            ).strip()

            encoding = tokenizer(
                input_text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_0_1 = float(outputs.logits.reshape(-1)[0].cpu().item())
            
            # 將 [0, 1] 轉回 [1.0, 4.0] 原始標度
            pred_score_1_4 = round(pred_0_1 * 3.0 + 1.0, 2)
            level = risk_class(pred_score_1_4)
            needs_repair = pred_0_1 >= 0.30 or pred_score_1_4 >= 2.0

            prediction_entry = {
                "id": item["id"],
                "type": item["label"],
                "relationship": item["relationship"],
                "scenario": item["scenario"],
                "history": item["history"],
                "current_message": item["current_message"],
                "predicted_risk_score_0_1": round(pred_0_1, 4),
                "predicted_risk_score_1_4": pred_score_1_4,
                "risk_level": level,
                "requires_repair": needs_repair,
            }
            results.append(prediction_entry)

            print(
                f"  ➡️ [{item['id']}] 風險分數 (1-4分): {pred_score_1_4:.2f} | "
                f"等級: {level.upper()} | 需預防修復: {needs_repair}"
            )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ [結果存檔] 已將預測結果寫入 {output_path}")


if __name__ == "__main__":
    main()
