#!/usr/bin/env python3
"""PersonaConflicts 衝突風險評估 Transformer 回歸模型訓練腳本。

本腳本預測 risk_score_0_1 (0.0 至 1.0 的浮點數風險分數)。
評估指標包含：
1. 回歸指標：MAE、RMSE、Pearson 相關係數 r、Spearman 排序相關係數 rho
2. 分類指標：高風險召回率 (High-Risk Recall)、精準率 (Precision)、F1-score
"""

from __future__ import annotations

import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
try:
    import scipy.stats as stats
except ImportError:
    stats = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def calc_correlations(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """計算 Pearson 線性相關與 Spearman 排序相關係數（具備 NumPy 降級安全機制）。"""
    if len(preds) <= 1 or np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
        return 0.0, 0.0
    if stats is not None:
        try:
            r, _ = stats.pearsonr(preds, targets)
            rho, _ = stats.spearmanr(preds, targets)
            return float(r), float(rho)
        except Exception:
            pass
    # 降級使用 NumPy 計算
    r = float(np.corrcoef(preds, targets)[0, 1])
    pred_ranks = np.argsort(np.argsort(preds))
    target_ranks = np.argsort(np.argsort(targets))
    rho = float(np.corrcoef(pred_ranks, target_ranks)[0, 1])
    return r, rho


def set_seed(seed: int):
    """固定所有隨機種子，確保實驗結果可重複呈現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """自動偵測目前硬體加速裝置 (NVIDIA CUDA / Apple Silicon MPS / CPU)。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConflictDataset(Dataset):
    """衝突對話 Dataset 類別，負責將 JSONL 轉為 PyTorch Tensor 格式。"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 256, max_samples: int | None = None):
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        if max_samples and max_samples > 0:
            self.examples = self.examples[:max_samples]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        text = item["input_text"]
        target = float(item["risk_score_0_1"])
        orig_score = float(item["risk_score"])
        risk_class = item.get("risk_class", "low")

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "example_id": item.get("example_id", f"idx_{idx}"),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.float),
            "orig_score": orig_score,
            "risk_class": risk_class,
            "relationship_subtype": item.get("relationship_subtype", "unknown"),
        }


def compute_metrics(
    preds: np.ndarray, targets: np.ndarray, orig_scores: np.ndarray, fixed_threshold: float = 0.6667
) -> Dict[str, Any]:
    """計算完整模型指標，包含 MAE, RMSE, Pearson, Spearman 與高風險 Recall。"""
    # 0-1 scale 上的回歸誤差
    mae_0_1 = float(np.mean(np.abs(preds - targets)))
    mse_0_1 = float(np.mean((preds - targets) ** 2))
    rmse_0_1 = float(np.sqrt(mse_0_1))

    # 還原至原始 1.0-4.0 分量表上的 MAE / RMSE
    preds_1_4 = preds * 3.0 + 1.0
    targets_1_4 = targets * 3.0 + 1.0
    mae_1_4 = float(np.mean(np.abs(preds_1_4 - targets_1_4)))
    rmse_1_4 = float(np.sqrt(np.mean((preds_1_4 - targets_1_4) ** 2)))

    # 計算相關係數
    pearson_r, spearman_rho = calc_correlations(preds, targets)

    # 二元高風險分類：目標值 >= 0.6667 (對應原始風險分數 >= 3.0)
    true_high = (targets >= 0.6667).astype(int)
    pred_high_fixed = (preds >= fixed_threshold).astype(int)

    tp_fixed = int(np.sum((pred_high_fixed == 1) & (true_high == 1)))
    fp_fixed = int(np.sum((pred_high_fixed == 1) & (true_high == 0)))
    fn_fixed = int(np.sum((pred_high_fixed == 0) & (true_high == 1)))
    tn_fixed = int(np.sum((pred_high_fixed == 0) & (true_high == 0)))

    recall_fixed = float(tp_fixed / (tp_fixed + fn_fixed)) if (tp_fixed + fn_fixed) > 0 else 0.0
    precision_fixed = float(tp_fixed / (tp_fixed + fp_fixed)) if (tp_fixed + fp_fixed) > 0 else 0.0
    f1_fixed = (
        float(2 * precision_fixed * recall_fixed / (precision_fixed + recall_fixed))
        if (precision_fixed + recall_fixed) > 0
        else 0.0
    )

    # 三類別風險分級：Low (<2.0 -> <0.3333), Medium (2.0-3.0 -> 0.3333-0.6667), High (>=3.0 -> >=0.6667)
    def to_class(scores: np.ndarray) -> np.ndarray:
        classes = np.zeros(len(scores), dtype=int)
        classes[scores >= 0.3333] = 1  # Medium
        classes[scores >= 0.6667] = 2  # High
        return classes

    true_classes = to_class(targets)
    pred_classes = to_class(preds)

    cm = confusion_matrix(true_classes, pred_classes, labels=[0, 1, 2]).tolist()
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_classes, pred_classes, labels=[0, 1, 2], zero_division=0
    )

    return {
        "mae_0_1": mae_0_1,
        "mse_0_1": mse_0_1,
        "rmse_0_1": rmse_0_1,
        "mae_1_4": mae_1_4,
        "rmse_1_4": rmse_1_4,
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "high_risk_fixed": {
            "threshold": fixed_threshold,
            "tp": tp_fixed,
            "fp": fp_fixed,
            "fn": fn_fixed,
            "tn": tn_fixed,
            "recall": recall_fixed,
            "precision": precision_fixed,
            "f1": f1_fixed,
        },
        "three_class": {
            "confusion_matrix": cm,
            "low_precision": float(prec[0]),
            "low_recall": float(rec[0]),
            "low_f1": float(f1[0]),
            "medium_precision": float(prec[1]),
            "medium_recall": float(rec[1]),
            "medium_f1": float(f1[1]),
            "high_precision": float(prec[2]),
            "high_recall": float(rec[2]),
            "high_f1": float(f1[2]),
            "macro_f1": float(np.mean(f1)),
        },
    }


def find_optimal_threshold(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    """在 Validation 集上網格搜尋最佳高風險判定門檻（極致化 High-Risk Recall）。"""
    true_high = (targets >= 0.6667).astype(int)
    best_thresh = 0.6667
    best_recall = 0.0
    best_prec = 0.0
    best_f1 = 0.0

    thresholds = np.linspace(0.3, 0.8, 51)
    best_score = -1.0

    for th in thresholds:
        pred_high = (preds >= th).astype(int)
        tp = int(np.sum((pred_high == 1) & (true_high == 1)))
        fp = int(np.sum((pred_high == 1) & (true_high == 0)))
        fn = int(np.sum((pred_high == 0) & (true_high == 1)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

        # 以召回率 (70%) 與 F1-score (30%) 為綜合指標進行調優
        score = recall * 0.7 + f1 * 0.3
        if score > best_score:
            best_score = score
            best_thresh = float(th)
            best_recall = float(recall)
            best_prec = float(prec)
            best_f1 = float(f1)

    return {
        "optimal_threshold": best_thresh,
        "recall": best_recall,
        "precision": best_prec,
        "f1": best_f1,
    }


def evaluate(
    model, dataloader: DataLoader, device: torch.device, fixed_threshold: float = 0.6667, disable_tqdm: bool = False
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """模型推論與評估涵式。"""
    model.eval()
    all_preds = []
    all_targets = []
    all_orig_scores = []
    details = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="🔍 [評估中]", disable=disable_tqdm, leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].numpy()
            orig_scores = batch["orig_score"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1).cpu().numpy()

            all_preds.extend(logits.tolist())
            all_targets.extend(targets.tolist())
            all_orig_scores.extend(orig_scores.tolist())

            for i in range(len(logits)):
                details.append(
                    {
                        "example_id": batch["example_id"][i],
                        "target_0_1": float(targets[i]),
                        "pred_0_1": float(logits[i]),
                        "target_score": float(orig_scores[i]),
                        "pred_score": float(logits[i] * 3.0 + 1.0),
                        "risk_class": batch["risk_class"][i],
                        "relationship_subtype": batch["relationship_subtype"][i],
                    }
                )

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    orig_scores_arr = np.array(all_orig_scores)

    metrics = compute_metrics(preds_arr, targets_arr, orig_scores_arr, fixed_threshold=fixed_threshold)
    return metrics, preds_arr, targets_arr, details


def main():
    parser = argparse.ArgumentParser(description="訓練 Transformer 衝突風險評估回歸模型")
    parser.add_argument("--data_dir", type=str, required=True, help="包含 train/val/test.jsonl 的資料目錄")
    parser.add_argument("--eval_data_dir", type=str, default=None, help="可選的二次評估資料目錄")
    parser.add_argument("--output_dir", type=str, required=True, help="模型權重與評估結果儲存目錄")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased", help="預訓練模型名稱或路徑")
    parser.add_argument("--epochs", type=int, default=5, help="訓練 Epoch 數")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch 大小")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="學習率 (Learning rate)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列長度")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--max_train_samples", type=int, default=None, help="限制最大訓練筆數（方便快速測試）")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="限制最大評估筆數（方便快速測試）")
    parser.add_argument("--disable_tqdm", action="store_true", help="停用 tqdm 進度條")
    parser.add_argument("--eval_only", action="store_true", help="僅載入已訓練模型進行測試集評估")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"📌 [運算裝置] 使用裝置: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # 僅評估模式
    if args.eval_only:
        print(f"📖 [模型載入] 載入模型 {args.model_name_or_path} 進行測試集評估...")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(device)

        eval_dir = Path(args.eval_data_dir) if args.eval_data_dir else Path(args.data_dir)
        test_dataset = ConflictDataset(str(eval_dir / "test.jsonl"), tokenizer, max_length=args.max_length, max_samples=args.max_eval_samples)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        metrics, preds, targets, details = evaluate(model, test_loader, device, disable_tqdm=args.disable_tqdm)
        opt_thresh_info = find_optimal_threshold(preds, targets)
        metrics["high_risk_optimal"] = opt_thresh_info

        print("\n==================================================")
        print("🏆 【測試集單獨評估結果 (Eval-Only Test Results)】")
        print(f" 1. 平均絕對誤差 (MAE, 1-4分 scale): {metrics['mae_1_4']:.4f}")
        print(f" 2. Pearson 線性相關係數 (r):        {metrics['pearson_r']:.4f}")
        print(f" 3. Spearman 排序相關係數 (rho):    {metrics['spearman_rho']:.4f}")
        print(f" 4. 高風險召回率 (預設門檻 0.6667):   {metrics['high_risk_fixed']['recall']*100:.2f}%  (精準率: {metrics['high_risk_fixed']['precision']*100:.2f}%)")
        print(f" 5. 高風險召回率 (最佳門檻 {opt_thresh_info['optimal_threshold']:.4f}):   {opt_thresh_info['recall']*100:.2f}%  (精準率: {opt_thresh_info['precision']*100:.2f}%)")

        eval_output_name = "metrics_eval_only.json" if args.eval_data_dir else "metrics.json"
        with open(output_path / eval_output_name, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(output_path / "predictions_eval_only.jsonl", "w", encoding="utf-8") as f:
            for item in details:
                f.write(json.dumps(item) + "\n")
        return

    # 完整訓練模式
    train_dir = Path(args.data_dir)
    train_dataset = ConflictDataset(str(train_dir / "train.jsonl"), tokenizer, max_length=args.max_length, max_samples=args.max_train_samples)
    val_dataset = ConflictDataset(str(train_dir / "val.jsonl"), tokenizer, max_length=args.max_length, max_samples=args.max_eval_samples)
    test_dataset = ConflictDataset(str(train_dir / "test.jsonl"), tokenizer, max_length=args.max_length, max_samples=args.max_eval_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"📦 [資料集大小] 訓練集: {len(train_dataset)} 筆 | 驗證集: {len(val_dataset)} 筆 | 測試集: {len(test_dataset)} 筆")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_val_recall = 0.0

    print("🚀 [開始模型訓練...]")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"🚀 [訓練中] Epoch {epoch}/{args.epochs}", disable=args.disable_tqdm)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(targets)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_dataset)

        val_metrics, val_preds, val_targets, _ = evaluate(model, val_loader, device, disable_tqdm=args.disable_tqdm)
        val_recall = val_metrics["high_risk_fixed"]["recall"]
        val_mae = val_metrics["mae_1_4"]

        print(
            f"📊 Epoch {epoch}/{args.epochs} | 訓練集 Loss (MSE): {train_loss:.4f} | "
            f"驗證集 MAE (1-4分): {val_mae:.4f} | Pearson 相關: {val_metrics['pearson_r']:.4f} | "
            f"高風險 Recall: {val_recall * 100:.2f}%",
            flush=True,
        )

        # 儲存最佳 Checkpoint（優先著重於更低的 MAE 與更高的 High-Risk Recall）
        if val_mae < best_val_mae or (abs(val_mae - best_val_mae) < 0.02 and val_recall > best_val_recall):
            best_val_mae = val_mae
            best_val_recall = val_recall
            print(f"  💾 -> 儲存新最佳模型 Checkpoint (MAE: {val_mae:.4f}, 高風險 Recall: {val_recall * 100:.2f}%)", flush=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    print("\n✅ [訓練完成！開始使用最佳 Checkpoint 評估測試集...]", flush=True)
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)

    # 在 Validation 集上尋找最佳判定門檻
    _, val_preds, val_targets, _ = evaluate(best_model, val_loader, device)
    opt_thresh_info = find_optimal_threshold(val_preds, val_targets)
    best_threshold = opt_thresh_info["optimal_threshold"]
    print(f"🎯 [驗證集最佳門檻搜尋] 最佳判定門檻: {best_threshold:.4f} (Validation 高風險召回率: {opt_thresh_info['recall'] * 100:.2f}%)")

    test_metrics, test_preds, test_targets, test_details = evaluate(
        best_model, test_loader, device, fixed_threshold=0.6667
    )
    opt_test_metrics = compute_metrics(test_preds, test_targets, np.array([d["target_score"] for d in test_details]), fixed_threshold=best_threshold)

    test_metrics["high_risk_optimal_on_val"] = {
        "optimal_threshold": best_threshold,
        "recall": opt_test_metrics["high_risk_fixed"]["recall"],
        "precision": opt_test_metrics["high_risk_fixed"]["precision"],
        "f1": opt_test_metrics["high_risk_fixed"]["f1"],
    }

    print("\n==================================================")
    print("🏆 【最終測試集評估結果 (Test Set Evaluation Results)】")
    print(f" 1. 平均絕對誤差 (MAE, 1-4分 scale): {test_metrics['mae_1_4']:.4f}")
    print(f" 2. Pearson 線性相關係數 (r):        {test_metrics['pearson_r']:.4f}")
    print(f" 3. Spearman 排序相關係數 (rho):    {test_metrics['spearman_rho']:.4f}")
    print(f" 4. 高風險召回率 (預設門檻 0.6667):   {test_metrics['high_risk_fixed']['recall']*100:.2f}%  (精準率 Precision: {test_metrics['high_risk_fixed']['precision']*100:.2f}%)")
    print(f" 5. 高風險召回率 (驗證集調優門檻 {best_threshold:.4f}): {test_metrics['high_risk_optimal_on_val']['recall']*100:.2f}%  (精準率 Precision: {test_metrics['high_risk_optimal_on_val']['precision']*100:.2f}%)")

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    with open(output_path / "predictions.jsonl", "w", encoding="utf-8") as f:
        for item in test_details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
