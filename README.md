# 伴侶對話衝突風險預測模型 (Couple Conversation Model)

## 腳本架構 (scripts/)

```text
scripts/
├── preprocess_persona_conflicts.py   # 【第 1 部份】資料清洗 (Data Cleaning & Context Preprocessing)
├── train_transformer.py             # 【第 2 部份】Tokenizer 載入與模型訓練/評估 (Tokenizer & Model Training)
└── run_experiments.py               # 【第 3 部份】最後跑的實驗與報告產出 (Final Execution Pipeline)
```

---

## 簡化三步驟執行指南

### 資料清洗 (Data Cleaning)

將 PersonaConflicts 原始對話資料進行清洗、清理敏感資訊並切分為 `train.jsonl`, `val.jsonl`, `test.jsonl`：

```bash
# 清洗全關係資料集
python3 scripts/preprocess_persona_conflicts.py

# 清洗伴侶專一資料集 (Couple Only)
python3 scripts/preprocess_persona_conflicts.py --relationship-subtype couple --output-dir data/processed/persona_conflicts_couple
```

---

### 2️⃣ 第 2 部份：Tokenizer & 模型訓練 (Tokenizer & Training)

載入 Transformer (DistilBERT) Tokenizer、將文字向量化，並訓練回歸模型預測風險分數 (評估重點：High-Risk Recall)：

```bash
python3 scripts/train_transformer.py \
  --data_dir data/processed/persona_conflicts/role_speakers \
  --output_dir models/my_step1_role_speakers \
  --epochs 3 \
  --batch_size 16
```

---

### 3️⃣ 第 3 部份：最後運行的完整實驗 (Final Part / Execution)

自動化執行全套 3 階段實驗（全關係 vs 伴侶資料、Role Speakers vs No Speakers），並產出對比報告：

```bash
python3 scripts/run_experiments.py --epochs 3 --batch_size 16
```

---

## 📊 實驗結論總結

1. **Speaker 標籤幫助**：`role_speakers` 提供對話發言順序，能將高風險召回率 (High-Risk Recall) 提升至 **88.17%**。
2. **資料量優於純領域**：全關係資料 (2,658 筆) 訓練之模型在伴侶測試集上的表現 (MAE=0.5294, Pearson=0.6296) 顯著優於僅用伴侶資料 (949 筆) 訓練之模型 (MAE=0.6700, Pearson=0.4826)。
