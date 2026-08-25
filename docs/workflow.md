# 伴侶衝突評分模型流程

這個專案建議先做「評分模型」，再把「建議與改寫」接到現成 LLM。原因很簡單：評分模型需要穩定、可量化、可測試；改寫建議比較像生成任務，用 Ollama 或其他 LLM 會快很多。

## 目標拆分

第一階段先做：

1. 輸入一則準備送出的訊息，加上前幾句對話脈絡。
2. 模型輸出「生氣/傷害風險分數」。
3. 額外輸出風險類別：`low`、`medium`、`high`。

第二階段再做：

1. 用評分模型判斷原句風險。
2. 把原句、前文、分數、可能風險原因丟給 Ollama。
3. 讓 LLM 產生「分析」和「幾種改寫」。

## 為什麼先不斷詞

這次資料是英文對話，而且之後會用 transformer 類模型。transformer tokenizer 會自己把文字切成 subword token，所以不需要自己先做中文斷詞或英文分詞。前處理重點不是「切詞」，而是：

1. 清掉不該讓模型背答案的欄位。
2. 把對話拆成模型能學的樣本。
3. 確保 train/validation/test 沒有資料外洩。
4. 檢查分數分布和標籤品質。

## 資料怎麼用

這份 corpus 有兩層資料：

1. `dataset_final.csv`：完整模擬對話，約 5,772 筆，但沒有完整人工逐輪評分。
2. `mturk_aggregate.csv`：人工標註彙整，240 筆對話，包含每一輪的問題程度分數。

第一版應該先用 `mturk_aggregate.csv`。雖然資料比較少，但標籤比較可靠。

## 建議的訓練樣本格式

每一列是一個 turn-level example：

```text
relationship: couple
scenario: ...
history:
speaker_a: 前一句
speaker_b: 再前一句
current_message:
speaker_a: 這句準備被評分
```

標籤：

```text
risk_score: 1.0 到 4.0
risk_score_0_1: 0.0 到 1.0
risk_class: low / medium / high
vc_labels: violent communication 類型
nvc_labels: nonviolent communication 類型
```

## Speaker 要不要拿掉

先不要只做一個版本。這份前處理會產出兩種資料：

1. `no_speakers`：完全移除 speaker 名字，只保留文字。
2. `role_speakers`：把真實名字換成 `speaker_a`、`speaker_b`。

建議第一版先訓練 `role_speakers`，因為伴侶對話裡「誰接著誰說」通常有用；但真實姓名可能會讓模型學到奇怪偏差，所以不要保留人名。

## Step 1：下載資料

```bash
git clone https://github.com/mitmedialab/persona-conflicts-corpus-emnlp-2025.git data/persona-conflicts-corpus-emnlp-2025
```

## Step 2：跑前處理

產生全部關係類型：

```bash
python3 scripts/preprocess_persona_conflicts.py
```

只產生伴侶資料：

```bash
python3 scripts/preprocess_persona_conflicts.py --relationship-subtype couple --output-dir data/processed/persona_conflicts_couple
```

## Step 3：檢查輸出

你會看到：

```text
data/processed/persona_conflicts/
  summary.json
  no_speakers/
    train.jsonl
    val.jsonl
    test.jsonl
  role_speakers/
    train.jsonl
    val.jsonl
    test.jsonl
```

先打開 `summary.json`，看三件事：

1. train/val/test 是否都有資料。
2. `low`、`medium`、`high` 是否太不平均。
3. couple 的資料量是否足夠。

## Step 4：第一版模型

第一版可以用 `distilbert-base-uncased` 或 `roberta-base` 做 regression：

```text
input_text -> risk_score_0_1
```

訓練完用這些指標看結果：

1. MAE：預測分數平均差多少。
2. Pearson/Spearman：模型排序能力好不好。
3. high-risk recall：真正高風險的句子有沒有被抓到。

你的產品比較在乎「高風險不要漏掉」，所以 high-risk recall 比整體 accuracy 更重要。

## Step 5：接 LLM 做建議

當評分模型輸出高風險時，再把下面資訊給 Ollama：

```text
前文：
...

原句：
...

模型風險：
high, 0.82

請輸出：
1. 目前真正需要處理的情緒/需求
2. 原句風險
3. 三種較溫和改寫
```

這樣系統會比較穩：評分由小模型負責，建議由 LLM 負責。

## 建議開發順序

1. 先把資料前處理跑通。
2. 訓練 `role_speakers` 的 regression baseline。
3. 訓練 `no_speakers`，比較是否變好。
4. 只用 couple 子集再訓練一次，確認是否資料太少。
5. 決定第一版產品用 all-relationship 模型還是 couple-only 模型。
6. 最後才接 Ollama 產生分析和改寫。
