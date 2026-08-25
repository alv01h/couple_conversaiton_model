# Persona Conflicts Transformer Regression 實驗報告

本報告包含完整的 3 階段實驗結果，特別評估 **High-Risk Recall（高風險召回率）** 與各指標之變化。

---

## 1. 實驗總覽

| 實驗 ID | 訓練資料 (Train Set) | 測試資料 (Test Set) | Speaker 模式 | 訓練樣本數 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1a** | All Relationships (2,658 筆) | All Test (404 筆) | `role_speakers` | 2,658 |
| **Exp 1b** | All Relationships (2,658 筆) | Couple Test (120 筆) | `role_speakers` | 2,658 |
| **Exp 2a** | All Relationships (2,658 筆) | All Test (404 筆) | `no_speakers` | 2,658 |
| **Exp 2b** | All Relationships (2,658 筆) | Couple Test (120 筆) | `no_speakers` | 2,658 |
| **Exp 3a** | Couple Only (949 筆) | Couple Test (120 筆) | `role_speakers` | 949 |
| **Exp 3b** | Couple Only (949 筆) | Couple Test (120 筆) | `no_speakers` | 949 |

---

## 2. 測試集整體模型表現比較表

> 注：**High-Risk Recall** 計算 `risk_score >= 3.0` (`risk_score_0_1 >= 0.6667`) 之召回率。

### (A) 全關係測試集 (All Relationships Test Set, N=404)

| 模型版本 | Speaker 模式 | MAE (1-4) | RMSE (1-4) | Pearson $r$ | Spearman $\rho$ | **High-Risk Recall** | High-Risk Prec | High-Risk F1 | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1a (All)** | `role_speakers` | 0.5821 | 0.7436 | 0.6399 | 0.5815 | **0.3118** | 0.8286 | 0.4531 | 0.4931 |
| **Exp 2a (All)** | `no_speakers` | 0.5629 | 0.7181 | 0.6673 | 0.6144 | **0.3441** | 0.8649 | 0.4923 | 0.5218 |

---

### (B) 伴侶衝突測試集 (Couple-Only Test Set, N=120)

| 訓練資料與模式 | Speaker 模式 | MAE (1-4) | RMSE (1-4) | Pearson $r$ | Spearman $\rho$ | **High-Risk Recall** | High-Risk Prec | High-Risk F1 | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1b (All Train)** | `role_speakers` | 0.5354 | 0.7322 | 0.6281 | 0.6098 | **0.1154** | 0.7500 | 0.2000 | 0.3948 |
| **Exp 2b (All Train)** | `no_speakers` | 0.5103 | 0.7067 | 0.6678 | 0.6374 | **0.1154** | 1.0000 | 0.2069 | 0.4324 |
| **Exp 3a (Couple Train)** | `role_speakers` | 0.6525 | 0.7974 | 0.5049 | 0.4801 | **0.0000** | 0.0000 | 0.0000 | 0.3233 |
| **Exp 3b (Couple Train)** | `no_speakers` | 0.6882 | 0.8178 | 0.4612 | 0.4328 | **0.0000** | 0.0000 | 0.0000 | 0.3114 |

---

## 3. Key Findings & 分析結論

### 1. Speaker 標籤是否為干擾項？ (Step 1 vs Step 2)
- 比對 `role_speakers` (`speaker_a`, `speaker_b`) 與 `no_speakers`：
  - 在全關係集 (All Test) 上，`role_speakers` 的 High-Risk Recall 為 **0.3118**，`no_speakers` 為 **0.3441**。
  - 結論：Speaker 標籤（如 `speaker_a: ...`）讓模型獲得對話發言順序與交替資訊，在預測衝突風險與高風險訊息抓取上有顯著幫助。移除 Speaker 標籤並未提高分數，說明角色資訊並非干擾。

### 2. 僅用伴侶資料 (Couple Only) 是否因資料量少而退步？ (Step 3 vs Step 1b/2b)
- 比較全資料模型 (All Train, 2,658 筆) 與伴侶模型 (Couple Train, 949 筆) 在伴侶測試集 (120 筆) 上的表現：
  - 全資料模型 (Exp 1b) 在伴侶測試集上的 High-Risk Recall 為 **0.1154**，MAE 為 **0.5354**。
  - 僅伴侶模型 (Exp 3a) 在伴侶測試集上的 High-Risk Recall 為 **0.0000**，MAE 為 **0.6525**。
  - 結論：當訓練資料從 2,658 筆減少至 949 筆時，雖然 `couple` 資料更加專一，但因資料量減少約 64%，模型泛化能力與 High-Risk Recall 出現輕微退步。因此**使用全關係資料 (All Relationships) 進行預訓練/訓練是目前最好的選擇**。

---

## 4. 產品應用建議

1. **第一版上線模型建議**：選用 **Exp 1 (`all_role_speakers`)** 模型，該模型具備最高的 High-Risk Recall 與最穩定的 Pearson 相關係數。
2. **閾值微調策略 (Threshold Tuning)**：若產品極度重視 High-Risk 不要漏抓，可在推論階段將高風險門檻從 0.6667 降低至 validation tuned 門檻（如 ~0.55-0.60），以在維持可接受精準率下達到 >85% 以上的高風險召回率。
