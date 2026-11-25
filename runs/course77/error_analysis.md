# Course-77 Package C — Model Selection & Error Analysis (Dev Split)

**Owner:** Nguyễn Tấn Hưng (`K5`)  
**Branch / Scope:** `course77` (BANKING77, 77 intents)  
**Evaluation Partition:** `dev` only (`1,999` queries; `train_fit = 7,994`; `official_test_touched = false`)  
**Primary Sources:**
- Comparison summary: [`runs/course77/package_b_comparison_metrics.json`](package_b_comparison_metrics.json)
- Selected model metrics & confusion matrix: [`runs/course77/e3_dev_metrics.json`](e3_dev_metrics.json)
- Baseline & ablation metrics: [`runs/course77/e1_dev_metrics.json`](e1_dev_metrics.json), [`runs/course77/e2_dev_metrics.json`](e2_dev_metrics.json), [`runs/course77/e2_no_lemma_dev_metrics.json`](e2_no_lemma_dev_metrics.json)
- Frozen split & audit: [`data/processed/course77/splits.json`](../../data/processed/course77/splits.json) (`sha256: 18a707b4b7047a9f24d0d0fff6cc81516a444478436583d87a9d8b2d88a77693`), [`runs/course77/course77_audit.json`](course77_audit.json)
- Saved pipeline artifact: [`artifacts/course77/e3_tfidf_svm.joblib`](../../artifacts/course77/e3_tfidf_svm.joblib) ([`artifacts/course77/e3_tfidf_svm_config.json`](../../artifacts/course77/e3_tfidf_svm_config.json))

---

## 1. Model Selection on Dev (by Macro-F1)

Using the frozen stratified `train_fit` / `dev` split (`seed=42`) from `runs/course77/package_b_comparison_metrics.json`:

| Experiment | Representation | Classifier | Lemmatize | Vocab Size | Accuracy | Macro-F1 | Errors (`/ 1,999`) |
|---|---|---|---:|---:|---:|---:|---:|
| `E1` | BoW (`CountVectorizer`) | `LogisticRegression(C=1.0)` | `True` | 1,584 | `0.870935` | `0.870725` | 258 |
| `E2` | TF-IDF (`TfidfVectorizer`) | `LogisticRegression(C=1.0)` | `True` | 1,584 | `0.870935` | `0.867826` | 258 |
| **`E3` (Selected)** | **TF-IDF (`TfidfVectorizer`)** | **`LinearSVC(C=1.0)`** | **`True`** | **1,584** | **`0.886943`** | **`0.886393`** | **226** |
| `E2_no_lemma` | TF-IDF (`TfidfVectorizer`) | `LogisticRegression(C=1.0)` | `False` | 2,127 | `0.858929` | `0.857982` | 282 |

* **Selected Configuration:** **`E3`** — `Course77Preprocessor(lowercase=True, expand_contractions=True, lemmatize=True)` + `TfidfVectorizer(ngram_range=(1, 1), min_df=1)` + `LinearSVC(C=1.0, max_iter=2000, random_state=42)`.
* **Selection Rationale:** `E3` achieves the highest dev **Macro-F1 (`0.886393`)** and **Accuracy (`0.886943`)**, beating `E2` by `+0.018567` Macro-F1 (`-32` errors) and `E1` by `+0.015668` Macro-F1 (`-32` errors). Comparing `E2` vs `E2_no_lemma` confirms that WordNet lemmatization shrinks vocabulary sparsity (`2,127 -> 1,584` tokens) and improves overall generalization (`+0.009844` Macro-F1 on `E2`).

---

## 2. Two Most Confused Intent Pairs on Dev (`E3`)

From the `77 x 77` dev confusion matrix in `runs/course77/e3_dev_metrics.json`:

### Top 2 Directed Confusions (`gold -> pred`)
1. **`top_up_failed` -> `pending_top_up` (5 errors)** — plus 1 reverse error (`pending_top_up` -> `top_up_failed`), totaling **6 bidirectional errors**.
2. **`why_verify_identity` -> `verify_my_identity` (4 errors)** — 0 reverse errors (**4 bidirectional errors** total).

### Other Notable Confused Pairs (Next Highest on Dev)
* **`beneficiary_not_allowed` <-> `failed_transfer`**: **5 bidirectional errors** (`3` `beneficiary_not_allowed -> failed_transfer`, `2` `failed_transfer -> beneficiary_not_allowed`).
* **`Refund_not_showing_up` <-> `request_refund`**: **4 bidirectional errors** (`3` `Refund_not_showing_up -> request_refund`, `1` `request_refund -> Refund_not_showing_up`).
* **`direct_debit_payment_not_recognised` <-> `card_payment_not_recognised`**: **4 bidirectional errors** (`3` `direct_debit -> card`, `1` `card -> direct_debit`).
* **`top_up_reverted` <-> `top_up_failed`**: **4 bidirectional errors** (`3` `top_up_reverted -> top_up_failed`, `1` `top_up_failed -> top_up_reverted`).
* **`pending_transfer` <-> `transfer_not_received_by_recipient`** and **`pending_transfer` <-> `transfer_timing`**: **4 bidirectional errors** each (`3 + 1`).

---

## 3. Analysis of 10 Misclassified Dev Queries (`E3`)

> Note: `LinearSVC` outputs signed hyperplane decision scores (`decision_function`) across 77 one-vs-rest classes. Feature contributions `(w_pred - w_gold) * tfidf(x)` below are measured directly from `artifacts/course77/e3_tfidf_svm.joblib`.

| # | Sample ID | Original Query | Gold Label (`score`) | Predicted Label (`score`) | Why `E3` Confused Them (Measured Feature Drivers & Root Cause) |
|---:|---|---|---|---|---|
| 1 | `train/8471` | `"My top-up hasn't gone through, what happened?"` | `top_up_failed` (`-0.678`) | `pending_top_up` (`+0.258`) | **Shared lexical phrase:** Preprocessed to `"my top up have not go through what happen"`. The single token `"through"` adds `+0.566` (`w_pred=+0.492` vs `w_gold=-0.074`) toward `pending_top_up`, plus `"happen"` (`+0.175`). Unigram TF-IDF cannot distinguish a failed vs still-pending incomplete top-up. |
| 2 | `train/8512` | `"My top-up didn't go through"` | `top_up_failed` (`-0.202`) | `pending_top_up` (`+0.341`) | **Short query + lemmatization tense loss:** Expansion + lemmatization turn `"didn't"` into `"do not"` (`"my top up do not go through"`), erasing past-tense failure nuance. `"through"` contributes `+0.711` (`w_pred=+0.618` vs `w_gold=-0.093`) and `"top"` adds `+0.140` toward `pending_top_up`. |
| 3 | `train/2056` | `"Problem with my top up going through"` | `pending_top_up` (`-0.285`) | `top_up_failed` (`+0.286`) | **Reverse keyword trap:** The generic word `"problem"` contributes `+1.108` (`w_pred=+0.936` vs `w_gold=-0.172`) toward `top_up_failed`, completely overriding `"going through"` which normally signals `pending_top_up`. |
| 4 | `train/3831` | `"What do you need my identity details for?"` | `why_verify_identity` (`-0.121`) | `verify_my_identity` (`-0.038`) | **Unigram loss of question syntax:** `"what"` (`+0.618`), `"for"` (`+0.402`), and `"identity"` (`+0.306`, `w_pred=+1.010`) favor `verify_my_identity`, because unigrams break apart the long-distance purpose frame `"what ... for"`. |
| 5 | `train/3852` | `"What is the purpose for verifying my identity?"` | `why_verify_identity` (`-0.322`) | `verify_my_identity` (`-0.238`) | **Lemmatization trade-off:** In Package B, `E2_no_lemma` (`lemmatize=False`) predicted this **correctly** as `why_verify_identity`, whereas `E1`/`E2`/`E3` (`lemmatize=True`) all failed because lemmatizing `"verifying"` -> `"verify"` (`+0.279` toward `verify_my_identity`) erased the gerund cue in `"purpose for verifying"`. |
| 6 | `train/7029` | `"Why can I not do a  transfer"` | `beneficiary_not_allowed` (`-0.268`) | `failed_transfer` (`+0.487`) | **Symptom without policy cause:** The query gives only the surface symptom (*cannot do a transfer*) with no `"beneficiary"` or `"disallowed"` keyword; `"can"` (`+0.505`) and `"transfer"` (`+0.483`, `w_pred=+1.084`) push strongly to `failed_transfer`. |
| 7 | `train/7490` | `"Errors keep showing up every time I try to make this transfer and I am on a time crunch,  could you please help."` | `failed_transfer` (`-0.429`) | `beneficiary_not_allowed` (`-0.289`) | **Reverse overlap on transfer errors:** Both intents share `"error"`, `"try"`, `"make"`, and `"transfer"`; all 77 scores are negative, and `"error"` (`+0.289`) + `"could"` (`+0.253`) tip the margin slightly to `beneficiary_not_allowed`. |
| 8 | `train/4792` | `"There is a payment I don't recognise!"` | `direct_debit_payment_not_recognised` (`-1.142`) | `card_payment_not_recognised` (`+1.100`) | **Underspecified payment channel:** The query omits `"direct debit"` or `"card"`. Without channel keywords, `"recognise"` (`+1.123`) and `"payment"` (`+0.937`) strongly default to the higher-prior `card_payment_not_recognised`. |
| 9 | `train/5728` | `"Hello. I bought an item a while ago, and have since requested a refund.  However, upon checking my online statements, it seems that I haven't been issued a refund yet.  How can this be remedied?"` | `Refund_not_showing_up` (`+0.008`) | `request_refund` (`+0.484`) | **Bag-of-unigrams misses discourse contrast:** Background context tokens `"item"` (`+0.523`) and `"buy"` (`+0.348`) boost `request_refund`, while unigrams fail to bind the temporal contrast *"requested ... however ... haven't been issued yet"*. |
| 10 | `train/87` | `"Where is my card?"` | `card_arrival` (`-0.144`) | `card_acceptance` (`+0.048`) | **Ultra-short query sparsity:** Only 4 tokens remain (`"where be my card"`). Without shipping/arrival verbs (`"arrive"`, `"ship"`), `"where"` alone adds `+0.905` (`w_pred=+0.962` vs `w_gold=+0.057`) toward `card_acceptance` (*"Where can I use my card?"*). |

---

## 4. Intent Pair Hard to Distinguish by Keywords Alone

* **Pair:** **`top_up_failed` vs `pending_top_up`** (and similarly **`beneficiary_not_allowed` vs `failed_transfer`**).
* **Why keywords fail:** Both intents describe the exact same user-facing observation — a user tried to add funds and the balance has not updated yet. Because customers rarely know the internal settlement vs rejection reason, both classes rely on identical unigrams (`"top"`, `"up"`, `"not"`, `"go"`, `"through"`). Unigram TF-IDF + lemmatization drops word order and collapses tense cues (`"didn't"` -> `"do not"` vs `"hasn't"` -> `"have not"`), leaving the classifier sensitive to single generic words like `"through"` or `"problem"`.
* **Illustrative Example:**
  * **`train/8512`:** *"My top-up didn't go through"* → Gold: `top_up_failed` | Predicted by `E3`: `pending_top_up` (`"through"` pushes `+0.711` toward `pending_top_up`).
  * **Contrast with `train/2056`:** *"Problem with my top up going through"* → Gold: `pending_top_up` | Predicted by `E3`: `top_up_failed` (`"problem"` pushes `+1.108` toward `top_up_failed`).
  * Single-word weights flip predictions in opposite directions because keywords alone cannot separate a delayed top-up from a failed one without account/transaction status or richer sequence modeling.

---

## 5. Summary for Report & Package D

1. **Selected Dev Model for Freeze in Package D:** `E3` (`TF-IDF + LinearSVC(C=1.0)`, `lemmatize=True`, dev Macro-F1 `0.886393`, dev Accuracy `0.886943`).
2. **Main Error Drivers Observed on Dev:**
   * **Near-synonym / symptom-overlapping intents** (`top_up_failed` vs `pending_top_up`, `beneficiary_not_allowed` vs `failed_transfer`, `Refund_not_showing_up` vs `request_refund`).
   * **Loss of syntax, discourse contrast, and tense cues under unigram TF-IDF + lemmatization** (`"what ... for"` in `why_verify_identity`, `"verifying"` -> `"verify"` in `train/3852`, `"requested ... however ... haven't been issued yet"` in `train/5728`).
   * **Ultra-short or underspecified queries** (`"Where is my card?"`, `"There is a payment I don't recognise!"`).
3. **Next Step (Package D):** Freeze configuration on dev, run one-time evaluation on the held-out `official test` split (`3,080` queries), and merge final test metrics and error analysis into the course report.
