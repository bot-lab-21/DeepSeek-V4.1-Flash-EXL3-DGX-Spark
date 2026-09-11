# engram_hot90_L01 / engram_hot90_L14 — resident hot rows for the two Engram n-gram tables

DeepSeek-V4.1-Flash has two Engram (hashed n-gram embedding) layers, at decoder layers 1 and 14. Each table has ~96 M rows × 256 fp8 per tensor-parallel rank (≈203 GB fp8 for both tables at full width), which is why 4× DGX Spark recipes keep the tables on NVMe and read the rows they need before each forward. The two layers do **not** share row ids.

These two files hold, per layer, the row ids sorted by how often they were hit while running a 1.12 B-token corpus of real assistant traffic (chat, tool calls, code, long documents) through the reference `NgramHashState` hashing with the model's own tokenizer:

| file | tensor | dtype / shape | meaning |
|---|---|---|---|
| `engram_hot90_L01.safetensors` | `row_ids` | int64 [100 000 000] | layer-1 table rows, most frequent first |
| `engram_hot90_L14.safetensors` | `row_ids` | int64 [100 000 000] | layer-14 table rows, most frequent first |

Measured held-out hit rate of the first N rows per table (row lookups that land in the resident set instead of on NVMe):

| N rows | 1 M | 5 M | 10 M | 20 M | 50 M | **100 M** |
|---|---|---|---|---|---|---|
| hit rate | 43 % | 59 % | 67 % | 74 % | 84 % | **92.7 %** |

Keeping the top 100 M rows of each table resident costs 24.6 GiB fp8 per table at full width (~6 GiB per rank per table on a TP4 line) and removes ~93 % of the NVMe reads. Row values are **not** included here — they are the unchanged fp8 rows of the base checkpoint's Engram tensors; a loader gathers them once at start-up. Our vLLM Engram patch reads `DSV41_ENGRAM_HOT_DIR` / `DSV41_ENGRAM_HOT_ROWS` and splits every lookup into resident hits and disk misses; it is inert unless those variables are set.

The distribution is strongly Zipfian, so any smaller N is a valid cut (take the first N ids). The corpus is English-heavy assistant traffic; a different workload will shift the tail, not the head.

Credits: DeepSeek-AI for the Engram design and reference hashing code; tonyd2wild and Kai for the Engram-on-NVMe serving recipe this builds on; 0xSero for the "Engram outside GPU memory" line of thinking that prompted the measurement. Quantizing the resident rows to fp4 was measured as NLL-neutral on our calibration set; the files here describe the rows, not their precision.
