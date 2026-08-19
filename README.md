# 自動化 Face LoRA 生產線 (Automated Face LoRA Pipeline)

這是一個全自動的批次處理引擎（Batch Pipeline），專為臉部 LoRA 模型的訓練與驗證所設計。系統能夠自動完成圖片質檢、解析度放大、自動打標、呼叫 Kohya_ss 進行訓練，並最終透過「兩階段驗收管線」與 InsightFace 計算相似度，自動找出該模型在各場景的「黃金引數 (Golden Rules)」。

### 📈 評估結果分析
- 系統會透過 InsightFace 計算人臉的餘弦相似度 (Cosine Similarity)。
- 在數學理論上，即便是同一個人的不同照片，分數通常會落在 **0.45 ~ 0.60 (45% ~ 60%)** 之間。因此 `config.py` 中預設的合格門檻為 **50%**，若產出分數大於 50%，即代表模型具有極高的人物還原度。
- 若產出的測試圖片沒有偵測到人臉或圖像過於扭曲，系統會自動跳過該張圖片的評估，並在日誌中顯示警告，不影響整體平均分數的計算。

---

## 🛠️ 開發與環境排錯紀錄 (Troubleshooting & Dependency Hell)

在開發與實際於 Google Colab 部署的過程中，我們遇到並解決了數個棘手的套件衝突問題。這些問題已被紀錄並修復，建議未來的開發者嚴格遵守以下套件版本，以免再次觸發 Bug：

1. **Torch 與 CUDA 執行階段錯誤 (`AssertionError: Torch not compiled with CUDA enabled`)**
   - **原因**：Colab 預設分配 CPU 資源，而產圖程式強制調用 GPU (`.to("cuda")`)。
   - **解法**：強制確保 Colab 執行階段已切換為 **T4 GPU**，並且程式碼中已加入 `torch.cuda.is_available()` 進行防呆降級機制。

2. **HuggingFace 驗證錯誤 (`HFValidationError`)**
   - **原因**：當 `diffusers` 使用完整檔案路徑呼叫 `load_lora_weights`，且未指定 `weight_name` 時，底層 API 會誤認路徑為雲端 Repo ID 並報錯。
   - **解法**：在程式碼中將路徑拆分為 `lora_dir` 與 `lora_file`，並明確傳入 `weight_name` 參數。同時必須確保 Colab 已正確掛載 Google Drive。

3. **PEFT 載入錯誤 (`IndexError: list index out of range` in `get_peft_kwargs`)**
   - **原因**：最新版的 `diffusers` (v0.30+) 與 `peft` (v0.12+) 對於 Kohya 訓練的 Text Encoder 權重解析非常嚴格，遇到些微不匹配便會擷取空清單並導致崩潰。
   - **解法**：將 `diffusers` 與 `peft` 鎖定在最穩定的黃金版本（見下方套件清單）。

4. **HuggingFace API 棄用錯誤 (`ImportError: cannot import name 'cached_download'`)**
   - **原因**：穩定版的 `diffusers==0.27.2` 依賴了舊版 `huggingface-hub` 中的 `cached_download` 函數，但 Colab 環境預設將 `huggingface-hub` 更新至 `0.26+` 移除了該函數。
   - **解法**：強制將 `huggingface-hub` 降級至 `0.25.2`。

5. **Google Drive 頻繁斷線錯誤 (`Transport endpoint is not connected`)**
   - **原因**：當 Colab 直接在掛載的 Google Drive (`/content/drive/MyDrive/...`) 上進行大量密集小檔案讀寫（如 `pip install` 與模型訓練）時，極易觸發 Google 的安全機制，導致雲端硬碟掛載點強制中斷。
   - **解法**：修改系統架構，將 `kohya_ss` 訓練環境強制安裝於 Colab 虛擬機本機端（`/content/kohya_ss`）。藉由犧牲每次重啟約 3 秒的下載時間，徹底根除了環境崩潰的地雷。

6. **嚴苛的靶圖配對導致階段二被忽略 (Benchmark Filename Matching)**
   - **原因**：腳本原先以極度嚴格的規則（區分大小寫、限用底線）搜尋 `benchmark_images/` 內的靶圖。若使用者以 `Wide Shot.jpg` 直覺命名，系統會因配對失敗而默默略過最重要的 Inpainting 測試。
   - **解法**：導入「全自動寬容比對邏輯」，系統會自動將檔名轉小寫，並將空格與連字號視為底線處理。確保使用者的直覺命名皆能 100% 被精準捕捉。

7. **過度防呆導致的強制終止 (`sys.exit` Bug)**
   - **原因**：原設計在階段一分數若低於 70% 則強制 `sys.exit()` 以節省算力。但在小樣本訓練中此條件過於嚴苛，反倒剝奪了使用者查看後續 Inpainting 壓力測試與 CSV 完整數據報表的權益。
   - **解法**：全面拔除強制終止機制，改為「黃色警告但繼續執行」。保證無論分數高低，系統皆會完美走完流程並產出報告。

8. **導入「起飛前環境檢查 (Pre-flight Check)」機制**
   - **原因**：過往若使用者忘記更改 `FACE_ID` 或少放靶圖，系統仍會先耗費數分鐘下載百 MB 的相依套件與模型後才報錯崩潰，嚴重浪費時間。
   - **解法**：在 Colab 自動化驗收管線的**最頂端**植入嚴格檢查清單。在下載任何套件前，先行確認模型存在與靶圖對齊，並透過 `input()` 交由使用者確認後放行，將掌控權徹底還給使用者。

### 📦 黃金穩定版套件清單
在進行獨立測試或重新部署環境時，強烈建議使用以下經過驗證的套件組合：
```bash
pip install -q diffusers==0.27.2 peft==0.10.0 transformers==4.40.0 accelerate==0.30.0 insightface onnxruntime-gpu huggingface-hub==0.25.2
```

---

## 🤝 貢獻指南

歡迎發起 Pull Request！在進行開發前，請注意：
- 測試資料夾 `training_data/`、產出資料夾 `output/` 及所有生成的模型檔 `*.safetensors` 已被加入 `.gitignore`，請勿上傳真實訓練資料。
- 專案根目錄下的 `lora_registry.json` 為測試用紀錄，請在提交前清空或保留為空陣列。

## 🌟 核心特色

1. **自動化質檢與放大 (QC & Pre-processing)**: 自動偵測臉部範圍並進行裁切，對於低解析度（低於 512x512）的圖片，支援呼叫 R-GAN (如 Real-ESRGAN) 演算法自動「腦補」放大。
2. **自動化訓練 (Automated Training)**: 根據輸入圖片數量動態計算合適的 `max_train_steps`，並封裝指令呼叫 Kohya 的 `train_network.py` 進行訓練。
3. **註冊表紀錄 (Logging & Registry)**: 所有的訓練結果（包括使用的參數與是否經過 R-GAN 處理）將被彙整寫入 `lora_registry.json`，方便後續應用程式呼叫。
4. **自動科學驗證與黃金引數決策 (Automated Evaluation & Golden Rules)**: 模型訓練完成後，會自動執行「文生圖天花板測試」與「Inpainting 壓力測試」兩階段驗收，並利用 InsightFace 計算相似度，將找出各場景的黃金引數回寫至註冊表。
5. **雲端與地端雙支援**: 系統會自動偵測執行環境。無論是在本機端，或是打包丟上 Google Colab，都能無縫切換路徑繼續執行。

## 📂 專案架構

```
Face_LoRA_Pipeline/
├── config.py                 # 全局設定檔（包含路徑、解析度、訓練參數等）
├── main.py                   # 系統執行主程式
├── colab_runner.ipynb        # 專門供 Google Colab 執行的筆記本
├── lora_registry.json        # 自動生成的模型註冊表 (執行後產生)
├── training_data/            # 放置原圖的地方 (例如：/face_id_001/)
├── output/                   # 系統產出目錄 (執行後產生)
│   ├── preprocessed_data/    # 處理並放大裁切後的訓練用圖片
│   ├── models/               # 訓練完成的 LoRA 模型 (.safetensors)
│   └── eval/                 # 驗證階段自動生成的測試圖片
└── src/
    ├── preprocessing/        # 階段一：素材質檢與標準化
    │   ├── data_qc.py
    │   ├── face_detector.py
    │   ├── auto_tagger.py
    │   └── r_gan_upscaler.py
    ├── training/             # 階段二：Kohya 核心訓練引擎
    │   └── kohya_runner.py
    ├── registry/             # 階段三：系統註冊表記錄
    │   └── logger.py
    └── evaluation/           # 階段四：科學相似度驗證
        ├── generator.py
        └── similarity_scorer.py
```

## 🛠 環境與套件需求

- Python 3.10+
- `opencv-python`
- `insightface`
- (可選) 需自行準備 Real-ESRGAN 模型與執行檔供放大使用
- (可選) 需自行安裝 [Kohya_ss](https://github.com/bmaltais/kohya_ss) 供實體模型訓練

## 🚀 使用說明

### 方式一：在本地端執行 (Local Execution)

1. 將欲訓練的目標圖片放入 `training_data/` 目錄中。請以目標名稱 `{FACE_ID}` 作為資料夾名稱，例如：`training_data/face_id_001/` 裡面放 face_id_001 的照片。
2. 根據需求修改 `config.py` 中的參數（例如：開關 R-GAN、調整 Learning Rate）。
3. 執行主程式：
   ```bash
   python main.py
   ```
4. 執行完畢後，產出的 LoRA 模型將會放在 `output/models/`，而詳細資訊與相似度評分會記錄在根目錄的 `lora_registry.json` 中。

### 📋 進階與實戰 (TODOs)
若您準備好解除本系統的「模擬 (Mock)」狀態，並開始進行真實的模型訓練，請務必參閱完整的實戰教材：
👉 **[臉部 LoRA 生產線：實戰訓練與部署指南 (TRAINING_TUTORIAL.md)](./TRAINING_TUTORIAL.md)**

### 方式二：在 Google Colab 執行 (Cloud Execution)

為了方便沒有 GPU 的使用者，本專案已完全支援 Google Colab。

1. 將整個專案資料夾上傳到您的 Google Drive 的根目錄，並確保資料夾名稱為 `Face_LoRA_Pipeline`。
2. 將圖片放進雲端硬碟中的 `Face_LoRA_Pipeline/training_data/` 內。
3. 在 Google Drive 中對著 `colab_runner.ipynb` 點擊右鍵 -> 選擇用 **Google Colaboratory** 開啟。
4. 依序執行筆記本內的各個區塊，即可在雲端完成所有訓練與驗證。

## ⚠️ 注意事項與限制 (Reality Check)

- **R-GAN 的腦補現象**：如果您的原圖臉部小於 512x512，系統強行放大所產生的毛孔與睫毛等細節，是由 AI「猜測」出來的。LoRA 雖然會 100% 還原這些細節，但與肉眼真實長相可能會有落差。高清晰度的原始素材仍是達到極高相似度的王道。
- **硬體效能要求**：階段四的產圖與評估模組已全面實作真實的 Stable Diffusion 與 InsightFace 演算法。請務必確保您的執行環境（包含 Google Colab）有成功掛載 GPU，否則純 CPU 運算將耗費極長的時間。
