這是一份綜合了我們所有深度討論、完全針對既有專案架構進行「無縫升級」的完整功能需求計畫書 (PRD)。  
我已重新盤點了所有的邊界條件，包含：**兩階段驗收邏輯 (文生圖 \+ Inpainting)、固定種子 (Fixed Seed) 的客觀性、5 大標準靶場的物理定義、底層 VRAM 記憶體隔離機制，以及最關鍵的——嚴格遵守既有目錄與檔案架構 (不憑空創造新設定檔)**。

# **📑 Face LoRA 自動化驗收與黃金引數決策系統：升級需求計畫書 (PRD)**

## **一、 系統升級目標**

本計畫旨在將既有 colab\_runner.ipynb 中的「第 5 階段：獨立評分測試」 升級為工業級的自動化驗收管線。系統將在嚴格鎖定環境相容性的前提下，透過「全域文生圖」與「5 大標準靶圖 Inpainting」兩階段矩陣測試，搭配固定種子控制變因，精準測出該 LoRA 模型在各極端場景下的「黃金引數」，並將決策結果直接寫入既有的註冊表架構中。

## **二、 核心環境與套件版本鎖定 (Environment & Dependencies)**

系統必須在獨立的 Colab 虛擬環境中執行，並嚴格繼承已耗費大量時間驗證的套件清單，杜絕因版本升級造成的權重載入失敗或 OOM 崩潰。

### **1\. 系統環境制約**

* **硬體配置**：Google Colab T4 GPU (16GB VRAM)。  
* **全域控制變因**：系統啟動測試前，必須將生成種子強制鎖定 (例如 fixed\_seed \= 42)，確保後續所有測試矩陣的初始雜訊絕對一致，以達成客觀評估。  
* **基底模型**：統一沿用 runwayml/stable-diffusion-v1-5。  
* **推理步數**：固定為 num\_inference\_steps: 30。

### **2\. 黃金相容性套件清單**

安裝階段必須嚴格指定以下版本號，禁止使用自動升級指令：

* diffusers==0.27.2  
* peft==0.10.0  
* huggingface-hub==0.25.2  
* transformers==4.40.0  
* accelerate==0.30.0  
* insightface  
* onnxruntime-gpu

## **三、 測試資產與標準靶場建置 (Benchmark Suite Assets)**

系統在進行矩陣展開前，必須載入以下靜態資產作為客觀評分的絕對基準。

### **1\. 絕對對照組 (Ground Truth)**

* **來源**：抓取該 LoRA 訓練時所使用的原始圖庫 (約 20 張高畫質照片)。  
* **用途**：作為 InsightFace 計算餘弦相似度 (Cosine Similarity) 的唯一標準答案。

### **2\. 系統級標準靶圖 (Standard Benchmark Images)**

於專案內建立一組固定的測試靶圖，涵蓋 5 大極端物理限制 (解析度需達 1024x1024)：

* **場景 A (標準平光)**：正面半身照，臉部佔比 15%\~25% (預設泛用基準)。  
* **場景 B (超大特寫)**：微距臉部，臉部佔比 40% 以上。  
* **場景 C (全身遠景)**：廣角全身照，臉部佔比 5% 以下。  
* **場景 D (極暗/逆光)**：強烈霓虹燈或背光剪影，臉部佔比 15%\~25%。  
* **場景 E (大側臉)**：90度側臉輪廓，臉部佔比 15%\~25%。

## **四、 兩階段驗收管線與 VRAM 管理 (Evaluation Pipeline)**

嚴格落實「模型主導之分段處理 (Model-Centric Batching)」，產圖階段與評分階段的 VRAM 必須徹底隔離。

### **1\. 階段一：文生圖天花板測試 (Text-to-Image Baseline)**

* **目標**：在無底圖干擾下，測出該 LoRA 特徵記憶的最高極限，並過濾訓練失敗的瑕疵品。  
* **矩陣展開**：使用固定種子，針對 5 大場景之「提示詞」，進行 lora\_scale 的單維度掃描 (例如 \[0.6, 0.8, 1.0, 1.2\])。  
* **防呆中斷**：若此階段產出之最高分數低於及格線 (例如 70%)，系統自動判定模型訓練失敗，終止後續所有測試，節省算力。

### **2\. 階段二：Inpainting 壓力測試 (Inpainting Matrix)**

* **目標**：在 5 張標準靶圖的物理光影與構圖限制下，找出能最完美「逼近真人」的引數。  
* **矩陣展開**：使用固定種子，針對 5 張靶圖進行雙維度交叉掃描：  
  * lora\_scale: \[0.5, 0.7, 0.9, 1.1\]  
  * denoising\_strength: \[0.35, 0.50, 0.65, 0.80\]

### **3\. VRAM 強制釋放與批量評分**

* **解耦執行**：階段一與階段二所有的產圖過程，皆先將圖片寫入硬碟。  
* **記憶體隔離**：產圖完畢後，嚴格執行 del pipe 與 torch.cuda.empty\_cache() 徹底卸載 SD 模型。  
* **批量評分**：載入 InsightFace (buffalo\_l)，對硬碟中所有暫存測試圖與「絕對對照組」進行批量相似度比對。

## **五、 決策產出與既有架構整合 (Deliverables & Integration)**

禁止隨意建立新目錄，所有測試日誌與決策結果必須無縫匯入既有架構。

### **1\. 測試日誌與圖片輸出**

* **目錄規範**：所有測試產生的圖片與詳細 CSV 日誌 (包含每個引數的得分)，必須寫入原專案定義的 EVAL\_OUTPUT\_DIR，即 output/eval/{FACE\_ID}\_standalone/ 路徑下。

### **2\. 黃金引數寫入註冊表 (Registry Update)**

* **檔案規範**：禁止獨立產出 rules.json。  
* **執行邏輯**：系統自動分析日誌中各場景的最高分組合，並以追加屬性 (Metadata) 的方式，直接更新寫入既有的 lora\_registry.json 中。  
* **資料結構範例**：  
  JSON  
  {  
    "face\_id": "TZUYU",  
    "lora\_file": "TZUYU.safetensors",  
    "trigger\_word": "TZUYU",  
    "golden\_rules": {  
      "scene\_standard": {"lora\_scale": 0.9, "denoising": 0.50},  
      "scene\_close\_up": {"lora\_scale": 0.7, "denoising": 0.35},  
      "scene\_wide": {"lora\_scale": 1.1, "denoising": 0.65},  
      "scene\_low\_light": {"lora\_scale": 0.8, "denoising": 0.60},  
      "scene\_profile": {"lora\_scale": 0.6, "denoising": 0.45}  
    }  
  }

* **量產預設值**：未來在 AI\_FaceRes 量產時，若偵測目標圖不屬於極端特徵，則預設讀取 scene\_standard 之黃金引數作為平均解答。