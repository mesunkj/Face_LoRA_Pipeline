# 臉部 LoRA 生產線：實戰訓練與部署指南 (Google Colab 篇)

本指南將手把手教您如何在 Google Drive 與 Colab 環境中完成 `kohya_ss` 的安裝，並解除系統目前的「模擬 (Mock)」，真正啟動 LoRA 模型訓練。

## 階段一：準備 Kohya_ss 訓練環境

### 步驟 1：在 Google Colab 安裝 Kohya_ss
我們將把 `kohya_ss` 程式碼直接下載到您的 Google Drive 專案目錄下，這樣以後就不必重複下載腳本了。
但是請注意：**由於 Colab 每次開啟都是一台全新的虛擬主機，因此 Python 相依套件 (`pip install`) 還是必須每次執行。**

請在您 Google Drive 中的 `colab_runner.ipynb` 筆記本裡，確認我們已經幫您準備好的這段程式碼：

```python
# 將 kohya_ss 腳本直接下載到 Google Drive 中
import os
KOHYA_DIR = os.path.join(PROJECT_DIR, "kohya_ss")

if not os.path.exists(KOHYA_DIR):
    print("尚未下載 kohya_ss，開始從 GitHub 複製...")
    !git clone --recursive https://github.com/bmaltais/kohya_ss.git {KOHYA_DIR}
else:
    print("已在 Google Drive 中找到 kohya_ss 腳本，跳過下載步驟。")

# 切換至 kohya_ss 目錄並安裝相依套件 (因為 Colab 虛擬機是全新的，套件必須每次重裝)
%cd {KOHYA_DIR}

# 確保子模組 (如 sd-scripts) 已正確初始化並更新
!git submodule update --init --recursive

!pip install -r requirements.txt
!pip install accelerate transformers diffusers

# 切換回專案目錄
%cd {PROJECT_DIR}
```

### 步驟 2：設定路徑 (config.py)
因為我們已經將 `kohya_ss` 裝在專案根目錄底下了，所以 `config.py` 中的預設路徑：
```python
KOHYA_DIR = os.path.join(BASE_DIR, "kohya_ss")
```
**已經完全正確，您無需做任何修改！**

---

## 階段二：解除封印 (啟動真正的訓練功能)

目前的系統在 `src/training/kohya_runner.py` 內僅為模擬運行，若要執行實際訓練，請依照以下步驟修改程式碼。

### 步驟 1：修改 `kohya_runner.py` 指令陣列
開啟 `src/training/kohya_runner.py`，找到 `train` 函數（大約第 45 行處）。
我們需要將原本呼叫 `mock_train_network.py` 的地方，換成真實的指令。

請將原本的 `command` 陣列替換為：

```python
        # 組裝實際的 Kohya 執行指令 (透過 accelerate 啟動)
        command = [
            "accelerate", "launch", 
            "--num_cpu_threads_per_process", "2", 
            self.train_script, # 這裡會指向真正的 train_network.py
            "--pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5",
            "--train_data_dir", dataset_dir,
            "--output_dir", output_dir,
            "--output_name", output_name,
            "--max_train_steps", str(steps),
            "--learning_rate", str(config.LEARNING_RATE),
            "--train_batch_size", str(config.TRAIN_BATCH_SIZE),
            "--network_module", "networks.lora",
            "--network_dim", str(config.NETWORK_DIM),
            "--network_alpha", str(config.NETWORK_ALPHA),
            "--save_every_n_epochs", str(config.SAVE_EVERY_N_EPOCHS),
            "--mixed_precision", "fp16",
            "--save_precision", "fp16"
        ]
```

### 步驟 2：執行 subprocess 並拿掉假檔案生成
往下捲動幾行，原本程式會寫入一個 `DUMMY LORA MODEL DATA` 檔案，請將那段「建立假檔案」的程式碼刪除，替換為實際執行指令的邏輯：

```python
        print(f"[KohyaRunner] 準備執行指令:\n{' '.join(command)}")
        print(f"[KohyaRunner] 開始實際訓練 {face_id}...")
        
        # 實際呼叫訓練進程
        try:
            subprocess.run(command, check=True)
            print(f"[KohyaRunner] {face_id} 訓練完成！")
        except subprocess.CalledProcessError as e:
            print(f"[KohyaRunner] 訓練發生錯誤: {e}")
            return None
        
        final_model_path = os.path.join(output_dir, f"{output_name}.safetensors")
        return final_model_path
```

---

## 階段三：執行完整生產線

1. **準備素材：**
   確保在 Google Drive 的 `Face_LoRA_Pipeline/training_data/` 目錄下，已經建立好您的模特兒資料夾（例如 `Model_A/`），並放入 10-20 張清晰無遮擋的臉部圖片。
   
2. **一鍵執行：**
   在 `colab_runner.ipynb` 中，依序執行所有的儲存格。系統會：
   - 掃描 `Model_A/` 裡面的圖片並進行質檢。
   - 動態計算適合的訓練步數 (max_train_steps)。
   - 將訓練參數與素材拋給 `kohya_ss` 開始真正的模型訓練。
   - 訓練產出的模型會自動儲存到您 Google Drive 的 `Face_LoRA_Pipeline/output/models/Model_A.safetensors`。
