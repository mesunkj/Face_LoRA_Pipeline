import os
import subprocess
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class KohyaRunner:
    def __init__(self):
        self.kohya_dir = config.KOHYA_DIR
        self.train_script = os.path.join(self.kohya_dir, "sd-scripts", "train_network.py")
        
    def calculate_steps(self, num_images):
        """
        Dynamically calculate max_train_steps based on the number of images.
        Rule of thumb: steps = (num_images * repeats * epochs) / batch_size
        """
        repeats = 10
        epochs = 10
        steps = math.ceil((num_images * repeats * epochs) / config.TRAIN_BATCH_SIZE)
        
        # Cap steps or set minimum if needed
        steps = max(500, min(steps, config.MAX_TRAIN_STEPS))
        return steps
        
    def train(self, face_id, dataset_dir):
        """
        Executes the Kohya SD 1.5 LoRA training script.
        """
        output_name = face_id
        output_dir = os.path.join(config.MODELS_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        num_images = sum(len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) for _, _, files in os.walk(dataset_dir))
        if num_images == 0:
            print(f"[KohyaRunner] No images found for {face_id} in {dataset_dir}. Skipping.")
            return None
            
        steps = self.calculate_steps(num_images)
        print(f"\n[KohyaRunner] Starting training for {face_id} with {num_images} images. Total steps: {steps}")
        
        # 下載單一 safetensors 模型檔以節省記憶體轉換消耗
        # 【關鍵修復】：正確偵測 Colab 環境 (因為 !python main.py 不會有 google.colab 模組)
        if "/content/" in config.MODELS_DIR.replace("\\", "/"):
            base_model_dir = "/content/models"
        else:
            base_model_dir = config.MODELS_DIR
        os.makedirs(base_model_dir, exist_ok=True)
        
        base_model_path = os.path.join(base_model_dir, "v1-5-pruned-emaonly.safetensors")
        if not os.path.exists(base_model_path):
            print(f"[KohyaRunner] Downloading base model to {base_model_path} (this may take a minute)...")
            base_model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
            # 跨平台相容的下載方式
            import urllib.request
            urllib.request.urlretrieve(base_model_url, base_model_path)
            print("[KohyaRunner] Base model download complete.")
        
        # 組裝實際的 Kohya 執行指令 (透過 accelerate 啟動)
        command = [
            "accelerate", "launch", 
            "--num_cpu_threads_per_process", "2",
            "--mixed_precision", "fp16",
            self.train_script, # 這裡會指向真正的 train_network.py
            "--pretrained_model_name_or_path", base_model_path, # 改用單一 safetensors 檔案
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
            "--save_precision", "fp16",
            "--gradient_checkpointing",
            "--lowram",
            "--cache_latents",
            "--optimizer_type", "AdamW",
            "--max_data_loader_n_workers", "0",
            "--console_log_simple",
            "--resolution", f"{config.TARGET_FACE_RESOLUTION},{config.TARGET_FACE_RESOLUTION}"
        ]
        
        print(f"[KohyaRunner] 準備執行指令:\n{' '.join(command)}")
        print(f"[KohyaRunner] 開始實際訓練 {face_id}...")
        
        # 釋放主程式佔用的 CPU 與 GPU 記憶體 (如 DataQC 殘留的 InsightFace 模型)
        # 這可以擠出大約 1~2GB 的空間，防止 Colab 的系統嚴格記憶體限制 (OOM Killer) 瞬間擊殺子進程
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 設定環境變數，強制關閉 TensorFlow 與 JAX 的載入，避免 Colab 內建的 TF 與 protobuf 版本衝突
        run_env = os.environ.copy()
        run_env["USE_TF"] = "0"
        run_env["USE_JAX"] = "0"
        
        # 實際呼叫訓練進程
        try:
            subprocess.run(command, check=True, env=run_env)
            print(f"[KohyaRunner] {face_id} 訓練完成！")
        except subprocess.CalledProcessError as e:
            print(f"[KohyaRunner] 訓練發生錯誤: {e}")
            return None
        
        final_model_path = os.path.join(output_dir, f"{output_name}.safetensors")
        return final_model_path

if __name__ == "__main__":
    # Test script locally
    runner = KohyaRunner()
    # runner.train("test_face", "./output/preprocessed_data/test_face")
