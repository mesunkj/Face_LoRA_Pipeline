#@title 6. 自動化驗收與黃金引數決策系統 (兩階段驗收管線)
# 此區塊會自動執行文生圖天花板測試與 Inpainting 壓力測試，並更新 lora_registry.json
# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import sys

# 6.0 參數與路徑定義 (起飛前檢查)
FACE_ID = "Tzuyu"  # ⚠️請替換為您剛剛訓練的臉部名稱
PROJECT_DIR = "/content/drive/MyDrive/app/AI/Lora/Face_LoRA_Pipeline"
MODEL_PATH = f"{PROJECT_DIR}/output/models/{FACE_ID}.safetensors"
BENCHMARK_IMAGES_DIR = f"{PROJECT_DIR}/benchmark_images"

print(f"=== 🛫 起飛前環境與檔案檢查 ===")
print(f"1. 測試對象 (FACE_ID): {FACE_ID}")
if not os.path.exists(MODEL_PATH):
    print(f"❌ 錯誤: 找不到訓練好的模型 {MODEL_PATH}")
    print("請確認 FACE_ID 是否正確，或是否尚未完成訓練！")
    sys.exit("檢查失敗，終止執行。")
else:
    print(f"✅ 找到模型: {MODEL_PATH}")

print("2. 檢查 benchmark_images 靶圖...")
if os.path.exists(BENCHMARK_IMAGES_DIR):
    files = [f for f in os.listdir(BENCHMARK_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   目前資料夾內共有 {len(files)} 張圖檔。")
    scenes = ['standard', 'close_up', 'wide', 'low_light', 'profile']
    matched = 0
    for s in scenes:
        if any(s in f.lower().replace(' ', '_').replace('-', '_') for f in files):
            matched += 1
    print(f"   成功辨識出 {matched}/5 個場景的標準靶圖。")
else:
    print(f"⚠️ 警告: 找不到 {BENCHMARK_IMAGES_DIR}，系統將自動略過 Inpainting 壓力測試。")

print("====================================")
confirm = input("⚠️ 請確認以上變數與檔案皆已到位。\n(按 Enter 繼續執行，或輸入 'q' 取消): ")
if confirm.lower() == 'q':
    sys.exit("使用者取消執行。")
print("檢查通過，開始安裝相依套件與載入模型 (需要幾分鐘，請稍候)...\n")

# 6.1 安裝黃金相容性套件
from google.colab import drive
drive.mount('/content/drive')
!pip install -q diffusers==0.27.2 peft==0.10.0 transformers==4.40.0 accelerate==0.30.0 insightface onnxruntime-gpu huggingface-hub==0.25.2

import json
import uuid
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import insightface
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from google.colab.patches import cv2_imshow

# 6.2 剩餘路徑定義
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
ORIGINAL_IMAGES_DIR = f"{PROJECT_DIR}/training_data/{FACE_ID}"
EVAL_OUTPUT_DIR = f"{PROJECT_DIR}/output/eval/{FACE_ID}_standalone"
REGISTRY_FILE = f"{PROJECT_DIR}/lora_registry.json"


PROMPTS = {
    "scene_standard": f"A portrait of {FACE_ID}, raw photo, highly detailed, 8k uhd, dslr",
    "scene_close_up": f"A macro close-up of {FACE_ID}'s face, highly detailed, 8k uhd, dslr",
    "scene_wide": f"A wide angle shot of {FACE_ID}, full body, highly detailed, 8k uhd, dslr",
    "scene_low_light": f"A portrait of {FACE_ID} in low light neon cyberpunk, highly detailed, 8k uhd",
    "scene_profile": f"A side profile portrait of {FACE_ID}, highly detailed, 8k uhd, dslr"
}
NEGATIVE_PROMPT = "blurry, out of focus, disfigured, low quality, bad anatomy"

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
FIXED_SEED = 42

# 初始化 InsightFace
print("\n載入 InsightFace 評分模型中...")
app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    faces = app.get(img)
    return faces[0].normed_embedding if faces else None

print("載入 Ground Truth Embeddings...")
original_images = [os.path.join(ORIGINAL_IMAGES_DIR, f) for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
gt_embeddings = [emb for f in original_images if (emb := get_embedding(f)) is not None]

def calculate_score(gen_path):
    gen_emb = get_embedding(gen_path)
    if gen_emb is None:
        print(f"⚠️ 警告：產出圖片代號 [{os.path.splitext(os.path.basename(gen_path))[0]}] (檔名: {os.path.basename(gen_path)}) 未偵測到人臉，給予 0 分。")
        return 0.0
    if len(gt_embeddings) == 0:
        print("⚠️ 警告：找不到 Ground Truth 特徵向量，無法進行評分。")
        return 0.0
    scores = [np.dot(gen_emb, gt) for gt in gt_embeddings]
    return float(np.mean(scores) * 100)

def add_watermark_cv2(img_cv2, text):
    # 使用 cv2 加上浮水印，位於右下角
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    margin = 20
    x = img_cv2.shape[1] - text_size[0] - margin
    y = img_cv2.shape[0] - margin
    
    # 加入黑色半透明背景框，增加白字辨識度
    overlay = img_cv2.copy()
    cv2.rectangle(overlay, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img_cv2, 1 - alpha, 0, img_cv2)
    
    # 壓印白色字體
    cv2.putText(img_cv2, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return img_cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到您的 LoRA 模型：{MODEL_PATH}，請確認檔名與路徑是否正確！")

# ==========================================
# 階段一：文生圖天花板測試 (Text-to-Image Baseline)
# ==========================================
print("\n=== 階段一：文生圖天花板測試 ===")
pipe_t2i = AutoPipelineForText2Image.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None).to(device)
lora_dir = os.path.dirname(MODEL_PATH)
lora_file = os.path.basename(MODEL_PATH)
pipe_t2i.load_lora_weights(lora_dir, weight_name=lora_file)

lora_scales_t2i = [0.6, 0.8, 1.0, 1.2]
t2i_results = []
max_t2i_score = 0.0

for scene, prompt in PROMPTS.items():
    for scale in lora_scales_t2i:
        generator = torch.Generator(device=device).manual_seed(FIXED_SEED)
        
        # 產生獨一無二的隨機代號 (8位數英數字)
        identifier = f"T2I_{uuid.uuid4().hex[:8].upper()}"
        out_path = os.path.join(EVAL_OUTPUT_DIR, f"{identifier}.jpg")
        
        img = pipe_t2i(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": scale}, generator=generator).images[0]
        
        # 轉換為 cv2 格式加入浮水印後存檔
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv2 = add_watermark_cv2(img_cv2, identifier)
        cv2.imwrite(out_path, img_cv2)
        
        score = calculate_score(out_path)
        t2i_results.append({
            "identifier": identifier,
            "scene": scene,
            "prompt": prompt,
            "lora_scale": scale,
            "denoising": None,
            "score": score
        })
        max_t2i_score = max(max_t2i_score, score)
        print(f"[{scene}] Scale: {scale} -> ID: {identifier} -> Score: {score:.2f}%")

del pipe_t2i
torch.cuda.empty_cache()

if max_t2i_score < 70.0:
    print(f"⚠️ 警告：階段一最高相似度僅為 {max_t2i_score:.2f}% (低於 70%)。但系統將忽略此警告，繼續執行後續測試。\n")
    import sys
    # sys.exit()

# ==========================================
# 階段二：Inpainting 壓力測試 (Inpainting Matrix)
# ==========================================
print("\n=== 階段二：Inpainting 壓力測試 ===")
inpaint_results = []

if not os.path.exists(BENCHMARK_IMAGES_DIR) or len(os.listdir(BENCHMARK_IMAGES_DIR)) == 0:
    print(f"⚠️ 警告：找不到 {BENCHMARK_IMAGES_DIR} 目錄或目錄為空，跳過 Inpainting 測試。")
else:
    pipe_inp = AutoPipelineForInpainting.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None).to(device)
    pipe_inp.load_lora_weights(lora_dir, weight_name=lora_file)
    
    lora_scales_inp = [0.5, 0.7, 0.9, 1.1]
    denoising_strengths = [0.35, 0.50, 0.65, 0.80]
    
    # 建立一個全白的 Dummy Mask 預設全圖替換
    mask_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    
    for scene, prompt in PROMPTS.items():
        # 尋找對應的靶圖
        search_term = scene.replace('scene_', '').replace('_', '')
        target_img_files = [f for f in os.listdir(BENCHMARK_IMAGES_DIR) if search_term in f.lower().replace(' ', '').replace('-', '')]
        if not target_img_files:
            print(f"⚠️ 警告：找不到場景 '{scene}' 的標靶圖，跳過此場景所有 Inpainting 測試 (無產出圖檔代號)！")
            continue
        target_img_path = os.path.join(BENCHMARK_IMAGES_DIR, target_img_files[0])
        init_image = Image.open(target_img_path).convert("RGB").resize((1024, 1024))
        
        for scale in lora_scales_inp:
            for denoise in denoising_strengths:
                generator = torch.Generator(device=device).manual_seed(FIXED_SEED)
                
                # 產生隨機代號
                identifier = f"INP_{uuid.uuid4().hex[:8].upper()}"
                out_path = os.path.join(EVAL_OUTPUT_DIR, f"{identifier}.jpg")
                
                img = pipe_inp(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, image=init_image, mask_image=mask_image, num_inference_steps=30, strength=denoise, guidance_scale=7.5, cross_attention_kwargs={"scale": scale}, generator=generator).images[0]
                
                # 壓印浮水印
                img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img_cv2 = add_watermark_cv2(img_cv2, identifier)
                cv2.imwrite(out_path, img_cv2)
                
                score = calculate_score(out_path)
                inpaint_results.append({
                    "identifier": identifier,
                    "scene": scene,
                    "prompt": prompt,
                    "lora_scale": scale,
                    "denoising": denoise,
                    "score": score
                })
                print(f"[{scene}] Scale: {scale}, Denoise: {denoise} -> ID: {identifier} -> Score: {score:.2f}%")

    del pipe_inp
    torch.cuda.empty_cache()

# ==========================================
# 儲存日誌與更新註冊表
# ==========================================
print("\n=== 彙整黃金引數與更新註冊表 ===")
df = pd.DataFrame(t2i_results + inpaint_results)
df.to_csv(os.path.join(EVAL_OUTPUT_DIR, "evaluation_matrix.csv"), index=False)

golden_rules = {}
for scene in PROMPTS.keys():
    scene_df = df[df['scene'] == scene]
    if not scene_df.empty:
        best_row = scene_df.loc[scene_df['score'].idxmax()]
        golden_rules[scene] = {
            "best_identifier": best_row['identifier'],
            "lora_scale": float(best_row['lora_scale']),
            "denoising": float(best_row.get('denoising', 0.50)) if 'denoising' in best_row and not pd.isna(best_row.get('denoising')) else 0.50,
            "max_score": float(best_row['score'])
        }

if os.path.exists(REGISTRY_FILE):
    with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    for face in registry.get('faces', []):
        if face.get('face_id') == FACE_ID:
            face['golden_rules'] = golden_rules
            break
            
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)
    print(f"已成功將黃金引數與最佳代號寫入 {REGISTRY_FILE} !")
else:
    print(f"⚠️ 警告：找不到註冊表 {REGISTRY_FILE}，無法寫入。")
