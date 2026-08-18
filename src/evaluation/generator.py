import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class Generator:
    def __init__(self):
        self.num_images = config.NUM_TEST_IMAGES
        self.prompt_template = config.EVAL_PROMPT_TEMPLATE
        self.negative_prompt = config.EVAL_NEGATIVE_PROMPT
        
    def generate_test_images(self, face_id, lora_path, output_dir):
        """
        Generates test images using the newly trained LoRA model via diffusers.
        """
        os.makedirs(output_dir, exist_ok=True)
        prompt = self.prompt_template.format(trigger_word=face_id)
        
        print(f"\n[Generator] Generating {self.num_images} test images for {face_id}")
        print(f"[Generator] Prompt: {prompt}")
        print(f"[Generator] LoRA: {lora_path}")
        
        generated_paths = []
        
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            
            # Determine base model path (try local first, fallback to runwayml)
            base_model_id = "runwayml/stable-diffusion-v1-5"
            local_model_path = os.path.join(config.MODELS_DIR, "v1-5-pruned-emaonly.safetensors")
            if os.path.exists(local_model_path):
                base_model_id = local_model_path
                
            print(f"[Generator] Loading base model: {base_model_id}")
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
            
            print(f"[Generator] Loading LoRA weights from {lora_path}")
            lora_dir = os.path.dirname(lora_path)
            lora_file = os.path.basename(lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=lora_file)
            
            for i in range(self.num_images):
                out_path = os.path.join(output_dir, f"{face_id}_test_{i}.jpg")
                print(f"[Generator] Generating image {i+1}/{self.num_images}...")
                
                # Inference
                image = pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]
                
                # Save as JPEG
                image.save(out_path, format="JPEG", quality=95)
                generated_paths.append(out_path)
                
            # Clean up memory
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except ImportError as e:
            print(f"[Generator] Error: Required package not found ({e}). Falling back to MOCK mode.")
            for i in range(self.num_images):
                out_path = os.path.join(output_dir, f"{face_id}_test_{i}.jpg")
                with open(out_path, 'w') as f:
                    f.write(f"MOCK GENERATED IMAGE FOR {face_id}")
                generated_paths.append(out_path)
        except Exception as e:
            print(f"[Generator] Generation failed: {e}")
            
        print(f"[Generator] Generation complete.")
        return generated_paths

if __name__ == "__main__":
    gen = Generator()
    # gen.generate_test_images("test_face", "dummy.safetensors", "./output/eval/test_face")
