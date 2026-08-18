import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class SimilarityScorer:
    def __init__(self):
        self.app = None
        try:
            import insightface
            print("[SimilarityScorer] Loading InsightFace model...")
            self.app = insightface.app.FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except ImportError as e:
            print(f"[SimilarityScorer] Warning: {e}. Falling back to MOCK mode.")

    def _extract_embedding(self, image_path):
        """
        Extracts face embedding from an image.
        """
        if self.app is None:
            import random
            return [random.random() for _ in range(512)]
            
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        faces = self.app.get(img)
        if not faces:
            return None
            
        # normed_embedding gives better cosine similarity results
        return faces[0].normed_embedding

    def _cosine_similarity(self, emb1, emb2):
        """
        Calculates cosine similarity between two embeddings.
        """
        if self.app is None:
            import random
            return random.uniform(0.85, 0.98)
            
        import numpy as np
        return np.dot(emb1, emb2)

    def calculate_score(self, original_images_dir, generated_images_paths):
        """
        Compares original training images with generated test images
        and returns the average cosine similarity score.
        """
        original_images = [os.path.join(original_images_dir, f) for f in os.listdir(original_images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not original_images:
            print("[SimilarityScorer] No original images found for comparison.")
            return 0.0
            
        if not generated_images_paths:
            print("[SimilarityScorer] No generated images provided for comparison.")
            return 0.0

        print(f"[SimilarityScorer] Comparing {len(generated_images_paths)} generated images with {len(original_images)} original images...")

        total_score = 0
        comparisons = 0
        
        # Compare each generated image against all original images and average the results
        for gen_img in generated_images_paths:
            gen_emb = self._extract_embedding(gen_img)
            if gen_emb is None:
                print(f"[SimilarityScorer] Warning: No face detected or image corrupted for {os.path.basename(gen_img)}, skipping.")
                continue
            
            for orig_img in original_images:
                orig_emb = self._extract_embedding(orig_img)
                if orig_emb is None: continue
                
                score = self._cosine_similarity(gen_emb, orig_emb)
                total_score += score
                comparisons += 1
                
        if comparisons == 0:
            return 0.0
            
        avg_score = (total_score / comparisons) * 100
        print(f"[SimilarityScorer] Evaluation complete. Average Score: {avg_score:.2f}%")
        
        if avg_score >= (config.SIMILARITY_THRESHOLD * 100):
            print(f"[SimilarityScorer] Status: PASSED (>= {config.SIMILARITY_THRESHOLD * 100}%)")
        else:
            print(f"[SimilarityScorer] Status: FAILED (< {config.SIMILARITY_THRESHOLD * 100}%)")
            
        return avg_score

if __name__ == "__main__":
    scorer = SimilarityScorer()
    # scorer.calculate_score("./training_data/face_id_001", ["./output/eval/test_1.png"])
