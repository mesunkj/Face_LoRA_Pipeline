import cv2
import os

class FaceDetector:
    def __init__(self):
        # Load a pre-trained face detection model, e.g., Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if hasattr(cv2, 'CascadeClassifier'):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None # Mock mode

    def detect_and_crop(self, image_path):
        """
        Detects a face in the image and returns the cropped face image 
        along with its original resolution.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        if self.face_cascade:
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        else:
            # Mock mode: assume the whole image is a face or center crop
            h, w = img.shape[:2]
            faces = [[0, 0, w, h]]
        
        if len(faces) == 0:
            return None, (0, 0) # No face detected
            
        # Assuming the largest face is the target
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add some padding
        padding = int(w * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        cropped_face = img[y1:y2, x1:x2]
        
        # Original width and height of the bounding box
        face_w, face_h = (x2 - x1), (y2 - y1)
        return cropped_face, (face_w, face_h)
