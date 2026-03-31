import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np

try:
    from mtcnn import MTCNN
    detector = MTCNN()
except Exception as e:
    print(f"[ERROR] MTCNN could not be initialized: {e}")
    detector = None

def analyze_face(frame):
    if detector is None:
        print("[DEBUG] MTCNN not available.")
        return 0.0
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        if not results:
            print("[DEBUG] No face detected by MTCNN.")
            return 0.0
            
        # Get landmarks from the primary detected face
        keypoints = results[0]['keypoints']
        
        left_eye = np.array(keypoints['left_eye'])
        right_eye = np.array(keypoints['right_eye'])
        nose_tip = np.array(keypoints['nose'])
        left_mouth = np.array(keypoints['mouth_left'])
        right_mouth = np.array(keypoints['mouth_right'])
        
        # Measure distances from nose tip
        d_left_eye = np.linalg.norm(left_eye - nose_tip)
        d_right_eye = np.linalg.norm(right_eye - nose_tip)
        d_left_mouth = np.linalg.norm(left_mouth - nose_tip)
        d_right_mouth = np.linalg.norm(right_mouth - nose_tip)
        
        # Extremely robust Yaw Face Turn evaluation
        # If distance from one eye to nose is less than 50% of the other eye to nose, they have severely turned their head
        yaw_ratio = min(d_left_eye, d_right_eye) / max(d_left_eye, d_right_eye, 1e-6)
        if yaw_ratio < 0.5:
             print("[DEBUG] Face is heavily turned (Yaw detected). Unreliable. Returning 0.")
             return 0.0
        
        # Asymmetry deviation
        eye_diff = abs(d_left_eye - d_right_eye) / max(d_left_eye, d_right_eye, 1e-6)
        mouth_diff = abs(d_left_mouth - d_right_mouth) / max(d_left_mouth, d_right_mouth, 1e-6)
        
        # Weighted asymmetry (mouth droop is a stronger stroke indicator)
        raw_asymmetry = (eye_diff * 0.3) + (mouth_diff * 0.7)
        
        # Normalize to 0-100 risk score (0.15 diff -> 100 max risk)
        normalized_risk = min((raw_asymmetry / 0.15) * 100.0, 100.0)
        
        print(f"\n[DEBUG FACE] MTCNN asymmetry: Eye={eye_diff:.4f}, Mouth={mouth_diff:.4f} (Yaw Stable: {yaw_ratio:.2f})")
        print(f"[DEBUG FACE] Normalized Face Risk: {normalized_risk:.1f}/100\n")
        return float(normalized_risk)
    except Exception as e:
        print(f"[DEBUG] Face processing failed: {e}")
        return 0.0

def draw_face_box(frame):
    if detector is None:
        return frame
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        for result in results:
            x, y, w, h = result['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Identify the 5 MTCNN keypoints
            for key, point in result['keypoints'].items():
                cv2.circle(frame, point, 3, (0, 255, 255), -1)
        return frame
    except:
        return frame