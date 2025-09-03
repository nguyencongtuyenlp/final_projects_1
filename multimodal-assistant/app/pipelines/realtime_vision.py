"""
Real-Time Vision Pipeline for MultiModal Assistant
🔥 Integrates gesture recognition and eye gaze tracking into API

Features:
- Hand gesture recognition via WebSocket
- Eye gaze tracking via WebSocket  
- Base64 image processing
- Real-time response streaming
"""

import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
from PIL import Image
import time
from collections import deque

class RealTimeVisionPipeline:
    """Real-time vision processing for gestures and eye tracking"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # lower threshold to pick up hands easier
            min_tracking_confidence=0.5
        )
        
        # Face mesh for eye tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices
        self.RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_IRIS = [474, 475, 476, 477]
        
        # Tracking state
        self.ear_history = deque(maxlen=10)
        self.gesture_history = deque(maxlen=5)
        self.blink_counter = 0
        self.total_blinks = 0
        self.session_start = time.time()
        
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image then numpy array
        pil_image = Image.open(BytesIO(img_data))
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        return img_array
        
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=80)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
        
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def classify_gesture(self, landmarks) -> Tuple[str, float]:
        """Classify hand gesture with confidence score"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Key landmark indices
        THUMB_TIP, THUMB_IP = 4, 3
        INDEX_TIP, INDEX_PIP = 8, 6
        MIDDLE_TIP, MIDDLE_PIP = 12, 10
        RING_TIP, RING_PIP = 16, 14
        PINKY_TIP, PINKY_PIP = 20, 18
        
        # Check if fingers are extended
        thumb_extended = points[THUMB_TIP][0] > points[THUMB_IP][0]
        index_extended = points[INDEX_TIP][1] < points[INDEX_PIP][1]
        middle_extended = points[MIDDLE_TIP][1] < points[MIDDLE_PIP][1]
        ring_extended = points[RING_TIP][1] < points[RING_PIP][1]
        pinky_extended = points[PINKY_TIP][1] < points[PINKY_PIP][1]
        
        extended_fingers = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        # Gesture classification
        if extended_fingers <= 1:
            return "Fist", 0.9
        elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return "Thumbs Up", 0.85
        elif index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]):
            return "Peace", 0.8
        elif extended_fingers >= 4:
            return "Open Palm", 0.75
        else:
            return "Unknown", 0.3
            
    def calculate_ear(self, eye_landmarks: List[np.ndarray]) -> float:
        """Calculate Eye Aspect Ratio"""
        eye = np.array(eye_landmarks)
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0
        
    def estimate_gaze_direction(self, eye_landmarks: List[np.ndarray], iris_landmarks: List[np.ndarray]) -> Tuple[str, float]:
        """Estimate gaze direction"""
        eye = np.array(eye_landmarks)
        iris = np.array(iris_landmarks)
        
        eye_left = eye[0]
        eye_right = eye[3]
        eye_center = (eye_left + eye_right) / 2
        iris_center = iris.mean(axis=0)
        
        eye_width = np.linalg.norm(eye_right - eye_left)
        iris_offset_x = (iris_center[0] - eye_center[0]) / eye_width
        
        if iris_offset_x < -0.05:
            return "Right", abs(iris_offset_x)
        elif iris_offset_x > 0.05:
            return "Left", abs(iris_offset_x)
        else:
            return "Center", abs(iris_offset_x)
            
    def process_gesture_recognition(self, base64_image: str) -> Dict[str, Any]:
        """Process image for gesture recognition"""
        try:
            # Convert base64 to image
            image = self.base64_to_image(base64_image)
            h, w = image.shape[:2]
            
            # Optional resize to stabilize detection
            if image.shape[1] > 960:
                scale = 960.0 / image.shape[1]
                image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))

            # Convert to RGB for MediaPipe and mark not writeable for speed
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = self.hands.process(rgb_image)
            rgb_image.flags.writeable = True
            
            # Process results
            gestures = []
            hand_count = 0
            
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Classify gesture
                    gesture, confidence = self.classify_gesture(hand_landmarks.landmark)
                    
                    # Get hand type (left/right)
                    handedness = "Unknown"
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness = results.multi_handedness[i].classification[0].label
                    
                    gestures.append({
                        "hand": handedness,
                        "gesture": gesture,
                        "confidence": confidence
                    })
                    
                    # Draw landmarks on image
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
            
            # Add gesture to history for smoothing
            if gestures:
                self.gesture_history.append(gestures[0]["gesture"])
            
            # Calculate dominant gesture from history
            if self.gesture_history:
                gesture_counts = {}
                for g in self.gesture_history:
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
                dominant_gesture = max(gesture_counts, key=gesture_counts.get)
            else:
                dominant_gesture = "No Hand"
            
            return {
                "type": "gesture_recognition",
                "timestamp": time.time(),
                "hand_count": hand_count,
                "gestures": gestures,
                "dominant_gesture": dominant_gesture,
                "processed_image": self.image_to_base64(image),
                "session_duration": time.time() - self.session_start
            }
            
        except Exception as e:
            return {
                "type": "gesture_recognition",
                "error": str(e),
                "timestamp": time.time()
            }
            
    def process_eye_tracking(self, base64_image: str) -> Dict[str, Any]:
        """Process image for eye gaze tracking"""
        try:
            # Convert base64 to image
            image = self.base64_to_image(base64_image)
            h, w = image.shape[:2]
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            # Default values
            gaze_data = {
                "left_gaze": "Unknown",
                "right_gaze": "Unknown", 
                "blink_detected": False,
                "ear_left": 0.0,
                "ear_right": 0.0,
                "face_detected": False
            }
            
            if results.multi_face_landmarks:
                gaze_data["face_detected"] = True
                
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks to pixel coordinates
                    landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                    
                    # Get eye landmarks
                    left_eye = landmarks[self.LEFT_EYE_EAR]
                    right_eye = landmarks[self.RIGHT_EYE_EAR]
                    left_iris = landmarks[self.LEFT_IRIS]
                    right_iris = landmarks[self.RIGHT_IRIS]
                    
                    # Calculate EAR
                    ear_left = self.calculate_ear(left_eye)
                    ear_right = self.calculate_ear(right_eye)
                    avg_ear = (ear_left + ear_right) / 2
                    
                    # Blink detection
                    self.ear_history.append(avg_ear)
                    smooth_ear = np.mean(list(self.ear_history))
                    blink_detected = smooth_ear < 0.25
                    
                    if blink_detected:
                        self.blink_counter += 1
                        if self.blink_counter > 3:  # Confirm blink
                            self.total_blinks += 1
                            self.blink_counter = 0
                    else:
                        self.blink_counter = 0
                    
                    # Gaze direction
                    left_gaze, _ = self.estimate_gaze_direction(left_eye, left_iris)
                    right_gaze, _ = self.estimate_gaze_direction(right_eye, right_iris)
                    
                    # Update gaze data
                    gaze_data.update({
                        "left_gaze": left_gaze,
                        "right_gaze": right_gaze,
                        "blink_detected": blink_detected,
                        "ear_left": float(ear_left),
                        "ear_right": float(ear_right),
                        "smooth_ear": float(smooth_ear),
                        "total_blinks": self.total_blinks
                    })
                    
                    # Draw eye landmarks
                    for idx in self.LEFT_EYE_EAR + self.RIGHT_EYE_EAR:
                        if idx < len(landmarks):
                            point = landmarks[idx]
                            cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)
                    
                    # Draw iris centers
                    for iris_indices in [self.LEFT_IRIS, self.RIGHT_IRIS]:
                        iris_points = []
                        for idx in iris_indices:
                            if idx < len(landmarks):
                                iris_points.append(landmarks[idx])
                        if iris_points:
                            iris_center = np.mean(iris_points, axis=0).astype(int)
                            cv2.circle(image, tuple(iris_center), 3, (255, 0, 0), -1)
            
            return {
                "type": "eye_tracking",
                "timestamp": time.time(),
                "gaze_data": gaze_data,
                "processed_image": self.image_to_base64(image),
                "session_duration": time.time() - self.session_start
            }
            
        except Exception as e:
            return {
                "type": "eye_tracking", 
                "error": str(e),
                "timestamp": time.time()
            }
            
    def reset_session(self):
        """Reset tracking session"""
        self.ear_history.clear()
        self.gesture_history.clear()
        self.blink_counter = 0
        self.total_blinks = 0
        self.session_start = time.time()
        
        return {
            "type": "session_reset",
            "message": "Tracking session reset successfully",
            "timestamp": time.time()
        }
