#!/usr/bin/env python3
"""
Fixed Face Recognition System
Resolves compatibility issues and ensures proper face matching
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import traceback

# Import libraries with fallbacks
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("❌ face_recognition library not available")

try:
    import mediapipe as mp
    import tensorflow as tf
    MEDIAPIPE_TF_AVAILABLE = True
    print("✅ MediaPipe and TensorFlow available")
except ImportError:
    MEDIAPIPE_TF_AVAILABLE = False
    print("❌ MediaPipe/TensorFlow not available")


class AdvancedFaceRecognitionSystem:
    """
    Advanced Face Recognition System with multi-modal detection
    Supports multiple encoding types and detection methods for optimal accuracy
    """
    
    def __init__(self):
        print("🚀 Initializing Advanced Face Recognition System...")
        
        # Initialize detection systems
        self.opencv_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_TF_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.3
            )
            self.mediapipe_available = True
        else:
            self.mediapipe_available = False
        
        # Initialize FaceNet model if available
        self.facenet_model = None
        self.tf_session = None
        self.facenet_available = False
        self._load_facenet_model()
        
        print("✅ Advanced Face Recognition System initialized!")
    
    def _load_facenet_model(self):
        """Load FaceNet model if available"""
        try:
            model_paths = [
                "facenet/facenet.pb",
                "detectionThroughApi/facenet/facenet.pb",
                "models/facenet.pb"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"📁 Loading FaceNet model from: {model_path}")
                    
                    # Load TensorFlow graph
                    graph = tf.Graph()
                    with graph.as_default():
                        graph_def = tf.compat.v1.GraphDef()
                        with tf.io.gfile.GFile(model_path, "rb") as f:
                            graph_def.ParseFromString(f.read())
                            tf.import_graph_def(graph_def, name="")
                    
                    self.tf_session = tf.compat.v1.Session(graph=graph)
                    self.facenet_model = graph
                    self.facenet_available = True
                    print("✅ FaceNet model loaded successfully!")
                    return
            
            print("⚠ FaceNet model not found - using alternative methods")
            
        except Exception as e:
            print(f"⚠ Could not load FaceNet model: {str(e)}")
    
    def detect_faces(self, image):
        """Detect faces using the best available method"""
        best_faces = []
        
        # Try MediaPipe first (most accurate and fast)
        if self.mediapipe_available:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_image)
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = image.shape
                        
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        
                        best_faces.append([x, y, w, h])
                    
                    if len(best_faces) > 0:
                        return best_faces
            except Exception as e:
                print(f"⚠ MediaPipe detection failed: {str(e)}")
        
        # Try face_recognition library
        if FACE_RECOGNITION_AVAILABLE:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                faces = []
                for (top, right, bottom, left) in face_locations:
                    x, y, w, h = left, top, right - left, bottom - top
                    faces.append([x, y, w, h])
                
                if len(faces) > 0:
                    return faces
            except Exception as e:
                print(f"⚠ face_recognition detection failed: {str(e)}")
        
        # Fallback to OpenCV
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try multiple parameters for better detection
            params = [(1.1, 4), (1.2, 3), (1.05, 5)]
            
            for scale_factor, min_neighbors in params:
                faces = self.opencv_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbors,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    return faces.tolist()
            
            print("⚠ No faces detected with any method")
            return []
            
        except Exception as e:
            print(f"❌ All face detection methods failed: {str(e)}")
            return []
    
    def extract_face_encoding(self, image, face_coords, method='auto'):
        """
        Extract face encoding using the best available method
        Returns encoding with metadata about the method used
        """
        x, y, w, h = face_coords
        
        # Ensure coordinates are valid
        ih, iw = image.shape[:2]
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = max(1, min(w, iw - x))
        h = max(1, min(h, ih - y))
        
        face_roi = image[y:y+h, x:x+w]
        
        # Try face_recognition library first (128 dimensions)
        if FACE_RECOGNITION_AVAILABLE and method in ['auto', 'face_recognition']:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = [(y, x + w, y + h, x)]
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if len(face_encodings) > 0:
                    encoding = face_encodings[0].tolist()
                    print(f"✅ face_recognition encoding: {len(encoding)} dimensions")
                    return {
                        'encoding': encoding,
                        'method': 'face_recognition',
                        'dimensions': len(encoding)
                    }
            except Exception as e:
                print(f"⚠ face_recognition encoding failed: {str(e)}")
        
        # Try FaceNet model (512 dimensions) - maintain compatibility with existing data
        if self.facenet_available and method in ['auto', 'facenet']:
            try:
                encoding = self._extract_facenet_features(face_roi)
                if encoding:
                    print(f"✅ FaceNet encoding: {len(encoding)} dimensions")
                    return {
                        'encoding': encoding,
                        'method': 'facenet',
                        'dimensions': len(encoding)
                    }
            except Exception as e:
                print(f"⚠ FaceNet encoding failed: {str(e)}")
        
        # Fallback to basic method (4096 dimensions)
        try:
            encoding = self._extract_basic_features(face_roi)
            print(f"✅ Basic encoding: {len(encoding)} dimensions")
            return {
                'encoding': encoding,
                'method': 'basic',
                'dimensions': len(encoding)
            }
        except Exception as e:
            print(f"❌ All encoding methods failed: {str(e)}")
            return None
    
    def _extract_facenet_features(self, face_roi):
        """Extract features using FaceNet model"""
        try:
            # Preprocess for FaceNet
            face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0)
            face = (face - 127.5) / 128.0
            
            # Run inference
            input_tensor = self.facenet_model.get_tensor_by_name('input:0')
            output_tensor = self.facenet_model.get_tensor_by_name('embeddings:0')
            phase_train = self.facenet_model.get_tensor_by_name('phase_train:0')
            
            embeddings = self.tf_session.run(output_tensor, feed_dict={
                input_tensor: face,
                phase_train: False
            })
            
            return embeddings.flatten().tolist()
            
        except Exception as e:
            print(f"FaceNet extraction error: {str(e)}")
            return None
    
    def _extract_basic_features(self, face_roi):
        """Extract basic features as fallback"""
        face = cv2.resize(face_roi, (64, 64))
        
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        face = cv2.equalizeHist(face)
        features = face.flatten().astype(np.float32) / 255.0
        
        return features.tolist()
    
    def compare_encodings(self, encoding1_data, encoding2_data, threshold=0.6):
        """
        Compare two encoding data structures intelligently
        Handles different encoding methods and dimensions
        """
        if not encoding1_data or not encoding2_data:
            return False, 0.0
        
        # Extract encodings and methods
        if isinstance(encoding1_data, dict):
            enc1 = encoding1_data['encoding']
            method1 = encoding1_data.get('method', 'unknown')
        else:
            enc1 = encoding1_data
            method1 = 'unknown'
        
        if isinstance(encoding2_data, dict):
            enc2 = encoding2_data['encoding']
            method2 = encoding2_data.get('method', 'unknown')
        else:
            enc2 = encoding2_data
            method2 = 'unknown'
        
        if not enc1 or not enc2:
            return False, 0.0
        
        # Convert to numpy arrays
        enc1 = np.array(enc1, dtype=np.float32)
        enc2 = np.array(enc2, dtype=np.float32)
        
        # Check if dimensions match
        if len(enc1) != len(enc2):
            print(f"⚠ Dimension mismatch: {len(enc1)} vs {len(enc2)} - cannot compare directly")
            return False, 0.0
        
        # Use appropriate comparison method based on encoding type
        if len(enc1) == 128 and FACE_RECOGNITION_AVAILABLE:
            # Use face_recognition's optimized comparison
            try:
                distance = face_recognition.face_distance([enc1], enc2)[0]
                similarity = 1 - distance
                is_match = distance < threshold
                print(f"face_recognition comparison: distance={distance:.3f}, similarity={similarity:.3f}")
                return is_match, similarity
            except:
                pass
        
        # Use Euclidean distance for other encodings
        try:
            distance = np.linalg.norm(enc1 - enc2)
            # Normalize based on encoding dimension
            max_distance = np.sqrt(len(enc1))  # Theoretical maximum for normalized vectors
            similarity = max(0, 1 - (distance / max_distance))
            is_match = similarity > threshold
            
            print(f"Euclidean comparison: distance={distance:.3f}, similarity={similarity:.3f}")
            return is_match, similarity
            
        except Exception as e:
            print(f"Comparison error: {str(e)}")
            return False, 0.0


def process_group_photo_optimized(photo_path, class_name, system=None):
    """
    Optimized group photo processing with advanced face recognition
    """
    if system is None:
        system = AdvancedFaceRecognitionSystem()
    
    results = {
        'total_students': 0,
        'present_count': 0,
        'attendance_data': []
    }
    
    try:
        print(f"🚀 Starting attendance processing for class: {class_name}")
        start_time = datetime.now()
        
        # Load image
        print("📸 Loading group photo...")
        group_image = cv2.imread(photo_path)
        if group_image is None:
            raise Exception(f"Could not load group photo: {photo_path}")
        
        print(f"✅ Photo loaded: {group_image.shape}")
        
        # Resize for faster processing if needed
        height, width = group_image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            group_image = cv2.resize(group_image, (new_width, new_height))
            print(f"📐 Resized to: {new_width}x{new_height}")
        
        # Detect faces
        print("🔍 Detecting faces...")
        faces = system.detect_faces(group_image)
        print(f"✅ Detected {len(faces)} faces")
        
        if len(faces) == 0:
            print("⚠ No faces detected - all students marked absent")
            # Import here to avoid circular imports
            from app import app, db, Student
            
            with app.app_context():
                students = Student.query.filter_by(class_name=class_name).all()
                results['total_students'] = len(students)
                
                for student in students:
                    results['attendance_data'].append({
                        'student_id': student.id,
                        'is_present': False,
                        'confidence': 0.0
                    })
            
            return results
        
        # Extract face encodings
        print("🔧 Extracting face encodings...")
        face_encodings = []
        
        for i, face in enumerate(faces):
            try:
                encoding_data = system.extract_face_encoding(group_image, face)
                face_encodings.append(encoding_data)
            except Exception as e:
                print(f"⚠ Face {i+1} encoding failed: {str(e)}")
                face_encodings.append(None)
        
        # Load student data
        print("👥 Loading student database...")
        from app import app, db, Student
        
        with app.app_context():
            students = Student.query.filter_by(class_name=class_name).all()
            results['total_students'] = len(students)
            print(f"Found {len(students)} students in class")
            
            # Process each student
            matched_faces = set()
            
            for i, student in enumerate(students):
                print(f"\n🔍 Processing student {i+1}/{len(students)}: {student.name}")
                
                is_present = False
                best_confidence = 0.0
                best_face_idx = None
                
                if student.face_encoding:
                    try:
                        # Parse student encoding from database
                        student_encoding_raw = json.loads(student.face_encoding)
                        
                        # Determine the format of stored encoding
                        if isinstance(student_encoding_raw, list):
                            # Old format - just the encoding list
                            student_encoding_data = {
                                'encoding': student_encoding_raw,
                                'method': 'facenet' if len(student_encoding_raw) == 512 else 'face_recognition',
                                'dimensions': len(student_encoding_raw)
                            }
                        else:
                            # New format - dict with metadata
                            student_encoding_data = student_encoding_raw
                        
                        print(f"Student encoding: {student_encoding_data.get('dimensions', 'unknown')} dimensions, method: {student_encoding_data.get('method', 'unknown')}")
                        
                        # Compare with all unmatched faces
                        for idx, face_encoding_data in enumerate(face_encodings):
                            if idx in matched_faces or face_encoding_data is None:
                                continue
                            
                            try:
                                match, confidence = system.compare_encodings(
                                    student_encoding_data,
                                    face_encoding_data,
                                    threshold=0.6
                                )
                                
                                if match and confidence > best_confidence:
                                    is_present = True
                                    best_confidence = confidence
                                    best_face_idx = idx
                                    print(f"✨ Potential match found: confidence={confidence:.3f}")
                                
                            except Exception as e:
                                print(f"⚠ Comparison failed: {str(e)}")
                                continue
                    
                    except Exception as e:
                        print(f"⚠ Error processing {student.name}: {str(e)}")
                else:
                    print(f"⚠ {student.name} has no face encoding in database")
                
                # Record results
                if is_present and best_face_idx is not None:
                    matched_faces.add(best_face_idx)
                    results['present_count'] += 1
                    print(f"✅ {student.name} - PRESENT (confidence: {best_confidence:.3f})")
                else:
                    print(f"❌ {student.name} - ABSENT")
                
                results['attendance_data'].append({
                    'student_id': student.id,
                    'is_present': is_present,
                    'confidence': best_confidence
                })
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n🎯 PROCESSING COMPLETED in {total_time:.2f}s")
        print(f"📊 Result: {results['present_count']}/{results['total_students']} students present")
        
        # Create debug image
        try:
            debug_image = group_image.copy()
            for i, face in enumerate(faces):
                x, y, w, h = face
                color = (0, 255, 0) if i in matched_faces else (0, 0, 255)
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(debug_image, f"Face {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            debug_path = f"debug_{os.path.basename(photo_path)}"
            cv2.imwrite(debug_path, debug_image)
            print(f"✅ Debug image saved: {debug_path}")
        except:
            pass
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        traceback.print_exc()
        raise e
    
    return results


if __name__ == "__main__":
    # Test the advanced system
    print("🧪 Testing Advanced Face Recognition System")
    
    system = AdvancedFaceRecognitionSystem()
    
    # Test with a sample image
    test_images = []
    if os.path.exists('uploads'):
        for img in os.listdir('uploads'):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join('uploads', img))
                break
    
    if test_images:
        test_image = test_images[0]
        print(f"\n📸 Testing with image: {test_image}")
        
        # Test detection
        image = cv2.imread(test_image)
        faces = system.detect_faces(image)
        print(f"Detected {len(faces)} faces")
        
        # Test encoding
        if len(faces) > 0:
            encoding_data = system.extract_face_encoding(image, faces[0])
            print(f"Encoding result: {encoding_data}")
    
    print("✅ System test completed!")
