"""
Enhanced Face Recognition System
Uses advanced ML and Computer Vision algorithms for better accuracy
"""

import cv2
import numpy as np
import face_recognition
import dlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import json
import os

class EnhancedFaceRecognitionSystem:
    """
    Enhanced Face Recognition System using state-of-the-art algorithms:
    
    1. Face Detection: 
       - OpenCV Haar Cascades (basic)
       - dlib HOG-based detector (better)
       - face_recognition library (uses CNN - best)
    
    2. Face Recognition:
       - face_recognition library (uses ResNet-based model)
       - 128-dimensional face encodings
       - Euclidean distance for matching
    
    3. Face Clustering:
       - DBSCAN clustering for grouping similar faces
       - Helps in removing duplicates and false positives
    """
    
    def __init__(self, use_cnn=False):
        self.use_cnn = use_cnn  # CNN is more accurate but slower
        
        # Initialize detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to initialize dlib face detector
        try:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_available = True
        except:
            self.dlib_available = False
            print("Warning: dlib not available, using OpenCV only")
        
        # Face recognition model (automatically downloaded on first use)
        self.face_recognition_model = "cnn" if use_cnn else "hog"
        
    def detect_faces_opencv(self, image):
        """Basic face detection using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        return faces
    
    def detect_faces_dlib(self, image):
        """Face detection using dlib HOG-based detector"""
        if not self.dlib_available:
            return self.detect_faces_opencv(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray)
        
        # Convert dlib rectangles to OpenCV format
        face_coords = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_coords.append([x, y, w, h])
        
        return np.array(face_coords)
    
    def detect_faces_advanced(self, image):
        """Advanced face detection using face_recognition library"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image, model=self.face_recognition_model)
        
        # Convert to OpenCV format (top, right, bottom, left) -> (x, y, w, h)
        face_coords = []
        for (top, right, bottom, left) in face_locations:
            x, y, w, h = left, top, right - left, bottom - top
            face_coords.append([x, y, w, h])
        
        return np.array(face_coords)
    
    def extract_face_encoding(self, image, face_location=None):
        """Extract 128-dimensional face encoding using face_recognition library"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if face_location is not None:
            # Convert OpenCV format to face_recognition format
            x, y, w, h = face_location
            face_locations = [(y, x + w, y + h, x)]
        else:
            # Auto-detect faces
            face_locations = face_recognition.face_locations(rgb_image, model=self.face_recognition_model)
        
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) > 0:
            return face_encodings[0].tolist()
        else:
            return None
    
    def extract_face_features_basic(self, image, face_coords):
        """Basic feature extraction (fallback method)"""
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        
        # Resize face to standard size
        face_roi = cv2.resize(face_roi, (128, 128))
        
        # Convert to grayscale and flatten
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better features
        face_roi = cv2.equalizeHist(gray_face)
        
        # Extract HOG features for better representation
        try:
            # Calculate HOG features
            from skimage.feature import hog
            features = hog(face_roi, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
            return features.tolist()
        except:
            # Fallback to basic pixel features
            feature_vector = face_roi.flatten()
            return feature_vector.tolist()
    
    def compare_faces_advanced(self, known_encoding, face_encoding, threshold=0.6):
        """Compare faces using advanced techniques"""
        if known_encoding is None or face_encoding is None:
            return False, 0.0
        
        try:
            # Use face_recognition library's comparison (Euclidean distance)
            known_array = np.array([known_encoding])
            face_array = np.array([face_encoding])
            
            # Calculate distance
            distances = face_recognition.face_distance(known_array, face_array[0])
            distance = distances[0]
            
            # Convert distance to similarity score
            similarity = 1 - distance
            is_match = distance < threshold
            
            return is_match, similarity
        except:
            # Fallback to cosine similarity
            return self.compare_faces_basic(known_encoding, face_encoding, threshold)
    
    def compare_faces_basic(self, known_encoding, face_encoding, threshold=0.6):
        """Basic face comparison using cosine similarity"""
        if known_encoding is None or face_encoding is None:
            return False, 0.0
        
        known_encoding = np.array(known_encoding).reshape(1, -1)
        face_encoding = np.array(face_encoding).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(known_encoding, face_encoding)[0][0]
        
        return similarity > threshold, similarity
    
    def cluster_faces(self, face_encodings, eps=0.5, min_samples=2):
        """Cluster similar faces using DBSCAN to remove duplicates"""
        if len(face_encodings) < 2:
            return [0] * len(face_encodings)
        
        # Convert to numpy array
        encodings_array = np.array(face_encodings)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(encodings_array)
        
        return labels.tolist()
    
    def process_group_photo(self, image, detection_method='advanced'):
        """
        Process group photo and return detected faces with their encodings
        
        detection_method options:
        - 'basic': OpenCV Haar Cascades
        - 'dlib': dlib HOG detector
        - 'advanced': face_recognition library (recommended)
        """
        
        if detection_method == 'basic':
            faces = self.detect_faces_opencv(image)
        elif detection_method == 'dlib':
            faces = self.detect_faces_dlib(image)
        else:  # advanced
            faces = self.detect_faces_advanced(image)
        
        face_data = []
        for i, face in enumerate(faces):
            # Extract encoding
            if detection_method == 'advanced':
                encoding = self.extract_face_encoding(image, face)
            else:
                encoding = self.extract_face_features_basic(image, face)
            
            if encoding is not None:
                face_data.append({
                    'face_id': i,
                    'coordinates': face.tolist(),
                    'encoding': encoding,
                    'detection_confidence': 1.0  # Could be improved with actual confidence scores
                })
        
        return face_data
    
    def identify_students(self, group_faces, student_database, threshold=0.6):
        """
        Identify students in group photo by comparing with database
        
        Args:
            group_faces: List of face data from process_group_photo
            student_database: List of student records with encodings
            threshold: Similarity threshold for matching
        
        Returns:
            Dictionary with attendance results
        """
        results = {
            'total_students': len(student_database),
            'present_count': 0,
            'attendance_data': []
        }
        
        # Track which faces have been matched to prevent double-counting
        matched_faces = set()
        
        for student in student_database:
            is_present = False
            best_confidence = 0.0
            matched_face_id = None
            
            if student.get('face_encoding'):
                student_encoding = student['face_encoding']
                
                # Compare with all detected faces
                for face_data in group_faces:
                    if face_data['face_id'] in matched_faces:
                        continue  # Skip already matched faces
                    
                    face_encoding = face_data['encoding']
                    
                    # Use advanced comparison if available
                    try:
                        match, confidence = self.compare_faces_advanced(
                            student_encoding, face_encoding, threshold
                        )
                    except:
                        match, confidence = self.compare_faces_basic(
                            student_encoding, face_encoding, threshold
                        )
                    
                    if match and confidence > best_confidence:
                        is_present = True
                        best_confidence = confidence
                        matched_face_id = face_data['face_id']
            
            if is_present:
                results['present_count'] += 1
                matched_faces.add(matched_face_id)
            
            results['attendance_data'].append({
                'student_id': student['id'],
                'student_name': student.get('name', 'Unknown'),
                'is_present': is_present,
                'confidence': best_confidence,
                'matched_face_id': matched_face_id
            })
        
        return results
    
    def save_debug_image(self, image, faces, output_path):
        """Save image with face detection boxes for debugging"""
        debug_image = image.copy()
        
        for face in faces:
            if isinstance(face, dict):
                x, y, w, h = face['coordinates']
            else:
                x, y, w, h = face
            
            # Draw rectangle around face
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add face ID
            cv2.putText(debug_image, f"Face {face.get('face_id', 0)}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, debug_image)
    
    def get_face_quality_score(self, image, face_coords):
        """
        Calculate quality score for a detected face
        Factors: size, blur, lighting, angle
        """
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        
        # Size score (larger faces are generally better)
        size_score = min(w * h / (100 * 100), 1.0)  # Normalize to 1.0
        
        # Blur detection using Laplacian variance
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_score = min(cv2.Laplacian(gray_face, cv2.CV_64F).var() / 500, 1.0)
        
        # Lighting score using standard deviation
        lighting_score = min(np.std(gray_face) / 50, 1.0)
        
        # Combined quality score
        quality_score = (size_score + blur_score + lighting_score) / 3.0
        
        return quality_score

# Usage example and configuration
def get_enhanced_face_recognition_system():
    """Factory function to create enhanced face recognition system"""
    # Use CNN for better accuracy in production, HOG for faster processing in development
    use_cnn = os.environ.get('USE_CNN_DETECTION', 'False').lower() == 'true'
    
    return EnhancedFaceRecognitionSystem(use_cnn=use_cnn)
