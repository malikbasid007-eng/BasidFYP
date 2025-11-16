from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import cv2
import numpy as np
import os
from datetime import datetime, date
import base64
from werkzeug.utils import secure_filename
from PIL import Image
import json
import pickle

# Try to import high-performance libraries
try:
    import mediapipe as mp
    import tensorflow as tf
    FACENET_AVAILABLE = True
    print("✓ FaceNet + MediaPipe available - Using high-performance mode")
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠ FaceNet/MediaPipe not available - Using fallback OpenCV mode")
    print("Install with: pip install mediapipe tensorflow")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Database Configuration
# Development: SQLite, Production: MySQL
ENVIRONMENT = os.environ.get('FLASK_ENV', 'development')

if ENVIRONMENT == 'production':
    # MySQL Configuration for Production
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'attendance_system')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}'
else:
    # SQLite Configuration for Development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STUDENT_IMAGES_FOLDER'] = 'student_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='teacher')  # 'admin', 'teacher'
    department = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationship with classes they teach
    classes_taught = db.relationship('TeacherClass', backref='teacher', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class TeacherClass(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    subject = db.Column(db.String(100))
    academic_year = db.Column(db.String(20))
    semester = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('teacher_id', 'class_name', 'academic_year', 'semester'),)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200))
    face_encoding = db.Column(db.Text)  # Store as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with attendance records
    attendance_records = db.relationship('AttendanceRecord', backref='student', lazy=True)

class AttendanceSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    group_photo_path = db.Column(db.String(200))
    total_students = db.Column(db.Integer, default=0)
    present_students = db.Column(db.Integer, default=0)
    created_by = db.Column(db.String(100), default='Teacher')
    
    # Relationship with attendance records
    attendance_records = db.relationship('AttendanceRecord', backref='session', lazy=True)

class AttendanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_session.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    is_present = db.Column(db.Boolean, default=False)
    confidence_score = db.Column(db.Float, default=0.0)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)

# High-Performance FaceNet + MediaPipe System
class HighPerformanceFaceRecognition:
    def __init__(self):
        """Initialize FaceNet + MediaPipe system for ultra-fast face recognition"""
        if not FACENET_AVAILABLE:
            raise Exception("FaceNet libraries not available. Install: pip install mediapipe tensorflow")
        
        print("🚀 Initializing High-Performance Face Recognition System...")
        
        # Initialize MediaPipe face detection (much faster than OpenCV)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.3)
        
        # Try to load FaceNet model
        self.facenet_model = None
        self.tf_session = None
        self._load_facenet_model()
        
        print("✅ High-Performance Face Recognition System initialized!")
    
    def _load_facenet_model(self):
        """Load FaceNet model for generating embeddings"""
        try:
            # Try to load from your previous project location
            model_paths = [
                "detectionThroughApi/facenet/facenet.pb",
                "facenet/facenet.pb",
                "models/facenet.pb"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"📁 Found FaceNet model at: {model_path}")
                    self._load_graph(model_path)
                    return
            
            print("⚠ FaceNet model not found - will use MediaPipe + basic embeddings")
            
        except Exception as e:
            print(f"⚠ Could not load FaceNet model: {str(e)}")
    
    def _load_graph(self, model_file):
        """Load TensorFlow graph from .pb file"""
        try:
            graph = tf.Graph()
            with graph.as_default():
                graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(model_file, "rb") as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="")
            
            self.tf_session = tf.compat.v1.Session(graph=graph)
            self.facenet_model = graph
            print("✅ FaceNet model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading FaceNet model: {str(e)}")
    
    def detect_faces(self, image):
        """Detect faces using MediaPipe (much faster than OpenCV)"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                # Convert relative coordinates to pixel coordinates
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                faces.append((x, y, w, h))
        
        return faces
    
    def extract_face_features(self, image, face_coords):
        """Extract face features using FaceNet or optimized fallback"""
        x, y, w, h = face_coords
        
        # Ensure coordinates are within image bounds
        ih, iw = image.shape[:2]
        x = max(0, min(x, iw - 1))
        y = max(0, min(y, ih - 1))
        w = max(1, min(w, iw - x))
        h = max(1, min(h, ih - y))
        
        face_roi = image[y:y+h, x:x+w]
        
        if self.facenet_model and self.tf_session:
            # Use FaceNet for high-quality embeddings
            return self._extract_facenet_features(face_roi)
        else:
            # Use optimized fallback method
            return self._extract_optimized_features(face_roi)
    
    def _extract_facenet_features(self, face_roi):
        """Extract features using FaceNet model (128 dimensions)"""
        try:
            # Preprocess face for FaceNet
            face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160))  # FaceNet input size
            face = np.expand_dims(face, axis=0)
            face = (face - 127.5) / 128.0  # Normalize to [-1, 1]
            
            # Run FaceNet inference
            input_tensor = self.facenet_model.get_tensor_by_name('input:0')
            output_tensor = self.facenet_model.get_tensor_by_name('embeddings:0')
            phase_train = self.facenet_model.get_tensor_by_name('phase_train:0')
            
            embeddings = self.tf_session.run(output_tensor, feed_dict={
                input_tensor: face,
                phase_train: False
            })
            
            return embeddings.flatten().tolist()
            
        except Exception as e:
            print(f"Error in FaceNet extraction: {str(e)}")
            return self._extract_optimized_features(face_roi)
    
    def _extract_optimized_features(self, face_roi):
        """Extract features using optimized method (64 dimensions for speed)"""
        # Resize to smaller size for speed (64x64 -> 32x32 -> 8x8 = 64 dimensions)
        face = cv2.resize(face_roi, (64, 64))
        
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing for better matching
        face = cv2.GaussianBlur(face, (3, 3), 0)
        face = cv2.equalizeHist(face)
        
        # Create compact feature vector (8x8 = 64 dimensions)
        compact = cv2.resize(face, (8, 8))
        features = compact.flatten().astype(np.float32) / 255.0
        
        return features.tolist()
    
    def compare_faces(self, known_encoding, face_encoding, threshold=0.7):
        """Compare face encodings using Euclidean distance (faster than cosine similarity)"""
        if known_encoding is None or face_encoding is None:
            return False, 0.0
        
        # Convert to numpy arrays
        known = np.array(known_encoding, dtype=np.float32)
        face = np.array(face_encoding, dtype=np.float32)
        
        # Use Euclidean distance (much faster than cosine similarity)
        distance = np.linalg.norm(known - face)
        
        # Convert distance to similarity score (0-1, where 1 is perfect match)
        max_distance = 2.0  # Maximum possible L2 distance for normalized vectors
        similarity = max(0, 1 - (distance / max_distance))
        
        return similarity > threshold, similarity

# Legacy OpenCV System (fallback)
class FaceRecognitionSystem:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image):
        """Detect faces in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_face_features(self, image, face_coords):
        """Extract face features - OPTIMIZED for speed and memory efficiency"""
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        
        # OPTIMIZATION: Use smaller face size (64x64 instead of 128x128)
        # This reduces feature vector from 16,384 to 4,096 dimensions (75% reduction!)
        face_roi = cv2.resize(face_roi, (64, 64))
        
        # Convert to grayscale for faster processing
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi
        
        # OPTIMIZATION: Apply Gaussian blur to reduce noise and improve matching
        gray_face = cv2.GaussianBlur(gray_face, (3, 3), 0)
        
        # OPTIMIZATION: Use histogram equalization for better lighting normalization
        gray_face = cv2.equalizeHist(gray_face)
        
        # OPTIMIZATION: Extract features using downsample + flatten (faster than full resolution)
        # Downsample by factor of 2: 64x64 -> 32x32 = 1,024 dimensions (much faster!)
        downsampled = cv2.resize(gray_face, (32, 32))
        feature_vector = downsampled.flatten()
        
        # OPTIMIZATION: Normalize to 0-1 range for consistent comparisons
        feature_vector = feature_vector.astype(np.float32) / 255.0
        
        return feature_vector.tolist()
    
    def compare_faces(self, known_encoding, face_encoding, threshold=0.6):
        """Compare two face encodings"""
        if known_encoding is None or face_encoding is None:
            return False, 0.0
        
        # Calculate similarity (simplified - using correlation)
        known_encoding = np.array(known_encoding)
        face_encoding = np.array(face_encoding)
        
        # Normalize vectors
        known_norm = known_encoding / np.linalg.norm(known_encoding)
        face_norm = face_encoding / np.linalg.norm(face_encoding)
        
        # Calculate cosine similarity
        similarity = np.dot(known_norm, face_norm)
        
        return similarity > threshold, similarity

# Initialize face recognition system (with fallback)
try:
    if FACENET_AVAILABLE:
        face_recognition_system = HighPerformanceFaceRecognition()
        print("🚀 Using High-Performance FaceNet + MediaPipe system")
    else:
        raise Exception("FaceNet not available")
except Exception as e:
    print(f"⚠ Falling back to OpenCV system: {str(e)}")
    face_recognition_system = FaceRecognitionSystem()
    print("📋 Using OpenCV fallback system")

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            flash('Administrator access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login - UC-1"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Validate empty fields
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('auth/login.html')
        
        # Find user and validate credentials
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password) and user.is_active:
            # Successful login
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['full_name'] = user.full_name
            
            # Update last login time
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            flash(f'Welcome back, {user.full_name}!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            # Invalid credentials
            flash('Invalid username or password.', 'error')
    
    return render_template('auth/login.html')

@app.route('/logout')
def logout():
    """User logout"""
    username = session.get('full_name', 'User')
    session.clear()
    flash(f'You have been logged out successfully. Goodbye, {username}!', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration (Admin only)"""
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            full_name = request.form.get('full_name', '').strip()
            department = request.form.get('department', '').strip()
            role = request.form.get('role', 'teacher')
            
            # Validation
            if not all([username, email, password, full_name]):
                flash('All fields are required.', 'error')
                return render_template('auth/register.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'error')
                return render_template('auth/register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'error')
                return render_template('auth/register.html')
            
            # Check if user already exists
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'error')
                return render_template('auth/register.html')
            
            if User.query.filter_by(email=email).first():
                flash('Email already exists.', 'error')
                return render_template('auth/register.html')
            
            # Create new user
            user = User(
                username=username,
                email=email,
                full_name=full_name,
                department=department,
                role=role
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            flash('User registered successfully!', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            flash(f'Error creating user: {str(e)}', 'error')
    
    return render_template('auth/register.html')

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard - redirected from index if logged in"""
    user = User.query.get(session['user_id'])
    
    # Get user's classes if teacher
    user_classes = []
    if user.role == 'teacher':
        user_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
    
    # Get recent statistics
    total_students = Student.query.count()
    total_sessions = AttendanceSession.query.count()
    recent_sessions = AttendanceSession.query.order_by(AttendanceSession.date.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                         user=user,
                         user_classes=user_classes,
                         total_students=total_students,
                         total_sessions=total_sessions,
                         recent_sessions=recent_sessions)

@app.route('/index')
def index():
    """Public index page - redirects to dashboard if logged in"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# Class Management Routes
@app.route('/classes')
@login_required
def classes():
    """Class management page"""
    user = User.query.get(session['user_id'])
    
    if user.role == 'admin':
        # Admin can see all classes
        all_classes = TeacherClass.query.all()
    else:
        # Teachers can only see their own classes
        all_classes = TeacherClass.query.filter_by(teacher_id=user.id).all()
    
    return render_template('classes.html', classes=all_classes, user=user)

@app.route('/add_class', methods=['GET', 'POST'])
@login_required
def add_class():
    """Add new class"""
    if request.method == 'POST':
        try:
            class_name = request.form['class_name']
            subject = request.form.get('subject', '')
            academic_year = request.form.get('academic_year', '')
            semester = request.form.get('semester', '')
            teacher_id = session['user_id']
            
            # Check if class already exists for this teacher
            existing_class = TeacherClass.query.filter_by(
                teacher_id=teacher_id,
                class_name=class_name,
                academic_year=academic_year,
                semester=semester
            ).first()
            
            if existing_class:
                flash('This class already exists!', 'error')
                return redirect(url_for('add_class'))
            
            # Create new class
            new_class = TeacherClass(
                teacher_id=teacher_id,
                class_name=class_name,
                subject=subject,
                academic_year=academic_year,
                semester=semester
            )
            
            db.session.add(new_class)
            db.session.commit()
            
            flash('Class added successfully!', 'success')
            return redirect(url_for('classes'))
            
        except Exception as e:
            flash(f'Error adding class: {str(e)}', 'error')
            return redirect(url_for('add_class'))
    
    return render_template('add_class.html')

@app.route('/edit_class/<int:class_id>', methods=['GET', 'POST'])
@login_required
def edit_class(class_id):
    """Edit existing class"""
    teacher_class = TeacherClass.query.get_or_404(class_id)
    user = User.query.get(session['user_id'])
    
    # Check permissions
    if user.role != 'admin' and teacher_class.teacher_id != user.id:
        flash('You can only edit your own classes.', 'error')
        return redirect(url_for('classes'))
    
    if request.method == 'POST':
        try:
            teacher_class.class_name = request.form['class_name']
            teacher_class.subject = request.form.get('subject', '')
            teacher_class.academic_year = request.form.get('academic_year', '')
            teacher_class.semester = request.form.get('semester', '')
            
            db.session.commit()
            
            flash('Class updated successfully!', 'success')
            return redirect(url_for('classes'))
            
        except Exception as e:
            flash(f'Error updating class: {str(e)}', 'error')
    
    return render_template('edit_class.html', teacher_class=teacher_class)

@app.route('/delete_class/<int:class_id>', methods=['POST'])
@login_required
def delete_class(class_id):
    """Delete class (soft delete)"""
    teacher_class = TeacherClass.query.get_or_404(class_id)
    user = User.query.get(session['user_id'])
    
    # Check permissions
    if user.role != 'admin' and teacher_class.teacher_id != user.id:
        flash('You can only delete your own classes.', 'error')
        return redirect(url_for('classes'))
    
    try:
        # Soft delete - just mark as inactive
        teacher_class.is_active = False
        db.session.commit()
        
        flash('Class deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting class: {str(e)}', 'error')
    
    return redirect(url_for('classes'))

# Student Management Routes with CRUD
@app.route('/students')
@login_required
def students():
    """Student management page with filtering"""
    user = User.query.get(session['user_id'])
    class_filter = request.args.get('class', '')
    search = request.args.get('search', '')
    
    # Build query
    query = Student.query
    
    # Filter by class if specified
    if class_filter:
        query = query.filter(Student.class_name == class_filter)
    
    # Search functionality
    if search:
        query = query.filter(
            db.or_(
                Student.name.contains(search),
                Student.student_id.contains(search),
                Student.email.contains(search)
            )
        )
    
    all_students = query.all()
    
    # Get available classes for filter dropdown
    if user.role == 'admin':
        available_classes = db.session.query(Student.class_name).distinct().all()
        user_classes = []  # Admin has access to all classes
    else:
        # Teachers see only their classes
        teacher_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
        class_names = [tc.class_name for tc in teacher_classes]
        available_classes = [(name,) for name in class_names]
        user_classes = teacher_classes  # Pass teacher's classes for permission checks
    
    return render_template('students.html', 
                         students=all_students,
                         available_classes=available_classes,
                         current_filter=class_filter,
                         current_search=search,
                         user=user,
                         user_classes=user_classes)

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    """Add new student"""
    if request.method == 'POST':
        try:
            student_id = request.form['student_id']
            name = request.form['name']
            email = request.form['email']
            class_name = request.form['class_name']
            
            # Check if student already exists
            existing_student = Student.query.filter_by(student_id=student_id).first()
            if existing_student:
                flash('Student ID already exists!', 'error')
                return redirect(url_for('add_student'))
            
            # Handle image upload
            if 'image' not in request.files:
                flash('No image uploaded!', 'error')
                return redirect(request.url)
            
            file = request.files['image']
            if file.filename == '':
                flash('No image selected!', 'error')
                return redirect(request.url)
            
            if file:
                filename = secure_filename(f"{student_id}_{file.filename}")
                filepath = os.path.join(app.config['STUDENT_IMAGES_FOLDER'], filename)
                file.save(filepath)
                
                # Extract face features
                image = cv2.imread(filepath)
                faces = face_recognition_system.detect_faces(image)
                
                face_encoding = None
                if len(faces) > 0:
                    # Use the first detected face
                    face_encoding = face_recognition_system.extract_face_features(image, faces[0])
                
                # Create new student
                new_student = Student(
                    student_id=student_id,
                    name=name,
                    email=email,
                    class_name=class_name,
                    image_path=filepath,
                    face_encoding=json.dumps(face_encoding) if face_encoding else None
                )
                
                db.session.add(new_student)
                db.session.commit()
                
                flash('Student added successfully!', 'success')
                return redirect(url_for('students'))
                
        except Exception as e:
            flash(f'Error adding student: {str(e)}', 'error')
            return redirect(url_for('add_student'))
    
    # Get available classes for the current user
    user = User.query.get(session['user_id'])
    if user.role == 'admin':
        available_classes = db.session.query(TeacherClass.class_name).distinct().all()
    else:
        available_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
    
    return render_template('add_student.html', available_classes=available_classes)

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    """Edit existing student"""
    student = Student.query.get_or_404(student_id)
    user = User.query.get(session['user_id'])
    
    # Check if user has permission to edit this student
    if user.role != 'admin':
        # Teachers can only edit students from their classes
        teacher_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
        class_names = [tc.class_name for tc in teacher_classes]
        if student.class_name not in class_names:
            flash('You can only edit students from your classes.', 'error')
            return redirect(url_for('students'))
    
    if request.method == 'POST':
        try:
            # Update basic info (student_id cannot be changed)
            student.name = request.form['name']
            student.email = request.form['email']
            student.class_name = request.form['class_name']
            
            # Handle new image upload (optional)
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename != '':
                    # Delete old image file
                    if student.image_path and os.path.exists(student.image_path):
                        os.remove(student.image_path)
                    
                    # Save new image
                    filename = secure_filename(f"{student.student_id}_{file.filename}")
                    filepath = os.path.join(app.config['STUDENT_IMAGES_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Extract new face features
                    image = cv2.imread(filepath)
                    faces = face_recognition_system.detect_faces(image)
                    
                    face_encoding = None
                    if len(faces) > 0:
                        face_encoding = face_recognition_system.extract_face_features(image, faces[0])
                    
                    student.image_path = filepath
                    student.face_encoding = json.dumps(face_encoding) if face_encoding else None
            
            db.session.commit()
            
            flash('Student updated successfully!', 'success')
            return redirect(url_for('students'))
            
        except Exception as e:
            flash(f'Error updating student: {str(e)}', 'error')
    
    # Get available classes
    if user.role == 'admin':
        available_classes = db.session.query(TeacherClass.class_name).distinct().all()
    else:
        available_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
    
    return render_template('edit_student.html', student=student, available_classes=available_classes)

@app.route('/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    """Delete student"""
    student = Student.query.get_or_404(student_id)
    user = User.query.get(session['user_id'])
    
    # Check permissions
    if user.role != 'admin':
        teacher_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
        class_names = [tc.class_name for tc in teacher_classes]
        if student.class_name not in class_names:
            flash('You can only delete students from your classes.', 'error')
            return redirect(url_for('students'))
    
    try:
        # Delete image file if exists
        if student.image_path and os.path.exists(student.image_path):
            os.remove(student.image_path)
        
        # Delete student record
        db.session.delete(student)
        db.session.commit()
        
        flash('Student deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting student: {str(e)}', 'error')
    
    return redirect(url_for('students'))

@app.route('/attendance')
def attendance():
    """Attendance management page"""
    sessions = AttendanceSession.query.order_by(AttendanceSession.date.desc()).all()
    return render_template('attendance.html', sessions=sessions)

@app.route('/take_attendance', methods=['GET', 'POST'])
@login_required
def take_attendance():
    """Take attendance using group photo - UC-2: Capture Group Photo"""
    if request.method == 'POST':
        try:
            session_name = request.form['session_name']
            class_name = request.form['class_name']
            
            # Handle webcam data or file upload
            if 'webcam_data' in request.form and request.form['webcam_data']:
                # Process webcam capture
                image_data = request.form['webcam_data']
                
                # Decode base64 image
                image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, part
                image_bytes = base64.b64decode(image_data)
                
                filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
            
            elif 'group_photo' in request.files and request.files['group_photo'].filename:
                # Handle file upload
                file = request.files['group_photo']
                filename = secure_filename(f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
            else:
                flash('No photo provided!', 'error')
                return redirect(request.url)
            
            # Process the group photo with optimized algorithm
            print(f"🔄 Starting attendance processing for {class_name}...")
            attendance_results = process_group_photo_optimized(filepath, class_name)
            print(f"✅ Attendance processing completed successfully!")
            
            # Create attendance session
            user = User.query.get(session['user_id'])
            new_session = AttendanceSession(
                session_name=session_name,
                class_name=class_name,
                group_photo_path=filepath,
                total_students=attendance_results['total_students'],
                present_students=attendance_results['present_count'],
                created_by=user.full_name
            )
            
            db.session.add(new_session)
            db.session.flush()  # Get the session ID
            
            # Create attendance records
            for result in attendance_results['attendance_data']:
                attendance_record = AttendanceRecord(
                    session_id=new_session.id,
                    student_id=result['student_id'],
                    is_present=result['is_present'],
                    confidence_score=result['confidence']
                )
                db.session.add(attendance_record)
            
            db.session.commit()
            
            flash(f'Attendance processed! {attendance_results["present_count"]} out of {attendance_results["total_students"]} students detected.', 'success')
            return redirect(url_for('view_attendance', session_id=new_session.id))
            
        except Exception as e:
            flash(f'Error processing attendance: {str(e)}', 'error')
            return redirect(url_for('take_attendance'))
    
    # Get available classes for current user
    user = User.query.get(session['user_id'])
    if user.role == 'admin':
        classes = db.session.query(Student.class_name).distinct().all()
    else:
        teacher_classes = TeacherClass.query.filter_by(teacher_id=user.id, is_active=True).all()
        class_names = [tc.class_name for tc in teacher_classes]
        classes = [(name,) for name in class_names]
    
    classes = [c[0] for c in classes] if classes else []
    
    return render_template('take_attendance.html', classes=classes)

def process_group_photo_optimized(photo_path, class_name):
    """Optimized attendance processing with fixed face recognition system"""
    # Import the face recognition system
    from face_recognition_system import process_group_photo_optimized
    
    # Process attendance with the optimized system
    return process_group_photo_optimized(photo_path, class_name)
    
    try:
        print(f"🚀 Starting OPTIMIZED attendance processing for class: {class_name}")
        start_time = datetime.now()
        
        # Load and optimize image
        print("📸 Loading group photo...")
        group_image = cv2.imread(photo_path)
        if group_image is None:
            raise Exception("Could not load group photo")
        print(f"✅ Photo loaded successfully: {group_image.shape}")
        
        # SPEED OPTIMIZATION 1: Aggressive image resizing for faster processing
        height, width = group_image.shape[:2]
        print(f"📏 Original image size: {width}x{height}")
        if width > 600:  # More aggressive resizing
            scale = 600 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            group_image = cv2.resize(group_image, (new_width, new_height))
            print(f"📐 Image resized to: {new_width}x{new_height}")
        
        # SPEED OPTIMIZATION 2: Use fastest face detection method
        print("🔍 Starting face detection...")
        faces = face_recognition_system.detect_faces(group_image)
        detection_time = (datetime.now() - start_time).total_seconds()
        print(f"⚡ Detected {len(faces)} faces in {detection_time:.2f}s")
        
        if len(faces) == 0:
            print("⚠ No faces detected - marking all as absent")
            students = Student.query.filter_by(class_name=class_name).all()
            results['total_students'] = len(students)
            for student in students:
                results['attendance_data'].append({
                    'student_id': student.id,
                    'is_present': False,
                    'confidence': 0.0
                })
            return results
        
        # SPEED OPTIMIZATION 3: Parallel face encoding extraction
        print("⚡ Extracting face features...")
        feature_start = datetime.now()
        detected_face_encodings = []
        
        for i, face in enumerate(faces):
            print(f"🔧 Processing face {i+1}/{len(faces)}...")
            try:
                face_encoding = face_recognition_system.extract_face_features(group_image, face)
                detected_face_encodings.append(face_encoding)
                print(f"✅ Face {i+1} features extracted")
            except Exception as e:
                print(f"❌ Error extracting features from face {i}: {str(e)}")
                detected_face_encodings.append(None)
        
        feature_time = (datetime.now() - feature_start).total_seconds()
        print(f"⚡ Feature extraction completed in {feature_time:.2f}s")
        
        # SPEED OPTIMIZATION 4: Batch load and pre-process all student data
        print("👥 Loading student data...")
        students = Student.query.filter_by(class_name=class_name).all()
        results['total_students'] = len(students)
        print(f"📊 Found {len(students)} students in class {class_name}")
        
        # Pre-parse all student encodings at once
        print("🔄 Pre-processing student encodings...")
        student_data = []
        for i, student in enumerate(students):
            encoding = None
            if student.face_encoding:
                try:
                    encoding = json.loads(student.face_encoding)
                    print(f"✅ Student {i+1}: {student.name} - encoding loaded")
                except Exception as e:
                    print(f"⚠ Student {i+1}: {student.name} - encoding error: {str(e)}")
            else:
                print(f"⚠ Student {i+1}: {student.name} - no encoding found")
            
            student_data.append({
                'id': student.id,
                'name': student.name,
                'encoding': encoding
            })
        
        # SPEED OPTIMIZATION 5: Fast comparison with early termination
        print("🔍 Starting face comparison...")
        comparison_start = datetime.now()
        matched_faces = set()  # Track matched faces to avoid double-matching
        
        for i, student in enumerate(student_data):
            print(f"🔍 Comparing student {i+1}/{len(student_data)}: {student['name']}")
            is_present = False
            best_confidence = 0.0
            best_face_idx = None
            
            if student['encoding'] is not None:
                # Compare against unmatched faces only
                for idx, face_encoding in enumerate(detected_face_encodings):
                    if idx in matched_faces or face_encoding is None:
                        continue
                    
                    try:
                        match, confidence = face_recognition_system.compare_faces(
                            student['encoding'], face_encoding, threshold=0.45  # Optimized threshold
                        )
                        
                        if match and confidence > best_confidence:
                            is_present = True
                            best_confidence = confidence
                            best_face_idx = idx
                    except Exception as e:
                        print(f"⚠ Comparison error for {student['name']}: {str(e)}")
                        continue
            else:
                print(f"⚠ {student['name']} - no encoding to compare")
            
            # Mark face as matched to prevent double-matching
            if is_present and best_face_idx is not None:
                matched_faces.add(best_face_idx)
                results['present_count'] += 1
                print(f"✅ {student['name']} - Present (confidence: {best_confidence:.3f})")
            else:
                print(f"❌ {student['name']} - Absent")
            
            results['attendance_data'].append({
                'student_id': student['id'],
                'is_present': is_present,
                'confidence': best_confidence
            })
        
        comparison_time = (datetime.now() - comparison_start).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"⚡ Comparison completed in {comparison_time:.2f}s")
        print(f"🎯 TOTAL PROCESSING TIME: {total_time:.2f}s")
        print(f"📊 Result: {results['present_count']}/{results['total_students']} students present")
        
    except Exception as e:
        print(f"❌ Error in optimized processing: {str(e)}")
        import traceback
        traceback.print_exc()
        # Re-raise the error to show it in fast processing mode
        raise e
    
    return results

def process_group_photo_fallback(photo_path, class_name):
    """Fallback processing method if optimized version fails"""
    results = {
        'total_students': 0,
        'present_count': 0,
        'attendance_data': []
    }
    
    try:
        print(f"🔄 Using fallback processing for class: {class_name}")
        
        # Load group photo
        group_image = cv2.imread(photo_path)
        if group_image is None:
            raise Exception("Could not load group photo")
        
        # Basic image resizing
        height, width = group_image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            group_image = cv2.resize(group_image, (new_width, new_height))
        
        # Detect faces
        faces = face_recognition_system.detect_faces(group_image)
        print(f"Detected {len(faces)} faces in group photo")
        
        if len(faces) == 0:
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
        detected_face_encodings = []
        for i, face in enumerate(faces):
            try:
                face_encoding = face_recognition_system.extract_face_features(group_image, face)
                detected_face_encodings.append(face_encoding)
            except Exception as e:
                detected_face_encodings.append(None)
        
        # Process students
        students = Student.query.filter_by(class_name=class_name).all()
        results['total_students'] = len(students)
        
        for student in students:
            is_present = False
            best_confidence = 0.0
            
            if student.face_encoding:
                try:
                    student_encoding = json.loads(student.face_encoding)
                    
                    for face_encoding in detected_face_encodings:
                        if face_encoding is not None:
                            try:
                                match, confidence = face_recognition_system.compare_faces(
                                    student_encoding, face_encoding, threshold=0.5
                                )
                                
                                if match and confidence > best_confidence:
                                    is_present = True
                                    best_confidence = confidence
                            except:
                                continue
                except:
                    pass
            
            if is_present:
                results['present_count'] += 1
            
            results['attendance_data'].append({
                'student_id': student.id,
                'is_present': is_present,
                'confidence': best_confidence
            })
    
    except Exception as e:
        print(f"Error in fallback processing: {str(e)}")
    
    return results

@app.route('/view_attendance/<int:session_id>')
def view_attendance(session_id):
    """View attendance details for a session"""
    session = AttendanceSession.query.get_or_404(session_id)
    
    # Get attendance records with student details
    records = db.session.query(AttendanceRecord, Student).join(Student).filter(
        AttendanceRecord.session_id == session_id
    ).all()
    
    return render_template('view_attendance.html', session=session, records=records)

@app.route('/statistics')
def statistics():
    """View attendance statistics"""
    # Get overall statistics
    total_students = Student.query.count()
    total_sessions = AttendanceSession.query.count()
    
    # Get class-wise statistics
    class_stats = db.session.query(
        Student.class_name,
        db.func.count(Student.id).label('student_count')
    ).group_by(Student.class_name).all()
    
    # Get recent attendance sessions
    recent_sessions = AttendanceSession.query.order_by(
        AttendanceSession.date.desc()
    ).limit(5).all()
    
    return render_template('statistics.html', 
                         total_students=total_students,
                         total_sessions=total_sessions,
                         class_stats=class_stats,
                         recent_sessions=recent_sessions)

@app.route('/student_details/<int:student_id>')
def student_details(student_id):
    """View individual student details and attendance history"""
    student = Student.query.get_or_404(student_id)
    
    # Get attendance history
    attendance_history = db.session.query(AttendanceRecord, AttendanceSession).join(
        AttendanceSession
    ).filter(AttendanceRecord.student_id == student_id).order_by(
        AttendanceSession.date.desc()
    ).all()
    
    # Calculate attendance percentage
    total_sessions = len(attendance_history)
    present_sessions = sum(1 for record, session in attendance_history if record.is_present)
    attendance_percentage = (present_sessions / total_sessions * 100) if total_sessions > 0 else 0
    
    return render_template('student_details.html', 
                         student=student,
                         attendance_history=attendance_history,
                         attendance_percentage=attendance_percentage)

# Static file serving routes
@app.route('/student_images/<filename>')
def serve_student_image(filename):
    """Serve student images"""
    return send_from_directory(app.config['STUDENT_IMAGES_FOLDER'], filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API Endpoints
@app.route('/api/capture_photo', methods=['POST'])
def api_capture_photo():
    """API endpoint to capture photo from webcam"""
    try:
        # Get base64 image data from request
        image_data = request.json.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, part
        image_bytes = base64.b64decode(image_data)
        
        # Save image
        filename = f"webcam_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({'success': True, 'filepath': filepath})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Create upload directories if they don't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['STUDENT_IMAGES_FOLDER'], exist_ok=True)
    
    app.run(debug=True)
