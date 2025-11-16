#!/usr/bin/env python3
"""
Initialization script for Group Image-based Class Attendance System
Creates default admin user and sample data for testing
"""

from app import app, db, User, TeacherClass, Student
from datetime import datetime
import os

def init_database():
    """Initialize database with default data"""
    
    print("🔧 Initializing Group Image-based Class Attendance System...")
    
    with app.app_context():
        # Create all database tables
        print("📊 Creating database tables...")
        db.create_all()
        
        # Create directories
        print("📁 Creating directories...")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['STUDENT_IMAGES_FOLDER'], exist_ok=True)
        
        # Check if admin already exists
        admin_exists = User.query.filter_by(role='admin').first()
        
        if not admin_exists:
            print("👤 Creating default admin user...")
            
            # Create default admin user
            admin = User(
                username='admin',
                email='admin@attendance.system',
                full_name='System Administrator',
                role='admin',
                department='IT Department'
            )
            admin.set_password('admin123')  # Change this in production!
            
            db.session.add(admin)
            db.session.commit()
            
            print("✅ Default admin user created:")
            print("   Username: admin")
            print("   Password: admin123")
            print("   ⚠️  IMPORTANT: Change the password after first login!")
        else:
            print("ℹ️  Admin user already exists")
        
        # Check if teacher exists
        teacher_exists = User.query.filter_by(username='teacher1').first()
        
        if not teacher_exists:
            print("👨‍🏫 Creating sample teacher user...")
            
            # Create sample teacher
            teacher = User(
                username='teacher1',
                email='teacher@attendance.system',
                full_name='John Doe',
                role='teacher',
                department='Computer Science'
            )
            teacher.set_password('teacher123')
            
            db.session.add(teacher)
            db.session.commit()
            
            # Create sample classes for the teacher
            print("📚 Creating sample classes...")
            
            sample_classes = [
                {
                    'class_name': 'CS101',
                    'subject': 'Introduction to Programming',
                    'academic_year': '2024-25',
                    'semester': 'Fall'
                },
                {
                    'class_name': 'CS201',
                    'subject': 'Data Structures',
                    'academic_year': '2024-25',
                    'semester': 'Fall'
                },
                {
                    'class_name': 'MATH101',
                    'subject': 'Calculus I',
                    'academic_year': '2024-25',
                    'semester': 'Fall'
                }
            ]
            
            for class_info in sample_classes:
                teacher_class = TeacherClass(
                    teacher_id=teacher.id,
                    **class_info
                )
                db.session.add(teacher_class)
            
            db.session.commit()
            
            print("✅ Sample teacher and classes created:")
            print("   Username: teacher1")
            print("   Password: teacher123")
            print("   Classes: CS101, CS201, MATH101")
        else:
            print("ℹ️  Sample teacher already exists")
        
        print("\n🎯 System Requirements:")
        print("1. For production use, install additional packages:")
        print("   pip install face-recognition dlib")
        print("   (These require additional system dependencies)")
        
        print("\n2. For better face recognition accuracy:")
        print("   - Use clear, well-lit photos")
        print("   - Ensure students face the camera")
        print("   - Upload high-quality student photos")
        
        print("\n3. Database Configuration:")
        print("   - Development: Using SQLite (current)")
        print("   - Production: Configure MySQL in environment variables")
        
        print("\n🔐 Security Notes:")
        print("   - Change default passwords immediately")
        print("   - Set up HTTPS in production")
        print("   - Configure proper session security")
        
        print("\n📋 Use Cases Implemented:")
        print("   ✅ UC-1: User Login with authentication")
        print("   ✅ UC-2: Capture Group Photo (file upload + webcam)")
        print("   ✅ UC-3: Detect Faces (OpenCV Haar Cascades)")
        print("   ✅ UC-4: Recognize Faces (Feature matching)")
        print("   ✅ UC-5: Mark Attendance (Database records)")
        print("   ✅ UC-6: Provide Attendance Statistics")
        print("   ✅ UC-7: View Student Details")
        print("   ✅ Multi-teacher support with class management")
        print("   ✅ Complete CRUD operations for students")
        
        print("\n🚀 System ready! Start the application with:")
        print("   python app.py")
        print("   Then visit: http://localhost:5000")
        
        print("\n🔧 ML/AI Algorithms Used:")
        print("   - Face Detection: OpenCV Haar Cascade Classifiers")
        print("   - Feature Extraction: HOG + Pixel-based features")
        print("   - Face Matching: Cosine similarity")
        print("   - Upgradeable to: CNN-based models (face_recognition lib)")

if __name__ == '__main__':
    init_database()
