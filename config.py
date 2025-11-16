"""
Configuration settings for different environments
"""
import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Upload folders
    UPLOAD_FOLDER = 'uploads'
    STUDENT_IMAGES_FOLDER = 'student_images'
    
    # Face recognition settings
    FACE_RECOGNITION_TOLERANCE = 0.6
    USE_CNN_DETECTION = False  # Set to True for better accuracy (slower)
    DETECTION_METHOD = 'advanced'  # 'basic', 'dlib', or 'advanced'
    
    # Session timeout
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///attendance_dev.db'
    
    # More lenient face recognition for development/testing
    FACE_RECOGNITION_TOLERANCE = 0.5
    USE_CNN_DETECTION = False  # Faster HOG detection for development

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    FACE_RECOGNITION_TOLERANCE = 0.4

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # MySQL database configuration
    MYSQL_HOST = os.environ.get('MYSQL_HOST') or 'localhost'
    MYSQL_USER = os.environ.get('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD') or ''
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE') or 'attendance_system'
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}'
    
    # Stricter security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Better face recognition for production
    FACE_RECOGNITION_TOLERANCE = 0.6
    USE_CNN_DETECTION = True  # More accurate CNN detection for production
    
    # Enable logging
    LOG_LEVEL = 'INFO'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
