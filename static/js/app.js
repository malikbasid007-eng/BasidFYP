// Main JavaScript file for Group Attendance System

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Add fade-in animation to main content
    const mainContent = document.querySelector('.container');
    if (mainContent) {
        mainContent.classList.add('fade-in');
    }

    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validations
    initializeFormValidation();
    
    // Initialize image previews
    initializeImagePreviews();
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

function initializeFormValidation() {
    // Custom form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

function initializeImagePreviews() {
    // Generic image preview functionality
    const imageInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
    
    imageInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            const previewContainer = this.closest('.form-group, .mb-3').querySelector('.image-preview');
            
            if (file && previewContainer) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    let img = previewContainer.querySelector('img');
                    if (!img) {
                        img = document.createElement('img');
                        img.className = 'img-thumbnail preview-image';
                        img.style.maxWidth = '200px';
                        img.style.maxHeight = '200px';
                        previewContainer.appendChild(img);
                    }
                    img.src = e.target.result;
                    previewContainer.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    });
}

// Utility Functions
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertContainer, container.firstChild);
    
    // Auto-dismiss after specified duration
    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.remove();
        }
    }, duration);
}

function showLoading(element, text = 'Loading...') {
    const originalContent = element.innerHTML;
    element.disabled = true;
    element.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${text}`;
    
    return function hideLoading() {
        element.disabled = false;
        element.innerHTML = originalContent;
    };
}

// Camera utilities for webcam functionality
class CameraManager {
    constructor() {
        this.stream = null;
        this.video = null;
        this.canvas = null;
    }
    
    async startCamera(videoElement) {
        try {
            this.video = videoElement;
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'user'
                } 
            });
            this.video.srcObject = this.stream;
            return true;
        } catch (error) {
            console.error('Error starting camera:', error);
            showAlert('Could not access camera. Please check permissions.', 'danger');
            return false;
        }
    }
    
    capturePhoto(canvasElement) {
        if (!this.video || !this.stream) {
            showAlert('Camera not initialized', 'danger');
            return null;
        }
        
        this.canvas = canvasElement;
        const context = this.canvas.getContext('2d');
        
        // Set canvas size to match video
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw video frame to canvas
        context.drawImage(this.video, 0, 0);
        
        // Get image data URL
        return this.canvas.toDataURL('image/jpeg', 0.8);
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
    }
}

// Face detection utilities (client-side basic detection)
class FaceDetectionClient {
    static detectFaces(imageData) {
        // This is a placeholder for client-side face detection
        // In a real implementation, you might use libraries like face-api.js
        return new Promise((resolve) => {
            // Simulate face detection processing
            setTimeout(() => {
                resolve([
                    { x: 100, y: 100, width: 150, height: 150, confidence: 0.95 },
                    { x: 300, y: 120, width: 140, height: 140, confidence: 0.88 }
                ]);
            }, 1000);
        });
    }
    
    static drawFaceBoxes(canvas, faces) {
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        
        faces.forEach(face => {
            ctx.strokeRect(face.x, face.y, face.width, face.height);
            
            // Draw confidence score
            ctx.fillStyle = '#00ff00';
            ctx.font = '16px Arial';
            ctx.fillText(
                `${Math.round(face.confidence * 100)}%`, 
                face.x, 
                face.y - 5
            );
        });
    }
}

// Data validation utilities
class ValidationUtils {
    static validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    static validateStudentId(studentId) {
        // Allow alphanumeric student IDs, 3-20 characters
        const idRegex = /^[a-zA-Z0-9]{3,20}$/;
        return idRegex.test(studentId);
    }
    
    static validateImageFile(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        const maxSize = 5 * 1024 * 1024; // 5MB
        
        if (!allowedTypes.includes(file.type)) {
            return { valid: false, message: 'Only JPEG and PNG images are allowed.' };
        }
        
        if (file.size > maxSize) {
            return { valid: false, message: 'Image size must be less than 5MB.' };
        }
        
        return { valid: true, message: '' };
    }
}

// Export utilities for use in other scripts
window.AttendanceApp = {
    CameraManager,
    FaceDetectionClient,
    ValidationUtils,
    showAlert,
    showLoading
};
