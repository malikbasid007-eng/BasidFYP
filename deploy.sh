#!/bin/bash

# Group Image-based Class Attendance System - Deployment Script
# This script sets up the production environment

echo "🚀 Starting deployment of Attendance System..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for security reasons"
   exit 1
fi

# Set environment variables
export FLASK_ENV=production
export SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 32)}

# Database configuration
export MYSQL_HOST=${MYSQL_HOST:-localhost}
export MYSQL_USER=${MYSQL_USER:-attendance_user}
export MYSQL_PASSWORD=${MYSQL_PASSWORD:-}
export MYSQL_DATABASE=${MYSQL_DATABASE:-attendance_system}

# Face recognition configuration
export USE_CNN_DETECTION=${USE_CNN_DETECTION:-true}
export FACE_RECOGNITION_TOLERANCE=${FACE_RECOGNITION_TOLERANCE:-0.6}

echo "📋 Environment configured:"
echo "   - Environment: $FLASK_ENV"
echo "   - Database: $MYSQL_HOST/$MYSQL_DATABASE"
echo "   - CNN Detection: $USE_CNN_DETECTION"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads student_images logs

# Set proper permissions
chmod 755 uploads student_images
chmod 755 logs

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "📦 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev python3-venv
    sudo apt-get install -y libmysqlclient-dev build-essential
    sudo apt-get install -y cmake libopenblas-dev liblapack-dev
    sudo apt-get install -y libx11-dev libgtk-3-dev
fi

# Install Python dependencies
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Additional production packages
pip install gunicorn supervisor

# Setup MySQL database
echo "🗄️  Setting up database..."
if command -v mysql &> /dev/null; then
    echo "Creating database and user..."
    mysql -u root -p << EOF
CREATE DATABASE IF NOT EXISTS $MYSQL_DATABASE CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '$MYSQL_USER'@'localhost' IDENTIFIED BY '$MYSQL_PASSWORD';
GRANT ALL PRIVILEGES ON $MYSQL_DATABASE.* TO '$MYSQL_USER'@'localhost';
FLUSH PRIVILEGES;
EOF
else
    echo "⚠️  MySQL not found. Please install MySQL and create database manually."
fi

# Initialize database
echo "🔧 Initializing database schema..."
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database tables created successfully!')
"

# Create Gunicorn configuration
echo "⚙️  Creating Gunicorn configuration..."
cat > gunicorn.conf.py << 'EOF'
# Gunicorn configuration file
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True
user = "www-data"
group = "www-data"
tmp_upload_dir = None
logfile = "logs/gunicorn.log"
loglevel = "info"
access_logfile = "logs/access.log"
error_logfile = "logs/error.log"
capture_output = True
enable_stdio_inheritance = True
EOF

# Create systemd service file
echo "🔄 Creating systemd service..."
sudo tee /etc/systemd/system/attendance-system.service > /dev/null << EOF
[Unit]
Description=Group Image-based Class Attendance System
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=FLASK_ENV=$FLASK_ENV
Environment=SECRET_KEY=$SECRET_KEY
Environment=MYSQL_HOST=$MYSQL_HOST
Environment=MYSQL_USER=$MYSQL_USER
Environment=MYSQL_PASSWORD=$MYSQL_PASSWORD
Environment=MYSQL_DATABASE=$MYSQL_DATABASE
Environment=USE_CNN_DETECTION=$USE_CNN_DETECTION
ExecStart=$(pwd)/venv/bin/gunicorn --config gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create Nginx configuration
echo "🌐 Creating Nginx configuration..."
sudo tee /etc/nginx/sites-available/attendance-system << 'EOF'
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /static {
        alias /path/to/your/app/static;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    location /uploads {
        alias /path/to/your/app/uploads;
        expires 7d;
    }

    location /student_images {
        alias /path/to/your/app/student_images;
        expires 7d;
    }
}
EOF

# Enable Nginx site
if command -v nginx &> /dev/null; then
    sudo ln -sf /etc/nginx/sites-available/attendance-system /etc/nginx/sites-enabled/
    sudo nginx -t
    echo "✅ Nginx configuration created. Please update server_name and paths."
fi

# Set proper ownership
sudo chown -R www-data:www-data uploads student_images logs

# Start services
echo "🚀 Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable attendance-system
sudo systemctl start attendance-system

# Check service status
echo "🔍 Checking service status..."
sudo systemctl status attendance-system --no-pager

# Setup log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/attendance-system << 'EOF'
/path/to/your/app/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload attendance-system
    endscript
}
EOF

# Create backup script
echo "💾 Creating backup script..."
cat > backup.sh << 'EOF'
#!/bin/bash
# Backup script for Attendance System

BACKUP_DIR="/var/backups/attendance-system"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Database backup
mysqldump -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE > "$BACKUP_DIR/database_$DATE.sql"

# Files backup
tar -czf "$BACKUP_DIR/files_$DATE.tar.gz" uploads student_images

# Keep only last 30 days of backups
find "$BACKUP_DIR" -type f -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

chmod +x backup.sh

# Setup cron job for backups
echo "⏰ Setting up automated backups..."
(crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/backup.sh >> logs/backup.log 2>&1") | crontab -

# Security recommendations
echo "🔒 Security recommendations:"
echo "   1. Change default passwords"
echo "   2. Setup SSL certificate with Let's Encrypt"
echo "   3. Configure firewall rules"
echo "   4. Regular security updates"
echo "   5. Monitor log files"

# Performance tuning suggestions
echo "⚡ Performance tuning:"
echo "   1. Adjust Gunicorn workers based on CPU cores"
echo "   2. Configure MySQL query cache"
echo "   3. Use Redis for session storage (optional)"
echo "   4. Setup CDN for static files (optional)"

echo "✅ Deployment completed successfully!"
echo ""
echo "🔧 Next steps:"
echo "   1. Update Nginx server_name in /etc/nginx/sites-available/attendance-system"
echo "   2. Update file paths in Nginx configuration"
echo "   3. Restart Nginx: sudo systemctl restart nginx"
echo "   4. Test the application at http://your-server-ip:8000"
echo "   5. Setup SSL certificate for production use"
echo ""
echo "📊 Service management:"
echo "   - Start: sudo systemctl start attendance-system"
echo "   - Stop: sudo systemctl stop attendance-system"
echo "   - Restart: sudo systemctl restart attendance-system"
echo "   - Status: sudo systemctl status attendance-system"
echo "   - Logs: sudo journalctl -u attendance-system -f"
echo ""
echo "🏁 Deployment script completed!"
