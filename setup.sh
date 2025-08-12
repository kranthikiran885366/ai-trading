#!/bin/bash

# Advanced AI Trading Bot Setup Script
echo "🚀 Setting up Advanced AI Trading Bot..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
        print_status "Python $PYTHON_VERSION found ✓"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_status "Docker found ✓"
else
    print_warning "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    print_status "Docker installed. Please log out and back in."
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_status "Docker Compose found ✓"
else
    print_warning "Docker Compose not found. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    print_status "Docker Compose installed ✓"
fi

# Create project structure
print_status "Creating project structure..."
mkdir -p {logs,models,data,config,scripts,tests}
mkdir -p {ai_engine,trading_engine,data_streams,brokers,utils}
mkdir -p frontend/{src,public,build}

# Set permissions
chmod +x start.sh
chmod +x setup.sh
chmod -R 755 logs models data

# Create environment file
print_status "Creating environment configuration..."
if [ ! -f .env ]; then
    cat > .env << EOF
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/trading_bot
REDIS_HOST=localhost
REDIS_PORT=6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# API Keys (Replace with your actual keys)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_api_secret_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
NEWSAPI_KEY=your_newsapi_key_here

# Trading Configuration
INITIAL_CAPITAL=100000.0
MAX_DAILY_LOSS=2000.0
MAX_DRAWDOWN=0.10
MAX_POSITION_SIZE=0.05
TARGET_SUCCESS_RATE=0.95

# AI Configuration
AI_CONFIDENCE_THRESHOLD=0.85
ENSEMBLE_MODELS=5
RETRAIN_INTERVAL_HOURS=6

# Notification Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
TO_EMAILS=alerts@yourdomain.com

WEBHOOK_URL=https://your-webhook-url.com/alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your_webhook

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Development
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production
EOF
    print_status "Environment file created. Please edit .env with your API keys."
else
    print_status "Environment file already exists ✓"
fi

# Install Python dependencies (if not using Docker)
if [[ "$1" == "--local" ]]; then
    print_status "Installing Python dependencies locally..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_status "Python dependencies installed ✓"
fi

# Build frontend
print_status "Setting up frontend..."
if [ -d "frontend" ]; then
    cd frontend
    
    # Check if Node.js is installed
    if command -v node &> /dev/null; then
        print_status "Node.js found ✓"
        
        # Install dependencies
        if [ ! -d "node_modules" ]; then
            npm install
            print_status "Frontend dependencies installed ✓"
        fi
        
        # Build frontend
        npm run build
        print_status "Frontend built ✓"
    else
        print_warning "Node.js not found. Frontend will not be built."
        print_status "Install Node.js and run 'cd frontend && npm install && npm run build'"
    fi
    
    cd ..
fi

# Create systemd service (optional)
if [[ "$1" == "--service" ]]; then
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/trading-bot.service > /dev/null << EOF
[Unit]
Description=Advanced AI Trading Bot
After=network.target
Requires=docker.service

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/start.sh
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable trading-bot.service
    print_status "Systemd service created and enabled ✓"
fi

# Create monitoring configuration
print_status "Creating monitoring configuration..."
mkdir -p monitoring/{prometheus,grafana}

# Prometheus configuration
cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Create backup script
print_status "Creating backup script..."
cat > scripts/backup.sh << 'EOF'
#!/bin/bash

# Backup script for trading bot data
BACKUP_DIR="/backup/trading-bot"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker exec trading-bot-mongodb mongodump --out $BACKUP_DIR/mongodb_$DATE

# Backup models
cp -r models $BACKUP_DIR/models_$DATE

# Backup logs
cp -r logs $BACKUP_DIR/logs_$DATE

# Backup configuration
cp .env $BACKUP_DIR/env_$DATE

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x scripts/backup.sh

# Create update script
print_status "Creating update script..."
cat > scripts/update.sh << 'EOF'
#!/bin/bash

echo "🔄 Updating Advanced AI Trading Bot..."

# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose down
docker-compose up -d

echo "✅ Update completed"
EOF

chmod +x scripts/update.sh

# Final setup steps
print_status "Performing final setup..."

# Create initial model directories
mkdir -p models/{ensemble,transformers,risk,sentiment}

# Create log rotation configuration
sudo tee /etc/logrotate.d/trading-bot > /dev/null << EOF
$(pwd)/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF

print_status "Setup completed successfully! 🎉"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Edit .env file with your API keys"
echo "2. Run: ./start.sh"
echo "3. Access dashboard at: http://localhost:8000"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "• Start bot: ./start.sh"
echo "• Stop bot: docker-compose down"
echo "• View logs: docker-compose logs -f trading-bot"
echo "• Backup data: ./scripts/backup.sh"
echo "• Update bot: ./scripts/update.sh"
echo ""
echo -e "${GREEN}Happy Trading! 💰${NC}"
