#!/bin/bash

# Advanced AI Trading Bot Startup Script
echo "🚀 Starting Advanced AI Trading Bot System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models data frontend/dist

# Set permissions
chmod +x start.sh
chmod -R 755 logs models data

# Pull latest images
echo "📥 Pulling Docker images..."
docker-compose pull

# Build the application
echo "🔨 Building application..."
docker-compose build --no-cache

# Start the services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Display access information
echo ""
echo "✅ Advanced AI Trading Bot is now running!"
echo ""
echo "🌐 Access Points:"
echo "   • Trading Bot Dashboard: http://localhost:8000"
echo "   • MongoDB Express: http://localhost:8081 (admin/admin123)"
echo "   • Redis Commander: http://localhost:8082"
echo "   • Kafka UI: http://localhost:8080"
echo "   • Grafana: http://localhost:3000 (admin/admin123)"
echo "   • Prometheus: http://localhost:9090"
echo ""
echo "📊 Real-time Features:"
echo "   • WebSocket connection for live updates"
echo "   • Real-time market data streaming"
echo "   • Live portfolio tracking"
echo "   • Instant trade notifications"
echo ""
echo "🤖 AI Features:"
echo "   • 95% target success rate"
echo "   • Ensemble ML models"
echo "   • Real-time sentiment analysis"
echo "   • Advanced risk management"
echo ""
echo "📈 To monitor logs: docker-compose logs -f trading-bot"
echo "🛑 To stop: docker-compose down"
echo ""
echo "Happy Trading! 💰"
