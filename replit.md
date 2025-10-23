# AI Trading Bot Platform

## Overview

An advanced algorithmic trading system that combines machine learning, multi-broker connectivity, and real-time data processing to execute automated trades. The platform features a Python-based AI engine for signal generation and risk management, a Node.js API gateway for real-time communication, and a Next.js/React frontend for monitoring and control.

The system targets a 95% trading success rate through ensemble AI models, comprehensive risk management, and intelligent order execution across multiple brokers (Binance, Alpaca, Zerodha).

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**: Next.js 14+ with TypeScript, React 18, Material-UI (MUI), and shadcn/ui components. Uses Tailwind CSS for styling with a dark mode theme.

**Design Pattern**: The frontend follows a component-based architecture with separate views for Dashboard, Trading, Portfolio, and Settings. Real-time updates are handled through WebSocket connections, with REST API calls for configuration and historical data.

**State Management**: Local component state with React hooks. Real-time data flows from WebSocket service to parent App component, then distributed to child components via props.

**Routing**: React Router DOM handles client-side navigation between dashboard, trading view, portfolio, and settings pages.

### Backend Architecture

**Dual Backend Design**: The system uses two backend services:

1. **Python Trading Engine** (`main.py`): Handles AI/ML processing, market data streaming, signal generation, risk management, and order execution. Built with FastAPI for async processing and WebSocket support.

2. **Node.js API Gateway** (`node_api_gateway/app.js`): Manages client connections, provides REST endpoints, handles real-time Socket.IO communication, and coordinates between frontend and Python backend.

**Rationale**: Python excels at AI/ML workloads while Node.js provides superior real-time communication and can act as a lightweight coordination layer. This separation allows independent scaling of compute-intensive AI operations versus I/O-bound client connections.

**Communication Flow**: Frontend ↔ Node.js Gateway ↔ Python Trading Engine. The Node.js layer shields clients from direct Python access and provides WebSocket abstraction.

### AI Engine Architecture

**Multi-Model Ensemble Approach**: Combines RandomForest, GradientBoosting, Neural Networks (MLPClassifier), and TensorFlow/PyTorch models to generate trading signals with high confidence.

**Key Components**:
- **Advanced AI Engine** (`ai_engine/advanced_ai_engine.py`): Orchestrates ensemble predictions and signal generation
- **Signal Processor** (`ai_engine/signal_processor.py`): Filters, deduplicates, and prioritizes trading signals
- **Risk Manager** (`ai_engine/risk_manager.py`): Calculates VaR, expected shortfall, drawdown, and enforces risk limits
- **Market Classifier** (`ai_engine/ai_market_classifier/market_regime_classifier.py`): Identifies market regimes (bull, bear, sideways, volatile, crisis)
- **Position Sizer** (`ai_engine/ai_position_sizer/position_size_calculator.py`): Uses Kelly criterion and ML models for optimal position sizing

**Feature Engineering**: Extracts 100+ technical indicators using TA-Lib including moving averages, RSI, MACD, Bollinger Bands, volume indicators, and custom price-based features.

**Model Training**: Separate training scripts for signal models and market classifiers using historical data from yfinance with time-series cross-validation.

### Trading Engine

**Execution Engine** (`trading_engine/execution_engine.py`): Handles order routing, execution, and tracking across multiple brokers with support for various order types (market, limit, stop-loss, trailing stop, OCO).

**Portfolio Manager** (`trading_engine/portfolio_manager.py`): Tracks positions, calculates P&L, maintains portfolio metrics (Sharpe ratio, max drawdown, win rate), and monitors cash balance.

**Multi-Broker Manager** (`brokers/multi_broker_manager.py`): Abstracts broker-specific implementations for Binance and Alpaca, enabling smart order routing based on liquidity, latency, and fees.

### Data Streaming

**Market Data Stream** (`data_streams/market_data_stream.py`): Aggregates real-time market data from multiple sources via WebSocket connections. Maintains current and historical data for analysis.

**News Stream** (`data_streams/news_stream.py`): Fetches financial news from APIs (NewsAPI, Alpha Vantage) and RSS feeds, performs sentiment analysis using transformers/FinBERT to gauge market sentiment.

**Real-Time Processing**: Uses asyncio for concurrent WebSocket management and Kafka for message queuing (optional).

### Risk Management

**Multi-Layer Risk Controls**:
- **Position-level**: Maximum position size (5% default), stop-loss enforcement
- **Portfolio-level**: Maximum drawdown limits (10-15%), daily loss caps
- **Correlation-based**: Monitors portfolio correlation to prevent concentrated risk
- **Dynamic Adjustment**: AI-powered risk predictor forecasts VaR and adjusts exposure

**Risk Metrics Tracking**: Continuously calculates Sharpe ratio, Sortino ratio, beta, Value at Risk (95% and 99%), expected shortfall (CVaR), and concentration risk.

### Database Design

**MongoDB**: Stores trades, signals, performance metrics, and news items. Uses Motor async driver for non-blocking I/O with collections for trades, signals, performance, and news.

**Redis**: Caches real-time data, market prices, and temporary state for fast access. Used for rate limiting and session management in Node.js gateway.

**SQLite**: Local performance tracking in Python monitoring module for quick queries without external dependencies.

## External Dependencies

### Third-Party Services

**Trading Brokers**:
- **Binance** (via `ccxt`, `python-binance`, custom WebSocket): Cryptocurrency trading with testnet support
- **Alpaca** (via `alpaca-trade-api`): US stock trading with paper trading environment
- **Zerodha** (custom API implementation): Indian stock market access

**Market Data Providers**:
- **yfinance**: Historical price data and company fundamentals
- **Alpha Vantage**: News and fundamental data (API key required)
- **Finnhub**: Real-time news and company data
- **NewsAPI**: Global financial news aggregation

**AI/ML Frameworks**:
- **scikit-learn**: Traditional ML models (RandomForest, GradientBoosting, SVM)
- **TensorFlow/Keras**: Deep learning models including transformers
- **PyTorch**: Custom neural networks for advanced predictions
- **TA-Lib**: Technical analysis indicator calculations
- **Transformers (HuggingFace)**: FinBERT for sentiment analysis

**Message Queuing & Caching**:
- **Kafka** (`kafka-python`): Event streaming for signals and market data
- **Redis**: Caching and pub/sub for real-time updates
- **Celery**: Task queue for async job processing

**Communication**:
- **FastAPI + Uvicorn**: Python async web framework
- **Express.js**: Node.js REST API and middleware
- **Socket.IO**: Bidirectional real-time communication between Node.js and frontend
- **WebSockets** (`websockets`, `ws`): Low-level WebSocket connections to exchanges

**Frontend Libraries**:
- **Next.js**: React framework with server-side rendering
- **Material-UI (MUI)**: React component library for UI
- **shadcn/ui**: Radix UI components with Tailwind styling
- **Recharts**: Data visualization for trading charts
- **Axios**: HTTP client for API requests

**Utilities**:
- **python-dotenv**: Environment variable management
- **YAML**: Configuration file parsing
- **Loguru/Winston**: Advanced logging
- **Pandas/NumPy**: Data manipulation and numerical computing

### API Keys Required

- Binance API key and secret (testnet or production)
- Alpaca API key and secret (paper or live trading)
- Alpha Vantage API key (for news and fundamentals)
- Finnhub API key (for real-time news)
- NewsAPI key (for news aggregation)
- Optional: Telegram bot token, Discord webhooks, email SMTP credentials for notifications

### Environment Configuration

The system expects environment variables for API credentials, database URIs (MongoDB, Redis), SMTP settings for notifications, and feature flags (testnet/production mode). Configuration can be managed through `.env` files or YAML config files.