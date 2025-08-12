#!/usr/bin/env python3
"""
Alert System - Handles notifications and alerts
"""

import smtplib
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    id: str
    type: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    source: str
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AlertSystem:
    """Comprehensive alert and notification system"""
    
    def __init__(self, email: str = None, webhook: str = None):
        self.email = email
        self.webhook = webhook
        self.alerts_history = []
        self.is_enabled = True
        
        # Email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_password = None  # Set via environment variable
        
        # Telegram configuration
        self.telegram_bot_token = None
        self.telegram_chat_id = None
        
        # Slack configuration
        self.slack_webhook_url = None
        
        # Discord configuration
        self.discord_webhook_url = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_daily_loss': -5000,
            'max_drawdown': -0.15,
            'consecutive_losses': 5,
            'position_size_limit': 0.1,
            'system_error_count': 3
        }
        
        logger.info("Alert System initialized")
    
    async def send_alert(self, alert_type: str, message: str, data: Dict = None):
        """Send alert through all configured channels"""
        try:
            alert = Alert(
                id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=alert_type.upper(),
                message=message,
                timestamp=datetime.now(),
                source="TradingBot",
                data=data or {}
            )
            
            # Store alert
            self.alerts_history.append(alert)
            
            # Keep only last 1000 alerts
            if len(self.alerts_history) > 1000:
                self.alerts_history = self.alerts_history[-1000:]
            
            if not self.is_enabled:
                return
            
            # Send through all configured channels
            tasks = []
            
            if self.email:
                tasks.append(self._send_email_alert(alert))
            
            if self.webhook:
                tasks.append(self._send_webhook_alert(alert))
            
            if self.telegram_bot_token and self.telegram_chat_id:
                tasks.append(self._send_telegram_alert(alert))
            
            if self.slack_webhook_url:
                tasks.append(self._send_slack_alert(alert))
            
            if self.discord_webhook_url:
                tasks.append(self._send_discord_alert(alert))
            
            # Execute all notifications concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info(f"Alert sent: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            if not self.email or not self.email_password:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.email
            msg['Subject'] = f"Trading Bot Alert - {alert.type}"
            
            # Email body
            body = f"""
            Trading Bot Alert
            
            Type: {alert.type}
            Time: {alert.timestamp}
            Source: {alert.source}
            
            Message: {alert.message}
            
            Additional Data:
            {json.dumps(alert.data, indent=2) if alert.data else 'None'}
            
            ---
            Automated Trading Bot Alert System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email, self.email, text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            if not self.webhook:
                return
            
            payload = {
                'id': alert.id,
                'type': alert.type,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'data': alert.data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 200:
                        logger.error(f"Webhook alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                return
            
            # Format message
            emoji_map = {
                'INFO': '📊',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }
            
            emoji = emoji_map.get(alert.type, '📢')
            
            message = f"""
{emoji} *Trading Bot Alert*

*Type:* {alert.type}
*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

*Message:* {alert.message}
            """
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Telegram alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            if not self.slack_webhook_url:
                return
            
            # Color coding
            color_map = {
                'INFO': '#36a64f',      # Green
                'WARNING': '#ff9500',   # Orange
                'ERROR': '#ff0000',     # Red
                'CRITICAL': '#8b0000'   # Dark Red
            }
            
            color = color_map.get(alert.type, '#36a64f')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f'Trading Bot Alert - {alert.type}',
                    'text': alert.message,
                    'fields': [
                        {
                            'title': 'Time',
                            'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        },
                        {
                            'title': 'Source',
                            'value': alert.source,
                            'short': True
                        }
                    ],
                    'footer': 'Trading Bot Alert System',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Slack alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    async def _send_discord_alert(self, alert: Alert):
        """Send Discord alert"""
        try:
            if not self.discord_webhook_url:
                return
            
            # Color coding (decimal values)
            color_map = {
                'INFO': 3581519,      # Green
                'WARNING': 16753920,  # Orange
                'ERROR': 16711680,    # Red
                'CRITICAL': 9109504   # Dark Red
            }
            
            color = color_map.get(alert.type, 3581519)
            
            embed = {
                'title': f'Trading Bot Alert - {alert.type}',
                'description': alert.message,
                'color': color,
                'timestamp': alert.timestamp.isoformat(),
                'fields': [
                    {
                        'name': 'Source',
                        'value': alert.source,
                        'inline': True
                    }
                ],
                'footer': {
                    'text': 'Trading Bot Alert System'
                }
            }
            
            payload = {
                'embeds': [embed]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook_url,
                    json=payload
                ) as response:
                    if response.status not in [200, 204]:
                        logger.error(f"Discord alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
    
    async def send_trade_alert(self, trade_data: Dict):
        """Send trade-specific alert"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            side = trade_data.get('side', 'Unknown')
            quantity = trade_data.get('quantity', 0)
            price = trade_data.get('price', 0)
            pnl = trade_data.get('pnl', 0)
            
            message = f"Trade Executed: {side} {quantity} {symbol} @ ${price:.2f}"
            if pnl != 0:
                message += f" | PnL: ${pnl:.2f}"
            
            alert_type = "INFO"
            if pnl < -1000:
                alert_type = "WARNING"
            elif pnl < -2000:
                alert_type = "ERROR"
            
            await self.send_alert(alert_type, message, trade_data)
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
    
    async def send_risk_alert(self, risk_data: Dict):
        """Send risk management alert"""
        try:
            risk_type = risk_data.get('type', 'Unknown')
            severity = risk_data.get('severity', 'INFO')
            message = risk_data.get('message', 'Risk alert triggered')
            
            await self.send_alert(severity, f"Risk Alert - {risk_type}: {message}", risk_data)
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
    
    async def send_system_alert(self, system_data: Dict):
        """Send system status alert"""
        try:
            status = system_data.get('status', 'Unknown')
            message = system_data.get('message', 'System status update')
            
            alert_type = "INFO"
            if status in ['ERROR', 'CRITICAL', 'DOWN']:
                alert_type = "CRITICAL"
            elif status in ['WARNING', 'DEGRADED']:
                alert_type = "WARNING"
            
            await self.send_alert(alert_type, f"System Alert: {message}", system_data)
            
        except Exception as e:
            logger.error(f"Error sending system alert: {e}")
    
    async def send_performance_alert(self, performance_data: Dict):
        """Send performance-related alert"""
        try:
            metric = performance_data.get('metric', 'Unknown')
            value = performance_data.get('value', 0)
            threshold = performance_data.get('threshold', 0)
            
            message = f"Performance Alert - {metric}: {value} (threshold: {threshold})"
            
            alert_type = "WARNING"
            if abs(value) > abs(threshold) * 1.5:
                alert_type = "ERROR"
            
            await self.send_alert(alert_type, message, performance_data)
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
    
    def configure_email(self, email: str, password: str, smtp_server: str = None, smtp_port: int = None):
        """Configure email settings"""
        self.email = email
        self.email_password = password
        if smtp_server:
            self.smtp_server = smtp_server
        if smtp_port:
            self.smtp_port = smtp_port
    
    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram settings"""
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
    
    def configure_slack(self, webhook_url: str):
        """Configure Slack settings"""
        self.slack_webhook_url = webhook_url
    
    def configure_discord(self, webhook_url: str):
        """Configure Discord settings"""
        self.discord_webhook_url = webhook_url
    
    def set_alert_thresholds(self, thresholds: Dict[str, Any]):
        """Set alert thresholds"""
        self.alert_thresholds.update(thresholds)
    
    def get_alerts_history(self, limit: int = 100, alert_type: str = None) -> List[Dict]:
        """Get alerts history"""
        alerts = self.alerts_history
        
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type.upper()]
        
        # Convert to dictionaries and limit
        alerts_dict = []
        for alert in alerts[-limit:]:
            alerts_dict.append({
                'id': alert.id,
                'type': alert.type,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'data': alert.data
            })
        
        return alerts_dict
    
    def clear_alerts_history(self):
        """Clear alerts history"""
        self.alerts_history = []
        logger.info("Alerts history cleared")
    
    def enable_alerts(self):
        """Enable alert system"""
        self.is_enabled = True
        logger.info("Alert system enabled")
    
    def disable_alerts(self):
        """Disable alert system"""
        self.is_enabled = False
        logger.info("Alert system disabled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {
            'enabled': self.is_enabled,
            'email_configured': bool(self.email and self.email_password),
            'webhook_configured': bool(self.webhook),
            'telegram_configured': bool(self.telegram_bot_token and self.telegram_chat_id),
            'slack_configured': bool(self.slack_webhook_url),
            'discord_configured': bool(self.discord_webhook_url),
            'total_alerts': len(self.alerts_history),
            'alert_thresholds': self.alert_thresholds
        }
