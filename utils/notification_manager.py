import asyncio
import logging
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class Notification:
    title: str
    message: str
    level: str  # 'info', 'warning', 'error', 'critical'
    timestamp: str
    channels: List[str]  # ['email', 'sms', 'webhook', 'discord']

class NotificationManager:
    """Comprehensive notification system for trading bot alerts"""
    
    def __init__(self):
        self.notification_history = []
        self.channels = {
            'email': self._send_email,
            'sms': self._send_sms,
            'webhook': self._send_webhook,
            'discord': self._send_discord,
            'slack': self._send_slack
        }
        
        # Configuration
        self.config = {
            'email': {
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'username': os.getenv('EMAIL_USERNAME', ''),
                'password': os.getenv('EMAIL_PASSWORD', ''),
                'from_email': os.getenv('FROM_EMAIL', ''),
                'to_emails': os.getenv('TO_EMAILS', '').split(',')
            },
            'webhook': {
                'url': os.getenv('WEBHOOK_URL', ''),
                'headers': {'Content-Type': 'application/json'}
            },
            'discord': {
                'webhook_url': os.getenv('DISCORD_WEBHOOK_URL', '')
            },
            'slack': {
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL', '')
            }
        }
        
        logger.info("Notification Manager initialized")
    
    async def send_alert(self, title: str, message: str, level: str = 'info', channels: List[str] = None):
        """Send alert through specified channels"""
        try:
            if channels is None:
                channels = ['email', 'webhook']  # Default channels
            
            notification = Notification(
                title=title,
                message=message,
                level=level,
                timestamp=datetime.now().isoformat(),
                channels=channels
            )
            
            # Send through each channel
            tasks = []
            for channel in channels:
                if channel in self.channels:
                    tasks.append(self.channels[channel](notification))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store in history
            self.notification_history.append(notification)
            self._cleanup_old_notifications()
            
            logger.info(f"Sent alert '{title}' through {len(channels)} channels")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def _send_email(self, notification: Notification):
        """Send email notification"""
        try:
            if not self.config['email']['username'] or not self.config['email']['to_emails'][0]:
                logger.warning("Email configuration incomplete, skipping email notification")
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_email'] or self.config['email']['username']
            msg['To'] = ', '.join(self.config['email']['to_emails'])
            msg['Subject'] = f"[{notification.level.upper()}] {notification.title}"
            
            # Create HTML body
            html_body = f"""
            <html>
                <body>
                    <h2 style="color: {'red' if notification.level == 'error' else 'orange' if notification.level == 'warning' else 'green'};">
                        {notification.title}
                    </h2>
                    <p><strong>Level:</strong> {notification.level.upper()}</p>
                    <p><strong>Time:</strong> {notification.timestamp}</p>
                    <p><strong>Message:</strong></p>
                    <p>{notification.message}</p>
                    <hr>
                    <p><em>Sent by Advanced AI Trading Bot</em></p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port']) as server:
                server.starttls()
                server.login(self.config['email']['username'], self.config['email']['password'])
                server.send_message(msg)
            
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_sms(self, notification: Notification):
        """Send SMS notification (placeholder - integrate with SMS service)"""
        try:
            # This would integrate with services like Twilio, AWS SNS, etc.
            logger.info("SMS notification would be sent here")
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    async def _send_webhook(self, notification: Notification):
        """Send webhook notification"""
        try:
            if not self.config['webhook']['url']:
                logger.warning("Webhook URL not configured, skipping webhook notification")
                return
            
            payload = {
                'title': notification.title,
                'message': notification.message,
                'level': notification.level,
                'timestamp': notification.timestamp,
                'source': 'AI Trading Bot'
            }
            
            response = requests.post(
                self.config['webhook']['url'],
                json=payload,
                headers=self.config['webhook']['headers'],
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Webhook notification sent successfully")
            else:
                logger.warning(f"Webhook notification failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    async def _send_discord(self, notification: Notification):
        """Send Discord notification"""
        try:
            if not self.config['discord']['webhook_url']:
                logger.warning("Discord webhook URL not configured, skipping Discord notification")
                return
            
            # Color based on level
            color_map = {
                'info': 0x00ff00,      # Green
                'warning': 0xffaa00,   # Orange
                'error': 0xff0000,     # Red
                'critical': 0x800080   # Purple
            }
            
            embed = {
                'title': notification.title,
                'description': notification.message,
                'color': color_map.get(notification.level, 0x00ff00),
                'timestamp': notification.timestamp,
                'footer': {
                    'text': 'AI Trading Bot'
                },
                'fields': [
                    {
                        'name': 'Alert Level',
                        'value': notification.level.upper(),
                        'inline': True
                    }
                ]
            }
            
            payload = {
                'embeds': [embed]
            }
            
            response = requests.post(
                self.config['discord']['webhook_url'],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("Discord notification sent successfully")
            else:
                logger.warning(f"Discord notification failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
    
    async def _send_slack(self, notification: Notification):
        """Send Slack notification"""
        try:
            if not self.config['slack']['webhook_url']:
                logger.warning("Slack webhook URL not configured, skipping Slack notification")
                return
            
            # Color based on level
            color_map = {
                'info': 'good',
                'warning': 'warning',
                'error': 'danger',
                'critical': 'danger'
            }
            
            payload = {
                'text': f"*{notification.title}*",
                'attachments': [
                    {
                        'color': color_map.get(notification.level, 'good'),
                        'fields': [
                            {
                                'title': 'Message',
                                'value': notification.message,
                                'short': False
                            },
                            {
                                'title': 'Level',
                                'value': notification.level.upper(),
                                'short': True
                            },
                            {
                                'title': 'Time',
                                'value': notification.timestamp,
                                'short': True
                            }
                        ],
                        'footer': 'AI Trading Bot',
                        'ts': int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(
                self.config['slack']['webhook_url'],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.warning(f"Slack notification failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    def _cleanup_old_notifications(self):
        """Remove old notifications from history"""
        # Keep only last 1000 notifications
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]
    
    async def send_trade_alert(self, trade_data: Dict[str, Any]):
        """Send trade-specific alert"""
        title = f"Trade Executed: {trade_data.get('symbol', 'Unknown')}"
        message = f"""
        Symbol: {trade_data.get('symbol', 'N/A')}
        Side: {trade_data.get('side', 'N/A')}
        Quantity: {trade_data.get('quantity', 'N/A')}
        Price: ${trade_data.get('price', 'N/A')}
        P&L: ${trade_data.get('pnl', 'N/A')}
        Status: {trade_data.get('status', 'N/A')}
        """
        
        level = 'info' if trade_data.get('pnl', 0) >= 0 else 'warning'
        await self.send_alert(title, message, level)
    
    async def send_performance_alert(self, performance_data: Dict[str, Any]):
        """Send performance-specific alert"""
        success_rate = performance_data.get('success_rate', 0)
        daily_pnl = performance_data.get('daily_pnl', 0)
        
        title = "Performance Update"
        message = f"""
        Success Rate: {success_rate:.2%}
        Daily P&L: ${daily_pnl:.2f}
        Total Trades: {performance_data.get('total_trades', 0)}
        Portfolio Value: ${performance_data.get('portfolio_value', 0):,.2f}
        """
        
        if success_rate < 0.8:
            level = 'warning'
        elif daily_pnl < -1000:
            level = 'error'
        else:
            level = 'info'
        
        await self.send_alert(title, message, level)
    
    async def send_system_alert(self, system_status: Dict[str, Any]):
        """Send system status alert"""
        title = f"System Status: {system_status.get('status', 'Unknown')}"
        message = f"""
        Bot Status: {system_status.get('bot_status', 'Unknown')}
        Database: {system_status.get('database_status', 'Unknown')}
        API Connections: {system_status.get('api_status', 'Unknown')}
        Memory Usage: {system_status.get('memory_usage', 'Unknown')}
        CPU Usage: {system_status.get('cpu_usage', 'Unknown')}
        """
        
        level = 'error' if system_status.get('status') == 'error' else 'info'
        await self.send_alert(title, message, level, channels=['email', 'discord'])
    
    def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        recent_notifications = self.notification_history[-limit:]
        return [
            {
                'title': n.title,
                'message': n.message,
                'level': n.level,
                'timestamp': n.timestamp,
                'channels': n.channels
            }
            for n in recent_notifications
        ]
