"""
Centralized Logging System for Trader-X
"""
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from config.supabase_config import supabase_manager
from config.trading_config import DatabaseConfig
import json

class TraderXLogger:
    def __init__(self, name: str = "TraderX"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(f'trader_x_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_to_database(self, level: str, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log message to Supabase database"""
        try:
            # Skip database logging if client is not available
            if not supabase_manager.client:
                return
            
            # Sanitize data to ensure JSON serialization
            sanitized_data = None
            if data:
                try:
                    # Convert to JSON and back to ensure serializable
                    sanitized_data = json.loads(json.dumps(data, default=str))
                except Exception:
                    sanitized_data = {'error': 'Data not serializable', 'original_type': str(type(data))}
                
            log_entry = {
                'level': level.upper(),
                'module': module,
                'message': str(message)[:1000],  # Limit message length
                'data': sanitized_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Use insert with error handling
            result = supabase_manager.client.table(DatabaseConfig.SYSTEM_LOGS_TABLE).insert(log_entry).execute()
            
        except Exception as e:
            # Silently fail database logging to avoid disrupting main operations
            pass
    
    def info(self, message: str, module: str = "CORE", data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(f"[{module}] {message}")
        self.log_to_database("INFO", module, message, data)
    
    def warning(self, message: str, module: str = "CORE", data: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(f"[{module}] {message}")
        self.log_to_database("WARNING", module, message, data)
    
    def error(self, message: str, module: str = "CORE", data: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self.logger.error(f"[{module}] {message}")
        self.log_to_database("ERROR", module, message, data)
    
    def debug(self, message: str, module: str = "CORE", data: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(f"[{module}] {message}")
        self.log_to_database("DEBUG", module, message, data)
    
    def trade_log(self, trade_data: Dict[str, Any], message: str):
        """Special logging for trade events"""
        self.info(f"TRADE: {message}", "TRADING", trade_data)
    
    def phase_log(self, phase: str, symbol: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log phase-specific events"""
        self.info(f"PHASE_{phase}: {symbol} - {message}", f"PHASE_{phase}", data)
    
    def ai_decision_log(self, symbol: str, decision: str, reasoning: str, confidence: float, data: Optional[Dict[str, Any]] = None):
        """Log AI decision making"""
        ai_data = {
            'symbol': symbol,
            'decision': decision,
            'reasoning': reasoning,
            'confidence': confidence,
            'additional_data': data
        }
        self.info(f"AI_DECISION: {symbol} - {decision} (confidence: {confidence:.2f})", "AI_ENGINE", ai_data)

# Global logger instance
logger = TraderXLogger()
