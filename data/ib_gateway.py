"""
Interactive Brokers Gateway Integration
Provides real-time market data and trading capabilities
"""
import socket
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from core.logger import logger
import json

@dataclass
class IBContract:
    """IB Contract specification"""
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str = ""

class IBGatewayConnector:
    """
    Interactive Brokers Gateway Connector
    Handles connection and data requests to IB Gateway
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.socket = None
        self.connected = False
        self.is_connected = False  # Add this property for compatibility
        self.next_req_id = 1
        self.market_data_cache = {}
        self.fundamental_data_cache = {}
        self.historical_data_cache = {}
        self.data_lock = threading.Lock()
        
        # Message handlers
        self.message_handlers = {
            1: self._handle_tick_price,
            2: self._handle_tick_size,
            4: self._handle_tick_string,
            5: self._handle_tick_efp,
            6: self._handle_tick_generic,
            9: self._handle_tick_snapshot_end,
            17: self._handle_historical_data,
            18: self._handle_historical_data_end,
            59: self._handle_fundamental_data,
        }
        
    def connect(self) -> bool:
        """Connect to IB Gateway"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
            # Send connection message
            version = "v100..151"
            msg = f"API\0{version}\0"
            self.socket.send(msg.encode())
            
            # Send client handshake
            handshake = f"{self.client_id}\0"
            self.socket.send(handshake.encode())
            
            self.connected = True
            self.is_connected = True  # Sync both properties
            logger.info(f"Connected to IB Gateway at {self.host}:{self.port}", "IB_GATEWAY")
            
            # Start message listener thread
            self.listener_thread = threading.Thread(target=self._message_listener, daemon=True)
            self.listener_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}", "IB_GATEWAY")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.socket:
            try:
                self.socket.close()
                self.connected = False
                logger.info("Disconnected from IB Gateway", "IB_GATEWAY")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}", "IB_GATEWAY")
    
    def _get_next_req_id(self) -> int:
        """Get next request ID"""
        req_id = self.next_req_id
        self.next_req_id += 1
        return req_id
    
    def _send_message(self, message: str):
        """Send message to IB Gateway"""
        if not self.connected or not self.socket:
            raise Exception("Not connected to IB Gateway")
        
        try:
            self.socket.send(message.encode())
        except Exception as e:
            logger.error(f"Failed to send message: {e}", "IB_GATEWAY")
            raise
    
    def _message_listener(self):
        """Listen for incoming messages from IB Gateway"""
        buffer = b""
        
        while self.connected:
            try:
                data = self.socket.recv(4096)
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages
                while b'\0' in buffer:
                    message, buffer = buffer.split(b'\0', 1)
                    if message:
                        self._process_message(message.decode())
                        
            except Exception as e:
                if self.connected:
                    logger.error(f"Message listener error: {e}", "IB_GATEWAY")
                break
    
    def _process_message(self, message: str):
        """Process incoming message"""
        try:
            parts = message.split('\0')
            if len(parts) < 2:
                return
            
            msg_type = int(parts[0])
            version = int(parts[1])
            
            if msg_type in self.message_handlers:
                self.message_handlers[msg_type](parts[2:])
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", "IB_GATEWAY")
    
    def request_market_data(self, symbol: str) -> int:
        """Request real-time market data for a symbol"""
        try:
            req_id = self._get_next_req_id()
            contract = IBContract(symbol)
            
            # Build market data request message
            msg_parts = [
                "1",  # REQ_MKT_DATA
                "11", # version
                str(req_id),
                contract.symbol,
                contract.sec_type,
                "",   # expiry
                "0",  # strike
                "",   # right
                "",   # multiplier
                contract.exchange,
                contract.primary_exchange,
                contract.currency,
                "",   # local_symbol
                "",   # trading_class
                "0",  # include_expired
                "",   # market_data_options
            ]
            
            message = "\0".join(msg_parts) + "\0"
            self._send_message(message)
            
            logger.debug(f"Requested market data for {symbol} (req_id: {req_id})", "IB_GATEWAY")
            return req_id
            
        except Exception as e:
            logger.error(f"Failed to request market data for {symbol}: {e}", "IB_GATEWAY")
            return -1
    
    def request_historical_data(self, symbol: str, duration: str = "1 Y", bar_size: str = "1 day") -> int:
        """Request historical data for a symbol"""
        try:
            req_id = self._get_next_req_id()
            contract = IBContract(symbol)
            
            end_time = datetime.now().strftime("%Y%m%d %H:%M:%S")
            
            # Build historical data request message
            msg_parts = [
                "20",  # REQ_HISTORICAL_DATA
                "6",   # version
                str(req_id),
                contract.symbol,
                contract.sec_type,
                "",    # expiry
                "0",   # strike
                "",    # right
                "",    # multiplier
                contract.exchange,
                contract.primary_exchange,
                contract.currency,
                "",    # local_symbol
                "",    # trading_class
                "0",   # include_expired
                end_time,
                bar_size,
                duration,
                "1",   # use_rth (regular trading hours)
                "TRADES",  # what_to_show
                "1",   # format_date
                "0",   # keep_up_to_date
                "",    # chart_options
            ]
            
            message = "\0".join(msg_parts) + "\0"
            self._send_message(message)
            
            logger.debug(f"Requested historical data for {symbol} (req_id: {req_id})", "IB_GATEWAY")
            return req_id
            
        except Exception as e:
            logger.error(f"Failed to request historical data for {symbol}: {e}", "IB_GATEWAY")
            return -1
    
    def request_fundamental_data(self, symbol: str) -> int:
        """Request fundamental data for a symbol"""
        try:
            req_id = self._get_next_req_id()
            contract = IBContract(symbol)
            
            # Build fundamental data request message
            msg_parts = [
                "52",  # REQ_FUNDAMENTAL_DATA
                "2",   # version
                str(req_id),
                contract.symbol,
                contract.sec_type,
                "",    # expiry
                "0",   # strike
                "",    # right
                "",    # multiplier
                contract.exchange,
                contract.primary_exchange,
                contract.currency,
                "",    # local_symbol
                "",    # trading_class
                "0",   # include_expired
                "ReportsFinSummary",  # report_type
            ]
            
            message = "\0".join(msg_parts) + "\0"
            self._send_message(message)
            
            logger.debug(f"Requested fundamental data for {symbol} (req_id: {req_id})", "IB_GATEWAY")
            return req_id
            
        except Exception as e:
            logger.error(f"Failed to request fundamental data for {symbol}: {e}", "IB_GATEWAY")
            return -1
    
    def get_market_data(self, symbol: str, timeout: int = 10) -> Dict[str, Any]:
        """Get current market data for a symbol"""
        try:
            # Check cache first
            with self.data_lock:
                if symbol in self.market_data_cache:
                    data = self.market_data_cache[symbol]
                    # Return cached data if recent (within 1 minute)
                    if datetime.now() - data.get('timestamp', datetime.min) < timedelta(minutes=1):
                        return data
            
            # Request fresh data
            req_id = self.request_market_data(symbol)
            if req_id == -1:
                return {}
            
            # Wait for data with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self.data_lock:
                    if symbol in self.market_data_cache:
                        data = self.market_data_cache[symbol]
                        if data.get('req_id') == req_id:
                            return data
                time.sleep(0.1)
            
            logger.warning(f"Timeout waiting for market data for {symbol}", "IB_GATEWAY")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}", "IB_GATEWAY")
            return {}
    
    def get_historical_data(self, symbol: str, duration: str = "1 Y", bar_size: str = "1 day", timeout: int = 30) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{duration}_{bar_size}"
            with self.data_lock:
                if cache_key in self.historical_data_cache:
                    data = self.historical_data_cache[cache_key]
                    # Return cached data if recent (within 5 minutes)
                    if datetime.now() - data.get('timestamp', datetime.min) < timedelta(minutes=5):
                        return data.get('dataframe', pd.DataFrame())
            
            # Request fresh data
            req_id = self.request_historical_data(symbol, duration, bar_size)
            if req_id == -1:
                return pd.DataFrame()
            
            # Wait for data with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self.data_lock:
                    if cache_key in self.historical_data_cache:
                        data = self.historical_data_cache[cache_key]
                        if data.get('req_id') == req_id and data.get('complete', False):
                            return data.get('dataframe', pd.DataFrame())
                time.sleep(0.1)
            
            logger.warning(f"Timeout waiting for historical data for {symbol}", "IB_GATEWAY")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}", "IB_GATEWAY")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str, timeout: int = 5) -> Dict[str, Any]:
        """Get fundamental data for a symbol"""
        try:
            # Check cache first
            with self.data_lock:
                if symbol in self.fundamental_data_cache:
                    data = self.fundamental_data_cache[symbol]
                    # Return cached data if recent (within 1 hour)
                    if datetime.now() - data.get('timestamp', datetime.min) < timedelta(hours=1):
                        return data
            
            # Request fresh data
            req_id = self.request_fundamental_data(symbol)
            if req_id == -1:
                return {}
            
            # Wait for data with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                with self.data_lock:
                    if symbol in self.fundamental_data_cache:
                        data = self.fundamental_data_cache[symbol]
                        if data.get('req_id') == req_id:
                            return data
                time.sleep(0.1)
            
            logger.warning(f"Timeout waiting for fundamental data for {symbol}", "IB_GATEWAY")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get fundamental data for {symbol}: {e}", "IB_GATEWAY")
            return {}
    
    # Message Handlers
    def _handle_tick_price(self, fields: List[str]):
        """Handle tick price message"""
        try:
            req_id = int(fields[0])
            tick_type = int(fields[1])
            price = float(fields[2])
            
            # Find symbol for this req_id
            symbol = self._find_symbol_by_req_id(req_id)
            if not symbol:
                return
            
            with self.data_lock:
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {'req_id': req_id, 'timestamp': datetime.now()}
                
                data = self.market_data_cache[symbol]
                
                # Map tick types to fields
                if tick_type == 1:  # BID
                    data['bid'] = price
                elif tick_type == 2:  # ASK
                    data['ask'] = price
                elif tick_type == 4:  # LAST
                    data['last'] = price
                elif tick_type == 6:  # HIGH
                    data['high'] = price
                elif tick_type == 7:  # LOW
                    data['low'] = price
                elif tick_type == 9:  # CLOSE
                    data['close'] = price
                
                data['timestamp'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error handling tick price: {e}", "IB_GATEWAY")
    
    def _handle_tick_size(self, fields: List[str]):
        """Handle tick size message"""
        try:
            req_id = int(fields[0])
            tick_type = int(fields[1])
            size = int(fields[2])
            
            symbol = self._find_symbol_by_req_id(req_id)
            if not symbol:
                return
            
            with self.data_lock:
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {'req_id': req_id, 'timestamp': datetime.now()}
                
                data = self.market_data_cache[symbol]
                
                # Map tick types to fields
                if tick_type == 0:  # BID_SIZE
                    data['bid_size'] = size
                elif tick_type == 3:  # ASK_SIZE
                    data['ask_size'] = size
                elif tick_type == 5:  # LAST_SIZE
                    data['last_size'] = size
                elif tick_type == 8:  # VOLUME
                    data['volume'] = size
                
                data['timestamp'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error handling tick size: {e}", "IB_GATEWAY")
    
    def _handle_tick_string(self, fields: List[str]):
        """Handle tick string message"""
        try:
            req_id = int(fields[0])
            tick_type = int(fields[1])
            value = fields[2]
            
            symbol = self._find_symbol_by_req_id(req_id)
            if not symbol:
                return
            
            with self.data_lock:
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {'req_id': req_id, 'timestamp': datetime.now()}
                
                data = self.market_data_cache[symbol]
                
                # Map tick types to fields
                if tick_type == 45:  # LAST_TIMESTAMP
                    data['last_timestamp'] = value
                
                data['timestamp'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error handling tick string: {e}", "IB_GATEWAY")
    
    def _handle_tick_efp(self, fields: List[str]):
        """Handle tick EFP message"""
        pass  # Not needed for basic functionality
    
    def _handle_tick_generic(self, fields: List[str]):
        """Handle tick generic message"""
        pass  # Not needed for basic functionality
    
    def _handle_tick_snapshot_end(self, fields: List[str]):
        """Handle tick snapshot end message"""
        req_id = int(fields[0])
        logger.debug(f"Tick snapshot end for req_id: {req_id}", "IB_GATEWAY")
    
    def _handle_historical_data(self, fields: List[str]):
        """Handle historical data message"""
        try:
            req_id = int(fields[0])
            date = fields[1]
            open_price = float(fields[2])
            high = float(fields[3])
            low = float(fields[4])
            close = float(fields[5])
            volume = int(fields[6])
            
            symbol = self._find_symbol_by_req_id(req_id)
            if not symbol:
                return
            
            # Find cache key for this req_id
            cache_key = None
            with self.data_lock:
                for key, data in self.historical_data_cache.items():
                    if data.get('req_id') == req_id:
                        cache_key = key
                        break
            
            if not cache_key:
                return
            
            with self.data_lock:
                if cache_key not in self.historical_data_cache:
                    self.historical_data_cache[cache_key] = {
                        'req_id': req_id,
                        'data': [],
                        'timestamp': datetime.now(),
                        'complete': False
                    }
                
                data_entry = {
                    'Date': date,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                }
                
                self.historical_data_cache[cache_key]['data'].append(data_entry)
                
        except Exception as e:
            logger.error(f"Error handling historical data: {e}", "IB_GATEWAY")
    
    def _handle_historical_data_end(self, fields: List[str]):
        """Handle historical data end message"""
        try:
            req_id = int(fields[0])
            
            # Find cache key for this req_id and mark as complete
            with self.data_lock:
                for key, data in self.historical_data_cache.items():
                    if data.get('req_id') == req_id:
                        # Convert data to DataFrame
                        df = pd.DataFrame(data['data'])
                        if not df.empty:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        
                        data['dataframe'] = df
                        data['complete'] = True
                        break
            
            logger.debug(f"Historical data complete for req_id: {req_id}", "IB_GATEWAY")
            
        except Exception as e:
            logger.error(f"Error handling historical data end: {e}", "IB_GATEWAY")
    
    def _handle_fundamental_data(self, fields: List[str]):
        """Handle fundamental data message"""
        try:
            req_id = int(fields[0])
            xml_data = fields[1]
            
            symbol = self._find_symbol_by_req_id(req_id)
            if not symbol:
                return
            
            # Parse XML data (simplified - would need proper XML parsing)
            fundamental_data = {
                'req_id': req_id,
                'xml_data': xml_data,
                'timestamp': datetime.now()
            }
            
            with self.data_lock:
                self.fundamental_data_cache[symbol] = fundamental_data
            
            logger.debug(f"Received fundamental data for {symbol}", "IB_GATEWAY")
            
        except Exception as e:
            logger.error(f"Error handling fundamental data: {e}", "IB_GATEWAY")
    
    def _find_symbol_by_req_id(self, req_id: int) -> Optional[str]:
        """Find symbol associated with a request ID"""
        with self.data_lock:
            for symbol, data in self.market_data_cache.items():
                if data.get('req_id') == req_id:
                    return symbol
        return None

# Global IB Gateway connector instance
ib_gateway = IBGatewayConnector()
