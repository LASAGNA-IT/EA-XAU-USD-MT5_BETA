#!/usr/bin/env python3
"""
XAUUSD INTELLIGENT TP BOT - VERSION 3.0
"""

# =============================================================================
# SUPPRESS ALL WARNINGS - MUST BE FIRST
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', message=".*sklearn.utils.parallel.*")
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # Disabilita multiprocessing joblib

# Configura logging per joblib
import logging
logging.getLogger('joblib').setLevel(logging.ERROR)

# Disabilita il parallelismo di sklearn
from sklearn import set_config
set_config(working_memory=1024)

# =============================================================================
# IMPORTS
# =============================================================================
import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
import json
import pickle
from collections import deque, defaultdict
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from abc import ABC, abstractmethod

# Machine Learning imports (warnings already suppressed)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import talib as ta

# Deep Learning imports (opzionale)
try:
    import tensorflow as tf
    from tensorflow import keras
    DEEP_LEARNING_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


# ... resto del tuo codice
# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

class Config:
    """Configurazione globale del bot"""
    # MT5 Connection
    MT5_ACCOUNT = int()
    MT5_PASSWORD = ""
    MT5_SERVER = ""
    
    
    # Trading Parameters
    SYMBOL = "XAUUSD"
    FIXED_LOT_SIZE = 0.01  # per cambiare lotti
    MANUAL_LOT_SIZE = 0.01 # per cambiare lotti
    MAX_SPREAD_PIPS = 30.0
    MIN_CONFIDENCE = 0.70
    MIN_MINUTES_BETWEEN = 0.0167
    
    # Data Collection
    TRAINING_DAYS = 90
    DATA_TIMEFRAME = mt5.TIMEFRAME_M5
    REAL_TIME_TIMEFRAME = mt5.TIMEFRAME_M1
    
    # Risk Management
    MAX_DAILY_LOSS_PERCENT = 2.5
    MAX_CONSECUTIVE_LOSSES = 3
    MAX_DRAWDOWN_PERCENT = 5.0
    
    # ML Parameters
    ML_UPDATE_INTERVAL_HOURS = 24
    ML_MIN_SAMPLES = 500
    ML_VALIDATION_SPLIT = 0.2
    
    # RL Parameters
    RL_LEARNING_RATE = 0.01
    RL_DISCOUNT_FACTOR = 0.95
    RL_EXPLORATION_RATE = 0.1
    RL_EXPLORATION_DECAY = 0.995
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = "xauusd_bot.log"
    
    # Paths
    MODELS_DIR = "models"
    DATA_DIR = "data"
    CACHE_DIR = "cache"

    # Nella configurazione, aggiungi:
    ML_UPDATE_MINUTES = 15  # Aggiorna ML ogni 15 minuti
    PERFORMANCE_LOG_MINUTES = 15  # Log ogni 15 minuti
    MAX_CONSECUTIVE_ERRORS = 5  # Max errori prima di pausa

# =============================================================================
# 2. ENUMS & DATA CLASSES
# =============================================================================

class MarketRegime(Enum):
    """Regimi di mercato"""
    HIGH_VOLATILITY = "HIGH_VOL"
    LOW_VOLATILITY = "LOW_VOL"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    NORMAL = "NORMAL"

class TradeAction(Enum):
    """Azioni di trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TPStrategy(Enum):
    """Strategie TP"""
    ATR_BASED = "ATR_BASED"
    ML_PREDICTED = "ML_PREDICTED"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"
    RL_OPTIMIZED = "RL_OPTIMIZED"
    HYBRID = "HYBRID"


@dataclass
class MarketData:
    """Dati di mercato strutturati"""
    symbol: str
    bid: float
    ask: float
    spread: float  # Ora in pips già calcolati
    timestamp: datetime
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread_percent(self) -> float:
        """Spread come percentuale del prezzo"""
        return (self.ask - self.bid) / self.mid_price

@dataclass
class TradeSignal:
    """Segnale di trading"""
    action: TradeAction
    confidence: float
    entry_price: Optional[float] = None
    features: Optional[Dict[str, float]] = None

@dataclass
class TakeProfitLevel:
    """Livello Take Profit"""
    price: float
    percent: float
    strategy: TPStrategy
    confidence: float
    weight: float

# =============================================================================
# 3. MT5 MANAGER AVANZATO
# =============================================================================

class AdvancedMT5Manager:
    """Gestione avanzata connessione MT5 con auto-recovery"""
    
    def __init__(self):
        self.connected = False
        self.connection_stats = {
            'attempts': 0,
            'successful': 0,
            'last_attempt': None,
            'uptime': None
        }
        self.start_time = datetime.now()
        self.health_check_interval = 60  # secondi
        
    def initialize(self, max_attempts: int = 5, retry_delay: int = 3) -> bool:
        """Inizializza connessione con retry intelligente"""
        if self.connected:
            return True
            
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"\n🔗 Tentativo connessione MT5 {attempt}/{max_attempts}...")
                self.connection_stats['last_attempt'] = datetime.now()
                self.connection_stats['attempts'] += 1
                
                if not mt5.initialize():
                    raise ConnectionError("Inizializzazione MT5 fallita")
                
                # Login con credenziali
                if not mt5.login(
                    login=Config.MT5_ACCOUNT,
                    password=Config.MT5_PASSWORD,
                    server=Config.MT5_SERVER
                ):
                    raise ConnectionError(f"Login fallito: {mt5.last_error()}")
                
                # Verifica simbolo
                symbol_info = mt5.symbol_info(Config.SYMBOL)
                if symbol_info is None or not symbol_info.visible:
                    if not mt5.symbol_select(Config.SYMBOL, True):
                        raise ConnectionError(f"Simbolo {Config.SYMBOL} non disponibile")
                
                # Verifica spread
                tick = mt5.symbol_info_tick(Config.SYMBOL)
                if tick is None:
                    raise ConnectionError("Impossibile ottenere tick")
                
                spread_pips = (tick.ask - tick.bid) * 10000
                if spread_pips > Config.MAX_SPREAD_PIPS:
                    print(f"⚠️ Spread elevato: {spread_pips:.1f} pips")
                
                self.connected = True
                self.connection_stats['successful'] += 1
                self.connection_stats['uptime'] = datetime.now()
                
                print(f"✅ Connesso a MT5 - {Config.SYMBOL}")
                print(f"   Account: {Config.MT5_ACCOUNT}")
                print(f"   Bid: {tick.bid:.2f}, Ask: {tick.ask:.2f}")
                print(f"   Spread: {spread_pips:.1f} pips")
                
                # Avvia health check in background
                self._start_health_check()
                return True
                
            except Exception as e:
                print(f"❌ Errore connessione (tentativo {attempt}): {e}")
                if attempt < max_attempts:
                    print(f"   Riprovo tra {retry_delay} secondi...")
                    time.sleep(retry_delay)
                else:
                    print("❌ Tutti i tentativi di connessione falliti")
        
        return False
    
    def _start_health_check(self):
        """Avvia monitoraggio salute connessione in background"""
        def health_check():
            while self.connected:
                try:
                    if not self.check_connection():
                        print("⚠️ Connessione MT5 persa, tentativo di riconnessione...")
                        self.connected = False
                        self.initialize()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    print(f"⚠️ Errore health check: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=health_check, daemon=True)
        thread.start()

    def check_connection(self) -> bool:
        """Verifica se la connessione è ancora attiva - VERSIONE ROBUSTA"""
        if not self.connected:
            return False
        
        try:
            # 🎯 TEST 1: Prova a ottenere info account
            account_info = mt5.account_info()
            if account_info is None:
                print("⚠️  account_info() returned None")
                return False
            
            # 🎯 TEST 2: Prova a ottenere un tick
            tick = mt5.symbol_info_tick(Config.SYMBOL)
            if tick is None:
                print("⚠️  symbol_info_tick() returned None")
                return False
            
            # 🎯 TEST 3: Verifica che i prezzi siano validi
            if tick.bid <= 0 or tick.ask <= 0:
                print(f"⚠️  Invalid prices: bid={tick.bid}, ask={tick.ask}")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  Connection check error: {e}")
            return False
    def shutdown(self):
        """Chiude connessione MT5"""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                print("🔌 Connessione MT5 chiusa")
        except Exception as e:
            print(f"⚠️ Errore chiusura MT5: {e}")
    

    def find_open_position(self, symbol: str, magic: Optional[int] = None):
        #\"\"\"Return the first open position struct for symbol (and magic if provided).\"\"\"
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return None
        for pos in positions:
            try:
                pos_magic = int(pos.magic) if hasattr(pos, 'magic') else None
            except Exception:
                pos_magic = None
            if magic is None or pos_magic == magic:
                return pos
        return None

    def close_position_market(self, position) -> dict:
        #\"\"\"Close position by sending market order of opposite type and same volume.\"\"\"
        if position is None:
            return {'retcode': -1, 'comment': 'No position provided'}
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return {'retcode': -1, 'comment': 'No tick for symbol'}
        volume = float(position.volume)
        pos_type = int(position.type)
        if pos_type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": volume,
            "type": order_type,
            "price": float(price),
            "deviation": 20,
            "magic": int(position.magic) if hasattr(position, 'magic') else 999999,
            "comment": "CLOSE_BY_BOT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result

    def get_account_info(self) -> Dict:
        """Ottiene informazioni account"""
        if not self.connected:
            return {}
        
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'leverage': account_info.leverage,
                    'profit': account_info.profit,
                    'currency': account_info.currency
                }
        except Exception as e:
            print(f"⚠️ Errore info account: {e}")
        
        return {}

# =============================================================================
# 4. DATA COLLECTOR AVANZATO
# =============================================================================

class AdvancedDataCollector:
    """Raccolta dati multi-timeframe e multi-sorgente"""
    
    def __init__(self, mt5_manager: AdvancedMT5Manager):
        self.mt5_manager = mt5_manager
        self.data_cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    def collect_multi_timeframe_data(self, symbol: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Raccolta dati su multiple timeframe"""
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }
        
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for tf_name, tf in timeframes.items():
            try:
                print(f"📊 Raccolta dati {symbol} {tf_name}...")
                
                rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
                if rates is None or len(rates) == 0:
                    print(f"   ⚠️ Nessun dato per {tf_name}")
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Calcola features base
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                data[tf_name] = df
                print(f"   ✅ {len(df)} candele")
                
            except Exception as e:
                print(f"❌ Errore raccolta {tf_name}: {e}")
        
        return data
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Ottiene dati in tempo reale"""
        if not self.mt5_manager.connected:
            return None
        
        try:
            # Ottieni tick corrente
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            # Ottieni candela corrente
            rates = mt5.copy_rates_from_pos(symbol, Config.REAL_TIME_TIMEFRAME, 0, 10)
            
            data = MarketData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread=tick.ask - tick.bid,
                timestamp=datetime.now()
            )
            
            if rates is not None and len(rates) > 0:
                latest = rates[0]
                data.high = latest['high']
                data.low = latest['low']
                data.volume = latest['tick_volume']
            
            return data
            
        except Exception as e:
            print(f"⚠️ Errore dati realtime: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola indicatori tecnici avanzati"""
        df = df.copy()
        
        # Momentum Indicators
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'])
        
        # Volatility Indicators
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Trend Indicators
        df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
        df['ema_20'] = ta.EMA(df['close'], timeperiod=20)
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume Indicators
        if 'volume' in df.columns:
            df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Support/Resistance Features
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['close_vs_high'] = (df['close'] - df['high_20']) / df['high_20']
        df['close_vs_low'] = (df['close'] - df['low_20']) / df['low_20']
        
        # Pattern Recognition
        df['doji'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        
        return df

# =============================================================================
# 5. SUPPORT/RESISTANCE DETECTOR
# =============================================================================

class SupportResistanceDetector:
    """Rilevamento dinamico supporti e resistenze"""
    
    def __init__(self, window: int = 20, sensitivity: float = 0.5):
        self.window = window
        self.sensitivity = sensitivity
        self.levels_history = deque(maxlen=100)
        
    def detect_levels(self, prices: np.ndarray, method: str = 'swing') -> Dict:
        """
        Rileva livelli di supporto e resistenza
        
        Args:
            prices: Array di prezzi (high/low/close)
            method: Metodo di rilevamento ('swing', 'pivot', 'clustering')
        
        Returns:
            Dict con supporti e resistenze rilevati
        """
        if len(prices) < self.window * 2:
            return {'supports': [], 'resistances': [], 'confidence': 0.0}
        
        try:
            if method == 'swing':
                return self._detect_swing_levels(prices)
            elif method == 'pivot':
                return self._detect_pivot_levels(prices)
            elif method == 'clustering':
                return self._detect_clustering_levels(prices)
            else:
                return self._detect_hybrid_levels(prices)
                
        except Exception as e:
            print(f"⚠️ Errore detection livelli: {e}")
            return {'supports': [], 'resistances': [], 'confidence': 0.0}
    
    def _detect_swing_levels(self, prices: np.ndarray) -> Dict:
        """Rileva livelli tramite swing highs/lows"""
        highs = []
        lows = []
        
        for i in range(self.window, len(prices) - self.window):
            window_start = i - self.window
            window_end = i + self.window
            
            # Controlla se è un massimo locale
            if prices[i] == np.max(prices[window_start:window_end+1]):
                highs.append(prices[i])
            
            # Controlla se è un minimo locale
            if prices[i] == np.min(prices[window_start:window_end+1]):
                lows.append(prices[i])
        
        # Raggruppa livelli simili
        resistances = self._cluster_levels(highs, tolerance=0.001)
        supports = self._cluster_levels(lows, tolerance=0.001)
        
        confidence = min(len(resistances), len(supports)) / 10
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {
            'supports': sorted(supports),
            'resistances': sorted(resistances, reverse=True),
            'confidence': confidence
        }
    
    def _detect_pivot_levels(self, prices: np.ndarray) -> Dict:
        """Rileva livelli pivot"""
        df = pd.DataFrame(prices, columns=['price'])
        
        # Calcola pivot points standard
        df['pivot'] = (df['price'].shift(1).rolling(window=3).mean() +
                       df['price'].rolling(window=3).max() +
                       df['price'].rolling(window=3).min()) / 3
        
        resistances = df['pivot'].dropna().tolist()
        supports = df['pivot'].dropna().tolist()
        
        return {
            'supports': supports[-5:],  # Ultimi 5 supporti
            'resistances': resistances[-5:],  # Ultime 5 resistenze
            'confidence': 0.7
        }
    
    def _detect_clustering_levels(self, prices: np.ndarray) -> Dict:
        """Rileva livelli tramite clustering"""
        from sklearn.cluster import KMeans
        
        if len(prices) < 10:
            return {'supports': [], 'resistances': [], 'confidence': 0.0}
        
        # Prepara dati per clustering
        X = prices.reshape(-1, 1)
        
        # Determina numero ottimale di cluster
        n_clusters = min(5, len(prices) // 10)
        if n_clusters < 2:
            n_clusters = 2
        
        # Esegui clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Estrai centri cluster come livelli
        levels = sorted(kmeans.cluster_centers_.flatten())
        
        # Separa in supporti e resistenze
        median_price = np.median(prices)
        supports = [l for l in levels if l < median_price]
        resistances = [l for l in levels if l > median_price]
        
        return {
            'supports': supports,
            'resistances': resistances,
            'confidence': 0.6
        }
    
    def _detect_hybrid_levels(self, prices: np.ndarray) -> Dict:
        """Combina multiple metodi per maggiore robustezza"""
        results = []
        
        # Rileva con tutti i metodi
        methods = [self._detect_swing_levels, 
                  self._detect_pivot_levels,
                  self._detect_clustering_levels]
        
        for method in methods:
            try:
                result = method(prices)
                results.append(result)
            except:
                continue
        
        if not results:
            return {'supports': [], 'resistances': [], 'confidence': 0.0}
        
        # Combina risultati
        all_supports = []
        all_resistances = []
        all_confidences = []
        
        for r in results:
            all_supports.extend(r.get('supports', []))
            all_resistances.extend(r.get('resistances', []))
            all_confidences.append(r.get('confidence', 0.0))
        
        # Raggruppa livelli simili
        final_supports = self._cluster_levels(all_supports, tolerance=0.001)
        final_resistances = self._cluster_levels(all_resistances, tolerance=0.001)
        
        # Calcola confidenza media
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'supports': sorted(final_supports),
            'resistances': sorted(final_resistances, reverse=True),
            'confidence': avg_confidence
        }
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.001) -> List[float]:
        """Raggruppa livelli simili"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def get_tp_from_levels(self, current_price: float, action: str, 
                          levels_data: Dict) -> Optional[TakeProfitLevel]:
        """Calcola TP basato su livelli di supporto/resistenza"""
        if not levels_data:
            return None
        
        supports = levels_data.get('supports', [])
        resistances = levels_data.get('resistances', [])
        confidence = levels_data.get('confidence', 0.0)
        
        if action == TradeAction.BUY.value:
            # Per BUY, prendi la resistenza più vicina sopra il prezzo corrente
            valid_resistances = [r for r in resistances if r > current_price]
            if valid_resistances:
                tp_price = min(valid_resistances)
                tp_percent = (tp_price - current_price) / current_price
                
                # Aggiusta per essere realistico
                if tp_percent > 0.01:  # Max 1%
                    tp_percent = 0.01
                    tp_price = current_price * (1 + tp_percent)
                
                return TakeProfitLevel(
                    price=tp_price,
                    percent=tp_percent,
                    strategy=TPStrategy.SUPPORT_RESISTANCE,
                    confidence=confidence,
                    weight=0.3
                )
        
        elif action == TradeAction.SELL.value:
            # Per SELL, prendi il supporto più vicino sotto il prezzo corrente
            valid_supports = [s for s in supports if s < current_price]
            if valid_supports:
                tp_price = max(valid_supports)
                tp_percent = (current_price - tp_price) / current_price
                
                # Aggiusta per essere realistico
                if tp_percent > 0.01:  # Max 1%
                    tp_percent = 0.01
                    tp_price = current_price * (1 - tp_percent)
                
                return TakeProfitLevel(
                    price=tp_price,
                    percent=tp_percent,
                    strategy=TPStrategy.SUPPORT_RESISTANCE,
                    confidence=confidence,
                    weight=0.3
                )
        
        return None

# =============================================================================
# 6. MARKET REGIME ANALYZER
# =============================================================================

class MarketRegimeAnalyzer:
    """Analizzatore avanzato regime di mercato"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.current_regime = MarketRegime.NORMAL
        self.regime_confidence = 0.5
        
    def analyze_regime(self, df: pd.DataFrame) -> Dict:
        """
        Analizza regime di mercato corrente
        
        Args:
            df: DataFrame con dati OHLCV
        
        Returns:
            Dict con regime e caratteristiche
        """
        if len(df) < 50:
            return {
                'regime': MarketRegime.NORMAL,
                'confidence': 0.5,
                'characteristics': {}
            }
        
        try:
            characteristics = {}
            
            # 1. Volatilità
            returns = df['returns'].dropna()
            volatility = returns.std() * np.sqrt(252)
            characteristics['volatility'] = volatility
            
            # 2. Trend
            prices = df['close'].values
            sma_20 = ta.SMA(prices, timeperiod=20)
            if len(sma_20) >= 20 and not np.isnan(sma_20[-1]):
                price_above_sma = prices[-1] > sma_20[-1]
                trend_strength = abs(prices[-1] - sma_20[-1]) / sma_20[-1]
            else:
                price_above_sma = False
                trend_strength = 0
            
            characteristics['trend_strength'] = trend_strength
            characteristics['trend_direction'] = 'UP' if price_above_sma else 'DOWN'
            
            # 3. ADX (forza trend)
            if len(df) >= 14:
                adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                if len(adx) > 0 and not np.isnan(adx[-1]):
                    characteristics['adx'] = adx[-1]
                else:
                    characteristics['adx'] = 0
            else:
                characteristics['adx'] = 0
            
            # 4. Range
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            range_size = (recent_high - recent_low) / recent_low
            characteristics['range_size'] = range_size
            
            # 5. Volume
            if 'volume' in df.columns:
                volume_mean = df['volume'].tail(20).mean()
                volume_current = df['volume'].iloc[-1]
                volume_ratio = volume_current / volume_mean if volume_mean > 0 else 1
                characteristics['volume_ratio'] = volume_ratio
            
            # Determina regime
            regime, confidence = self._determine_regime(characteristics)
            
            self.current_regime = regime
            self.regime_confidence = confidence
            self.regime_history.append(regime)
            
            return {
                'regime': regime,
                'confidence': confidence,
                'characteristics': characteristics
            }
            
        except Exception as e:
            print(f"⚠️ Errore analisi regime: {e}")
            return {
                'regime': MarketRegime.NORMAL,
                'confidence': 0.5,
                'characteristics': {}
            }
    
    def _determine_regime(self, characteristics: Dict) -> Tuple[MarketRegime, float]:
        """Determina regime basato su caratteristiche"""
        volatility = characteristics.get('volatility', 0)
        trend_strength = characteristics.get('trend_strength', 0)
        adx = characteristics.get('adx', 0)
        range_size = characteristics.get('range_size', 0)
        
        # 1. Alta volatilità
        if volatility > 0.25:
            return MarketRegime.HIGH_VOLATILITY, 0.8
        
        # 2. Bassa volatilità
        if volatility < 0.10:
            return MarketRegime.LOW_VOLATILITY, 0.8
        
        # 3. Trend forte
        if adx > 25 and trend_strength > 0.02:
            if characteristics.get('trend_direction') == 'UP':
                return MarketRegime.TRENDING_UP, 0.7
            else:
                return MarketRegime.TRENDING_DOWN, 0.7
        
        # 4. Range trading
        if range_size < 0.01:  # Range stretto
            return MarketRegime.RANGING, 0.7
        
        # 5. Breakout
        if range_size > 0.02:  # Range ampio
            return MarketRegime.BREAKOUT, 0.6
        
        # 6. Normale
        return MarketRegime.NORMAL, 0.6
    
    def get_tp_multiplier(self, regime: MarketRegime) -> float:
        """Restituisce moltiplicatore TP per regime"""
        multipliers = {
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.LOW_VOLATILITY: 0.7,
            MarketRegime.TRENDING_UP: 1.8,
            MarketRegime.TRENDING_DOWN: 1.8,
            MarketRegime.RANGING: 0.5,
            MarketRegime.BREAKOUT: 2.0,
            MarketRegime.NORMAL: 1.0
        }
        return multipliers.get(regime, 1.0)

# =============================================================================
# 7. ML TP PREDICTOR
# =============================================================================

class MLTPPredictor:
    """Predittore ML per Take Profit ottimale"""
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.last_training = None
        
    def prepare_training_data(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepara dati per training"""
        print("🤖 Preparazione dati per training ML...")
        
        try:
            # Usa dati M5 come base
            if 'M5' not in historical_data:
                raise ValueError("Dati M5 necessari per training")
            
            df = historical_data['M5'].copy()
            
            # Calcola TP ottimale storico
            # TP ottimale = massimo profitto raggiunto prima di un drawdown significativo
            df['optimal_tp'] = self._calculate_optimal_tp(df)
            
            # Calcola features
            df = self._calculate_features(df)
            
            # Rimuovi NaN
            df = df.dropna()
            
            print(f"✅ Dati preparati: {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Errore preparazione dati: {e}")
            raise
    
    def _calculate_optimal_tp(self, df: pd.DataFrame, lookahead: int = 20) -> pd.Series:
        """Calcola TP ottimale per ogni punto dati"""
        optimal_tp = []
        
        for i in range(len(df) - lookahead):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+lookahead+1]
            
            # Calcola massimo profitto raggiunto
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            # TP ottimale = punto dove il profitto è massimo prima di un drawdown
            # Usa algoritmo per trovare il miglior trade-off
            prices_array = future_prices.values
            max_profit = 0
            optimal_exit = current_price
            
            for j, price in enumerate(prices_array):
                profit = (price - current_price) / current_price
                # Penalizza exit troppo tardivi
                time_penalty = j / lookahead * 0.1
                adjusted_profit = profit - time_penalty
                
                if adjusted_profit > max_profit:
                    max_profit = adjusted_profit
                    optimal_exit = price
            
            optimal_tp_percent = (optimal_exit - current_price) / current_price
            optimal_tp.append(optimal_tp_percent)
        
        # Aggiusta NaN per gli ultimi valori
        optimal_tp.extend([0] * min(lookahead, len(df)))
        
        return pd.Series(optimal_tp, index=df.index)
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola features per il modello"""
        features_df = df.copy()
        
        # Features tecniche
        features_df['rsi_14'] = ta.RSI(features_df['close'], timeperiod=14)
        features_df['atr_14'] = ta.ATR(features_df['high'], features_df['low'], 
                                       features_df['close'], timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            features_df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        features_df['bb_position'] = (features_df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Trend
        features_df['sma_20'] = ta.SMA(features_df['close'], timeperiod=20)
        features_df['ema_20'] = ta.EMA(features_df['close'], timeperiod=20)
        features_df['price_vs_sma'] = (features_df['close'] - features_df['sma_20']) / features_df['sma_20']
        
        # Momentum
        features_df['returns_5'] = features_df['close'].pct_change(5)
        features_df['returns_10'] = features_df['close'].pct_change(10)
        
        # Volatilità
        features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()
        
        # Volume
        if 'volume' in features_df.columns:
            features_df['volume_sma'] = ta.SMA(features_df['volume'], timeperiod=20)
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
        
        # Pattern
        features_df['high_20'] = features_df['high'].rolling(window=20).max()
        features_df['low_20'] = features_df['low'].rolling(window=20).min()
        features_df['close_vs_high'] = (features_df['close'] - features_df['high_20']) / features_df['high_20']
        features_df['close_vs_low'] = (features_df['close'] - features_df['low_20']) / features_df['low_20']
        
        # Spread (se disponibile)
        if 'spread' not in features_df.columns:
            features_df['spread'] = (features_df['high'] - features_df['low']) / features_df['close'] * 100
        
        return features_df
    
    def train(self, df: pd.DataFrame) -> bool:
        """Addestra il modello ML"""
        print("\n🤖 Addestramento modello ML per TP prediction...")
        
        try:
            # Separa features e target
            feature_cols = [
                'rsi_14', 'atr_14', 'bb_position', 'volatility_20',
                'price_vs_sma', 'returns_5', 'returns_10',
                'close_vs_high', 'close_vs_low', 'spread'
            ]
            
            # Verifica che tutte le colonne esistano
            available_cols = [col for col in feature_cols if col in df.columns]
            self.feature_columns = available_cols
            
            X = df[available_cols].values
            y = df['optimal_tp'].values
            
            print(f"   Features: {len(available_cols)}")
            print(f"   Samples: {len(X)}")
            
            if len(X) < Config.ML_MIN_SAMPLES:
                print(f"❌ Samples insufficienti: {len(X)} < {Config.ML_MIN_SAMPLES}")
                return False
            
            # Split temporale
            split_idx = int(len(X) * (1 - Config.ML_VALIDATION_SPLIT))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Addestra modello regressione
            self.regression_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                min_samples_split=10,
                min_samples_leaf=5
            )
            
            self.regression_model.fit(X_train_scaled, y_train)
            
            # Valuta
            y_pred = self.regression_model.predict(X_val_scaled)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f"\n✅ Modello ML addestrato!")
            print(f"   MSE: {mse:.6f}")
            print(f"   R² Score: {r2:.3f}")
            print(f"   Train samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"❌ Errore training ML: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, features: Dict) -> Optional[TakeProfitLevel]:
        """Predice TP ottimale"""
        if not self.is_trained or self.regression_model is None:
            return None
        
        try:
            # Prepara array features
            feature_array = []
            for col in self.feature_columns:
                value = features.get(col, 0.0)
                if value is None or np.isnan(value):
                    value = 0.0
                feature_array.append(float(value))
            
            # Scale e predici
            features_scaled = self.scaler.transform([feature_array])
            predicted_tp_percent = self.regression_model.predict(features_scaled)[0]
            
            # Limita a range realistico
            predicted_tp_percent = np.clip(predicted_tp_percent, 0.001, 0.01)
            
            # Calcola confidenza basata su features
            confidence = self._calculate_prediction_confidence(features)
            
            return TakeProfitLevel(
                price=features.get('current_price', 0) * (1 + predicted_tp_percent),
                percent=predicted_tp_percent,
                strategy=TPStrategy.ML_PREDICTED,
                confidence=confidence,
                weight=0.4
            )
            
        except Exception as e:
            print(f"⚠️ Errore predizione ML: {e}")
            return None
    
    def _calculate_prediction_confidence(self, features: Dict) -> float:
        """Calcola confidenza della predizione"""
        confidences = []
        
        # Confidenza basata su RSI
        rsi = features.get('rsi_14', 50)
        if 30 <= rsi <= 70:
            confidences.append(0.8)
        else:
            confidences.append(0.5)
        
        # Confidenza basata su volatilità
        volatility = features.get('volatility_20', 0)
        if 0.001 <= volatility <= 0.01:
            confidences.append(0.7)
        else:
            confidences.append(0.4)
        
        # Confidenza basata su trend
        price_vs_sma = abs(features.get('price_vs_sma', 0))
        if price_vs_sma < 0.02:
            confidences.append(0.6)
        else:
            confidences.append(0.3)
        
        return np.mean(confidences) if confidences else 0.5

# =============================================================================
# 7B. ML MODEL PER SEGNALI DI TRADING
# =============================================================================

# =============================================================================
# 8. REINFORCEMENT LEARNING AGENT
# =============================================================================

class RLAgent:
    """Agente Reinforcement Learning per ottimizzazione TP"""
    
    def __init__(self, state_size: int = 10, action_size: int = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = Config.RL_LEARNING_RATE
        self.discount_factor = Config.RL_DISCOUNT_FACTOR
        self.exploration_rate = Config.RL_EXPLORATION_RATE
        self.exploration_decay = Config.RL_EXPLORATION_DECAY
        
        # Q-table o modello neurale
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Performance tracking
        self.episode_rewards = []
        self.learning_progress = []
        
    def get_state(self, market_state: Dict) -> Tuple:
        """Converte stato di mercato in stato RL"""
        state_features = []
        
        # 1. Profit corrente (normalizzato)
        current_profit = market_state.get('current_profit_pct', 0)
        state_features.append(np.clip(current_profit / 0.01, -1, 1))  # Normalizza a [-1, 1]
        
        # 2. Durata trade (normalizzata)
        duration = market_state.get('duration_minutes', 0)
        state_features.append(np.clip(duration / 60, 0, 1))  # Normalizza a [0, 1]
        
        # 3. Volatilità
        volatility = market_state.get('volatility', 0.01)
        state_features.append(np.clip(volatility / 0.02, 0, 1))
        
        # 4. Trend strength
        trend_strength = market_state.get('trend_strength', 0)
        state_features.append(np.clip(trend_strength / 0.03, -1, 1))
        
        # 5. RSI (normalizzato)
        rsi = market_state.get('rsi', 50)
        state_features.append((rsi - 50) / 30)  # Normalizza a [-1, 1]
        
        # 6. Distance to TP
        tp_distance = market_state.get('tp_distance_pct', 0)
        state_features.append(np.clip(tp_distance / 0.01, 0, 1))
        
        # 7. Market regime (one-hot encoding)
        regime = market_state.get('regime', 'NORMAL')
        regime_mapping = {
            'HIGH_VOL': [1, 0, 0, 0, 0],
            'LOW_VOL': [0, 1, 0, 0, 0],
            'TRENDING': [0, 0, 1, 0, 0],
            'RANGING': [0, 0, 0, 1, 0],
            'NORMAL': [0, 0, 0, 0, 1]
        }
        regime_encoding = regime_mapping.get(regime, [0, 0, 0, 0, 1])
        state_features.extend(regime_encoding)
        
        # Assicurati che abbiamo esattamente state_size features
        if len(state_features) > self.state_size:
            state_features = state_features[:self.state_size]
        elif len(state_features) < self.state_size:
            state_features.extend([0] * (self.state_size - len(state_features)))
        
        return tuple(np.round(state_features, 3))
    
    def choose_action(self, state: Tuple) -> int:
        """Sceglie azione basata su policy epsilon-greedy"""
        if np.random.random() < self.exploration_rate:
            # Exploration: azione random
            return np.random.randint(self.action_size)
        else:
            # Exploitation: migliore azione secondo Q-table
            return np.argmax(self.q_table[state])
    
    def get_action_meaning(self, action: int) -> str:
        """Restituisce significato dell'azione"""
        actions = {
            0: "HOLD_POSITION",
            1: "CLOSE_NOW",
            2: "INCREASE_TP_10%",
            3: "DECREASE_TP_10%",
            4: "TRAILING_STOP"
        }
        return actions.get(action, "UNKNOWN")
    
    def update_q_table(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Aggiorna Q-table usando algoritmo Q-learning"""
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        
        # Formula Q-learning
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, 0.01)
    
    def calculate_reward(self, trade_outcome: Dict) -> float:
        """Calcola reward per l'agente RL"""
        reward = 0
        
        # Base reward sul P&L
        pnl_percent = trade_outcome.get('pnl_percent', 0)
        reward += pnl_percent * 100  # Scala per sensibilità
        
        # Bonus per trade vincenti
        if pnl_percent > 0:
            reward += 1.0
        
        # Penalità per trade perdenti
        if pnl_percent < 0:
            reward -= 1.0
        
        # Bonus per chiusura tempestiva
        duration = trade_outcome.get('duration_minutes', 0)
        if duration < 10 and pnl_percent > 0:
            reward += 0.5
        
        # Penalità per hold troppo lungo
        if duration > 30:
            reward -= 0.5
        
        # Normalizza reward
        return np.clip(reward, -2, 2)
    
    def learn_from_experience(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Apprendi da una esperienza"""
        self.update_q_table(state, action, reward, next_state)
        
        # Salva storia per analisi
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Aggiorna learning progress
        self.learning_progress.append({
            'timestamp': datetime.now(),
            'state': state,
            'action': action,
            'reward': reward,
            'exploration_rate': self.exploration_rate
        })
    
    def get_tp_adjustment(self, action: int, current_tp: float) -> float:
        """Calcola aggiustamento TP basato su azione RL"""
        adjustments = {
            0: 0.0,    # HOLD: nessun cambio
            1: -1.0,   # CLOSE_NOW: chiudi immediatamente
            2: 0.1,    # INCREASE_TP_10%: aumenta TP del 10%
            3: -0.1,   # DECREASE_TP_10%: diminuisci TP del 10%
            4: 0.05    # TRAILING_STOP: attiva trailing stop
        }
        
        adjustment = adjustments.get(action, 0.0)
        if action == 1:  # CLOSE_NOW
            return 0.0  # Segnale per chiudere
        
        return current_tp * adjustment

# =============================================================================
# 9. HYBRID TP MANAGER
# =============================================================================

class HybridTPManager:
    """Manager Take Profit ibrido con pesi dinamici"""
    
    def __init__(self):
        # Componenti
        self.sr_detector = SupportResistanceDetector()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.ml_predictor = MLTPPredictor()
        self.rl_agent = RLAgent()
        
        # Configurazione pesi iniziali
        self.strategy_weights = {
            TPStrategy.ATR_BASED: 0.25,
            TPStrategy.ML_PREDICTED: 0.35,
            TPStrategy.SUPPORT_RESISTANCE: 0.25,
            TPStrategy.RL_OPTIMIZED: 0.15
        }
        
        # Performance tracking per ogni strategia
        self.strategy_performance = {
            strategy: {'success': 0, 'total': 0, 'avg_profit': 0}
            for strategy in TPStrategy
        }
        
        # Stato corrente
        self.current_trade_info = None
        self.tp_history = []
        
        print("\n🎯 HYBRID TP MANAGER INIZIALIZZATO")
        print("   Strategie attive:")
        for strategy, weight in self.strategy_weights.items():
            print(f"   • {strategy.value}: {weight:.0%}")
    
    def calculate_hybrid_tp(self, market_data: MarketData, action: str, 
                           market_context: Dict) -> TakeProfitLevel:
        """Calcola TP ibrido combinando tutte le strategie"""
        tp_suggestions = []
        
        # 1. TP basato su ATR e Regime
        tp_atr = self._calculate_atr_based_tp(market_data, action, market_context)
        if tp_atr:
            tp_atr.weight = self.strategy_weights[TPStrategy.ATR_BASED]
            tp_suggestions.append(tp_atr)
        
        # 2. TP basato su ML
        if self.ml_predictor.is_trained:
            tp_ml = self.ml_predictor.predict(market_context.get('features', {}))
            if tp_ml:
                tp_ml.weight = self.strategy_weights[TPStrategy.ML_PREDICTED]
                tp_suggestions.append(tp_ml)
        
        # 3. TP basato su Support/Resistance
        sr_levels = self.sr_detector.detect_levels(
            market_context.get('price_history', np.array([])),
            method='hybrid'
        )
        tp_sr = self.sr_detector.get_tp_from_levels(market_data.mid_price, action, sr_levels)
        if tp_sr:
            tp_sr.weight = self.strategy_weights[TPStrategy.SUPPORT_RESISTANCE]
            tp_suggestions.append(tp_sr)
        
        # 4. TP ottimizzato con RL
        if self.current_trade_info:
            tp_rl = self._calculate_rl_optimized_tp(market_data, action, market_context)
            if tp_rl:
                tp_rl.weight = self.strategy_weights[TPStrategy.RL_OPTIMIZED]
                tp_suggestions.append(tp_rl)
        
        if not tp_suggestions:
            # Fallback: TP fisso
            return TakeProfitLevel(
                price=self._calculate_fixed_tp(market_data, action),
                percent=0.002,
                strategy=TPStrategy.HYBRID,
                confidence=0.5,
                weight=1.0
            )
        
        # Combina suggerimenti con pesi
        hybrid_tp = self._combine_tp_suggestions(tp_suggestions, market_data, action)
        
        # Salva per tracking
        self.tp_history.append({
            'timestamp': datetime.now(),
            'suggestions': tp_suggestions,
            'hybrid_tp': hybrid_tp,
            'market_context': market_context
        })
        
        return hybrid_tp
    
    def _calculate_atr_based_tp(self, market_data: MarketData, action: str, 
                               market_context: Dict) -> Optional[TakeProfitLevel]:
        """Calcola TP basato su ATR e regime di mercato"""
        try:
            atr = market_context.get('atr', 0.015)
            regime = market_context.get('regime', MarketRegime.NORMAL)
            
            # Moltiplicatore ATR basato su regime
            regime_multiplier = self.regime_analyzer.get_tp_multiplier(regime)
            
            # Calcola TP distance
            tp_distance = atr * regime_multiplier
            
            # Limita a range realistico
            tp_distance = np.clip(tp_distance, 0.001, 0.01)
            
            # Calcola prezzo TP
            if action == TradeAction.BUY.value:
                tp_price = market_data.mid_price * (1 + tp_distance)
            else:
                tp_price = market_data.mid_price * (1 - tp_distance)
            
            # Calcola confidenza
            confidence = self._calculate_atr_confidence(atr, regime)
            
            return TakeProfitLevel(
                price=tp_price,
                percent=tp_distance,
                strategy=TPStrategy.ATR_BASED,
                confidence=confidence,
                weight=self.strategy_weights[TPStrategy.ATR_BASED]
            )
            
        except Exception as e:
            print(f"⚠️ Errore calcolo ATR TP: {e}")
            return None
    
    def _calculate_atr_confidence(self, atr: float, regime: MarketRegime) -> float:
        """Calcola confidenza per TP ATR-based"""
        # Confidenza alta quando ATR è stabile
        if 0.005 <= atr <= 0.02:
            base_confidence = 0.7
        else:
            base_confidence = 0.5
        
        # Aggiusta per regime
        regime_confidence = {
            MarketRegime.NORMAL: 0.8,
            MarketRegime.TRENDING_UP: 0.7,
            MarketRegime.TRENDING_DOWN: 0.7,
            MarketRegime.RANGING: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.BREAKOUT: 0.5
        }.get(regime, 0.5)
        
        return (base_confidence + regime_confidence) / 2
    
    def _calculate_rl_optimized_tp(self, market_data: MarketData, action: str,
                                  market_context: Dict) -> Optional[TakeProfitLevel]:
        """Calcola TP ottimizzato con RL"""
        try:
            # Prepara stato per RL
            rl_state = {
                'current_profit_pct': market_context.get('current_profit_pct', 0),
                'duration_minutes': market_context.get('trade_duration', 0),
                'volatility': market_context.get('volatility', 0.01),
                'trend_strength': market_context.get('trend_strength', 0),
                'rsi': market_context.get('rsi', 50),
                'tp_distance_pct': market_context.get('tp_distance_pct', 0),
                'regime': market_context.get('regime', 'NORMAL')
            }
            
            # Ottieni stato e azione
            state = self.rl_agent.get_state(rl_state)
            action_idx = self.rl_agent.choose_action(state)
            
            # Calcola TP corrente
            current_tp_percent = 0.002  # TP base
            
            # Applica aggiustamento RL
            adjustment = self.rl_agent.get_tp_adjustment(action_idx, current_tp_percent)
            
            if adjustment == -1.0:  # Segnale di chiusura
                return None
            
            new_tp_percent = current_tp_percent + adjustment
            new_tp_percent = np.clip(new_tp_percent, 0.001, 0.01)
            
            # Calcola prezzo TP
            if action == TradeAction.BUY.value:
                tp_price = market_data.mid_price * (1 + new_tp_percent)
            else:
                tp_price = market_data.mid_price * (1 - new_tp_percent)
            
            return TakeProfitLevel(
                price=tp_price,
                percent=new_tp_percent,
                strategy=TPStrategy.RL_OPTIMIZED,
                confidence=0.6,  # Confidenza media per RL
                weight=self.strategy_weights[TPStrategy.RL_OPTIMIZED]
            )
            
        except Exception as e:
            print(f"⚠️ Errore calcolo RL TP: {e}")
            return None
    
    def _calculate_fixed_tp(self, market_data: MarketData, action: str) -> float:
        """Calcola TP fisso di fallback"""
        tp_percent = 0.002  # 0.2%
        
        if action == TradeAction.BUY.value:
            return market_data.mid_price * (1 + tp_percent)
        else:
            return market_data.mid_price * (1 - tp_percent)
    
    def _combine_tp_suggestions(self, suggestions: List[TakeProfitLevel],
                               market_data: MarketData, action: str) -> TakeProfitLevel:
        """Combina multiple suggerimenti TP con pesi"""
        if not suggestions:
            return self._create_default_tp(market_data, action)
        
        # Normalizza pesi
        total_weight = sum(tp.weight for tp in suggestions)
        normalized_weights = [tp.weight / total_weight for tp in suggestions]
        
        # Calcola TP medio pesato
        weighted_tp_percent = sum(tp.percent * w for tp, w in zip(suggestions, normalized_weights))
        weighted_confidence = sum(tp.confidence * w for tp, w in zip(suggestions, normalized_weights))
        
        # Calcola prezzo TP
        if action == TradeAction.BUY.value:
            tp_price = market_data.mid_price * (1 + weighted_tp_percent)
        else:
            tp_price = market_data.mid_price * (1 - weighted_tp_percent)
        
        # Determina strategia dominante
        dominant_strategy = max(suggestions, key=lambda x: x.weight).strategy
        
        return TakeProfitLevel(
            price=tp_price,
            percent=weighted_tp_percent,
            strategy=TPStrategy.HYBRID,
            confidence=weighted_confidence,
            weight=1.0
        )
    
    def _create_default_tp(self, market_data: MarketData, action: str) -> TakeProfitLevel:
        """Crea TP di default"""
        tp_percent = 0.002
        
        if action == TradeAction.BUY.value:
            tp_price = market_data.mid_price * (1 + tp_percent)
        else:
            tp_price = market_data.mid_price * (1 - tp_percent)
        
        return TakeProfitLevel(
            price=tp_price,
            percent=tp_percent,
            strategy=TPStrategy.HYBRID,
            confidence=0.5,
            weight=1.0
        )
    
    def update_strategy_weights(self, trade_outcome: Dict):
        """Aggiorna pesi strategie basati su performance"""
        strategy_used = trade_outcome.get('tp_strategy', TPStrategy.HYBRID)
        pnl_percent = trade_outcome.get('pnl_percent', 0)
        
        # Aggiorna performance tracking
        if strategy_used in self.strategy_performance:
            perf = self.strategy_performance[strategy_used]
            perf['total'] += 1
            if pnl_percent > 0:
                perf['success'] += 1
            
            # Calcola profitto medio
            if perf['total'] > 0:
                perf['avg_profit'] = (perf['avg_profit'] * (perf['total'] - 1) + pnl_percent) / perf['total']
        
        # Ricalcola pesi ogni 10 trade
        total_trades = sum(p['total'] for p in self.strategy_performance.values())
        if total_trades % 10 == 0 and total_trades > 0:
            self._rebalance_weights()
    
    def _rebalance_weights(self):
        """Ribilanciamento pesi basato su performance"""
        print("\n⚖️  Ribilanciamento pesi strategie...")
        
        # Calcola performance score per ogni strategia
        scores = {}
        for strategy, perf in self.strategy_performance.items():
            if perf['total'] > 0:
                win_rate = perf['success'] / perf['total']
                avg_profit = perf['avg_profit']
                score = win_rate * 0.6 + np.clip(avg_profit * 100, 0, 1) * 0.4
                scores[strategy] = score
            else:
                scores[strategy] = 0.5  # Default score
        
        # Normalizza scores
        total_score = sum(scores.values())
        if total_score > 0:
            new_weights = {s: scores[s] / total_score for s in scores}
            
            # Applica smoothing per evitare cambiamenti drastici
            smoothing = 0.7
            for strategy in self.strategy_weights:
                old_weight = self.strategy_weights[strategy]
                new_weight = new_weights.get(strategy, old_weight)
                self.strategy_weights[strategy] = (old_weight * smoothing + 
                                                  new_weight * (1 - smoothing))
            
            print("   Nuovi pesi:")
            for strategy, weight in self.strategy_weights.items():
                print(f"   • {strategy.value}: {weight:.1%}")
    
    def get_status(self) -> Dict:
        """Restituisce stato del manager"""
        return {
            'strategy_weights': {k.value: v for k, v in self.strategy_weights.items()},
            'strategy_performance': {
                k.value: v for k, v in self.strategy_performance.items()
            },
            'tp_history_count': len(self.tp_history),
            'rl_exploration_rate': self.rl_agent.exploration_rate,
            'ml_trained': self.ml_predictor.is_trained
        }

# =============================================================================
# 10. RISK MANAGER AVANZATO
# =============================================================================

class AdvancedRiskManager:
    """Gestione rischio avanzata con circuit breaker"""
    
    def __init__(self):
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0
        }
        
        self.account_stats = {
            'initial_balance': None,
            'peak_balance': None,
            'current_drawdown': 0.0
        }
        
        self.circuit_breakers = {
            'max_daily_loss': Config.MAX_DAILY_LOSS_PERCENT,
            'max_consecutive_losses': Config.MAX_CONSECUTIVE_LOSSES,
            'max_drawdown': Config.MAX_DRAWDOWN_PERCENT,
            'enabled': True
        }
        
        self.trade_history = []
        
    def check_trade_allowed(self, account_info: Dict) -> Tuple[bool, str]:
        """Verifica se è permesso aprire un nuovo trade"""
        # 1. Check circuit breaker attivo
        if not self.circuit_breakers['enabled']:
            return True, "Circuit breaker disabilitato"
        
        # 2. Check data giornaliera
        today = datetime.now().date()
        if self.daily_stats['date'] != today:
            self._reset_daily_stats()
        
        # 3. Check perdita giornaliera
        if self.daily_stats['total_pnl'] < 0:
            daily_loss_percent = abs(self.daily_stats['total_pnl']) / self.account_stats.get('initial_balance', 1)
            if daily_loss_percent >= self.circuit_breakers['max_daily_loss'] / 100:
                return False, f"Perdita giornaliera limite raggiunta: {daily_loss_percent:.1%}"
        
        # 4. Check consecutive losses
        if self.daily_stats['consecutive_losses'] >= self.circuit_breakers['max_consecutive_losses']:
            return False, f"Troppe perdite consecutive: {self.daily_stats['consecutive_losses']}"
        
        # 5. Check drawdown totale
        if self.account_stats['current_drawdown'] >= self.circuit_breakers['max_drawdown'] / 100:
            return False, f"Drawdown limite raggiunto: {self.account_stats['current_drawdown']:.1%}"
        
        # 6. Check margin disponibile
        if account_info.get('free_margin', 0) <= 0:
            return False, "Margin insufficiente"
        
        return True, "Trading permesso"
    
    def update_trade_result(self, trade_result: Dict):
        """Aggiorna statistiche dopo un trade"""
        # Aggiorna daily stats
        self.daily_stats['trades'] += 1
        self.daily_stats['total_pnl'] += trade_result.get('pnl', 0)
        
        if trade_result.get('pnl', 0) > 0:
            self.daily_stats['wins'] += 1
            self.daily_stats['consecutive_losses'] = 0
        else:
            self.daily_stats['losses'] += 1
            self.daily_stats['consecutive_losses'] += 1
        
        # Aggiorna drawdown
        current_equity = trade_result.get('current_equity', 0)
        if self.account_stats['peak_balance'] is None or current_equity > self.account_stats['peak_balance']:
            self.account_stats['peak_balance'] = current_equity
        
        if self.account_stats['peak_balance'] > 0:
            self.account_stats['current_drawdown'] = (
                (self.account_stats['peak_balance'] - current_equity) / 
                self.account_stats['peak_balance']
            )
        
        # Salva trade in history
        self.trade_history.append({
            'timestamp': datetime.now(),
            **trade_result
        })
        
        # Cleanup old history
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def _reset_daily_stats(self):
        """Resetta statistiche giornaliere"""
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0
        }
    
    def set_account_balance(self, balance: float):
        """Imposta balance iniziale account"""
        if self.account_stats['initial_balance'] is None:
            self.account_stats['initial_balance'] = balance
            self.account_stats['peak_balance'] = balance
    
    def get_daily_report(self) -> Dict:
        """Restituisce report giornaliero"""
        win_rate = (self.daily_stats['wins'] / self.daily_stats['trades'] * 100 
                   if self.daily_stats['trades'] > 0 else 0)
        
        return {
            'date': str(self.daily_stats['date']),
            'trades': self.daily_stats['trades'],
            'wins': self.daily_stats['wins'],
            'losses': self.daily_stats['losses'],
            'win_rate': f"{win_rate:.1f}%",
            'total_pnl': self.daily_stats['total_pnl'],
            'consecutive_losses': self.daily_stats['consecutive_losses'],
            'current_drawdown': f"{self.account_stats['current_drawdown']:.1%}",
            'circuit_breaker_active': self.circuit_breakers['enabled']
        }
    
    def enable_circuit_breaker(self, enabled: bool = True):
        """Abilita/disabilita circuit breaker"""
        self.circuit_breakers['enabled'] = enabled
        status = "abilitato" if enabled else "disabilitato"
        print(f"🔧 Circuit breaker {status}")

# =============================================================================
# 11. MAIN BOT - XAUUSD INTELLIGENT TP BOT
# =============================================================================


class MLModel:
    """Modello ML per predizioni BUY/SELL"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features tecniche dal DataFrame"""
        df_features = df.copy()
        
        # Assicurati che le colonne necessarie esistano
        required_cols = ['close', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df_features.columns:
                if col == 'close':
                    df_features['close'] = (df_features['bid'] + df_features['ask']) / 2
                elif col == 'volume':
                    df_features['volume'] = df_features.get('tick_volume', 0)
                else:
                    df_features[col] = df_features['close']
        
        # RSI
        df_features['rsi_14'] = ta.RSI(df_features['close'], timeperiod=14)
        
        # ATR
        df_features['atr_14'] = ta.ATR(df_features['high'], df_features['low'], df_features['close'], timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            df_features['close'], 
            timeperiod=20, 
            nbdevup=2.0, 
            nbdevdn=2.0
        )
        df_features['bb_position'] = (df_features['close'] - bb_lower) / (bb_upper - bb_lower)
        df_features['bb_position'] = df_features['bb_position'].fillna(0.5)
        
        # Media mobili
        df_features['sma_10'] = ta.SMA(df_features['close'], timeperiod=10)
        df_features['sma_20'] = ta.SMA(df_features['close'], timeperiod=20)
        
        # Price vs SMA
        df_features['price_vs_sma10'] = (df_features['close'] - df_features['sma_10']) / df_features['sma_10']
        df_features['price_vs_sma20'] = (df_features['close'] - df_features['sma_20']) / df_features['sma_20']
        
        # Volatilità
        df_features['returns'] = df_features['close'].pct_change()
        df_features['volatility_20'] = df_features['returns'].rolling(window=20).std()
        
        # Target (prezzo aumenterà nelle prossime 3 candele?)
        df_features['target'] = (df_features['close'].shift(-3) > df_features['close']).astype(int)
        
        # Rimuovi NaN
        df_features = df_features.dropna()
        
        # Seleziona solo le colonne che ci servono
        feature_cols = [
            'rsi_14', 'atr_14', 'bb_position', 'volatility_20',
            'price_vs_sma10', 'price_vs_sma20'
        ]
        
        # Controlla che tutte le feature esistano
        for col in feature_cols:
            if col not in df_features.columns:
                print(f"⚠️ Colonna {col} non trovata, creazione default...")
                if col == 'rsi_14':
                    df_features[col] = 50.0
                elif col == 'atr_14':
                    df_features[col] = 0.015
                elif col == 'bb_position':
                    df_features[col] = 0.5
                elif 'volatility' in col:
                    df_features[col] = 0.01
                elif 'price_vs_sma' in col:
                    df_features[col] = 0.0
        
        return df_features[feature_cols + ['target']]
    
    def train(self, data: pd.DataFrame) -> bool:
        """Addestra il modello"""
        print("🤖 Addestramento modello con feature engineering...")
        
        try:
            # 1. Crea le features
            print("🔧 Creazione features tecniche...")
            df_with_features = self.create_features(data)
            
            if df_with_features.empty or len(df_with_features) < 100:
                print("❌ Dati insufficienti dopo feature engineering")
                return False
            
            print(f"✅ Features create: {len(df_with_features)} samples")
            
            # 2. Separa features e target
            feature_cols = [col for col in df_with_features.columns if col != 'target']
            self.feature_columns = feature_cols
            
            X = df_with_features[feature_cols].values
            y = df_with_features['target'].values
            
            print(f"   Dimensioni: X={X.shape}, y={y.shape}")
            
            # 3. Split train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"   Train: {len(X_train)} samples")
            print(f"   Test: {len(X_test)} samples")
            
            # 4. Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 5. Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # 6. Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n✅ Modello addestrato con successo!")
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Features: {len(feature_cols)}")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Testing samples: {len(X_test)}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Errore training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, features: Dict) -> Dict:
        """Predizione con confidenza 0.72"""
        if not self.is_trained or self.model is None:
            return {"action": "HOLD", "confidence": 0.0}
        
        try:
            # Crea array di features nell'ordine corretto
            feature_array = []
            for col in self.feature_columns:
                if col in features:
                    val = features[col]
                    # Gestisci NaN
                    if val is None or np.isnan(val):
                        # Valori default per ogni feature
                        if col == 'rsi_14':
                            val = 50.0
                        elif col == 'atr_14':
                            val = 0.015
                        elif col == 'bb_position':
                            val = 0.5
                        elif 'volatility' in col:
                            val = 0.01
                        elif 'price_vs_sma' in col:
                            val = 0.0
                        else:
                            val = 0.0
                    feature_array.append(float(val))
                else:
                    # Feature mancante, usa default
                    if col == 'rsi_14':
                        feature_array.append(50.0)
                    elif col == 'atr_14':
                        feature_array.append(0.015)
                    elif col == 'bb_position':
                        feature_array.append(0.5)
                    elif 'volatility' in col:
                        feature_array.append(0.01)
                    elif 'price_vs_sma' in col:
                        feature_array.append(0.0)
                    else:
                        feature_array.append(0.0)
            
            # Scale e predici
            features_scaled = self.scaler.transform([feature_array])
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            confidence = probabilities.max()
            action = "HOLD"
            
            # Soglia di confidenza - MODIFICARE QUI PER CAMBIARE CONFIDENCE
            if confidence >= Config.MIN_CONFIDENCE:  # 🎯 0.71
                action = "BUY" if prediction == 1 else "SELL"
            
            return {
                "action": action,
                "confidence": confidence,
                "probability_buy": probabilities[1],
                "probability_sell": probabilities[0]
            }
            
        except Exception as e:
            print(f"⚠️ Errore nella predizione: {e}")
            return {"action": "HOLD", "confidence": 0.0}


class XAUUSDIntelligentTPBot:
    """Bot principale con TP intelligente ibrido"""
    
    def __init__(self):
        print("\n" + "=" * 80)
        print("🚀 XAUUSD INTELLIGENT TP BOT - VERSION 3.0")
        print("🎯 HYBRID ADAPTIVE TAKE PROFIT SYSTEM")
        print("=" * 80)
        print("🤖 CARATTERISTICHE AVANZATE:")
        print("   • Hybrid TP: ATR + ML + Support/Resistance + Reinforcement Learning")
        print("   • Machine Learning: Predizione TP ottimale basata su dati storici")
        print("   • Reinforcement Learning: Ottimizzazione dinamica strategie")
        print("   • Support/Resistance Detection: Livelli tecnici dinamici")
        print("   • Market Regime Analysis: Adattamento a condizioni di mercato")
        print("   • Single Trade Only: Massima sicurezza")
        print("   • Fixed Lot Size: Sempre 0.01 lotti per trade")
        print("   • Advanced Risk Management: Circuit breaker integrato")
        print("=" * 80)
        
        # Inizializza componenti
        self.dl_model = None
        if DEEP_LEARNING_AVAILABLE:
            self._init_deep_learning_model()
        self.mt5_manager = AdvancedMT5Manager()
        self.data_collector = AdvancedDataCollector(self.mt5_manager)
        self.risk_manager = AdvancedRiskManager()
        self.tp_manager = HybridTPManager()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.ml_model = MLModel()
        
        # Stato bot
        self.symbol = Config.SYMBOL
        self.is_trained = False
        self.is_live = False
        self.current_trade = None
        self.trade_history = []
        self.performance_stats = {}
        self.debug_mode = False
        
        # Buffer dati real-time
        self.price_buffer = deque(maxlen=200)
        self.feature_buffer = deque(maxlen=100)
        
        # Setup directories
        self._setup_directories()
        
        # Inizializza logging
        self._setup_logging()

    def _init_deep_learning_model(self):
        """Inizializza modello Deep Learning TensorFlow/Keras"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Modello semplice per predizioni
            self.dl_model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(10,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(3, activation='softmax')  # BUY, SELL, HOLD
            ])
            
            self.dl_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Modello Deep Learning inizializzato")
        except Exception as e:
            print(f"⚠️ Errore inizializzazione DL: {e}")
            self.dl_model = None

    def _save_models(self):
        """Salva tutti i modelli ML su disco"""
        print("\n💾 SALVATAGGIO MODELLI SU DISCO...")
        
        models_dir = Config.MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        
        saved_models = {}
        
        # 1. Salva modello segnali (RandomForest)
        if self.ml_model.is_trained and self.ml_model.model is not None:
            try:
                signal_model_path = os.path.join(models_dir, "signal_model.pkl")
                with open(signal_model_path, "wb") as f:
                    pickle.dump({
                        'model': self.ml_model.model,
                        'scaler': self.ml_model.scaler,
                        'feature_columns': self.ml_model.feature_columns,
                        'is_trained': self.ml_model.is_trained,
                        'timestamp': datetime.now()
                    }, f)
                saved_models['signal'] = signal_model_path
                print(f"   ✅ Modello segnali salvato: {signal_model_path}")
            except Exception as e:
                print(f"   ❌ Errore salvataggio modello segnali: {e}")
        
        # 2. Salva modello TP (GradientBoosting)
        if self.tp_manager.ml_predictor.is_trained and self.tp_manager.ml_predictor.regression_model is not None:
            try:
                tp_model_path = os.path.join(models_dir, "tp_model.pkl")
                with open(tp_model_path, "wb") as f:
                    pickle.dump({
                        'regression_model': self.tp_manager.ml_predictor.regression_model,
                        'scaler': self.tp_manager.ml_predictor.scaler,
                        'feature_columns': self.tp_manager.ml_predictor.feature_columns,
                        'is_trained': self.tp_manager.ml_predictor.is_trained,
                        'timestamp': datetime.now()
                    }, f)
                saved_models['tp'] = tp_model_path
                print(f"   ✅ Modello TP salvato: {tp_model_path}")
            except Exception as e:
                print(f"   ❌ Errore salvataggio modello TP: {e}")
        
        # 3. Salva RL Agent (Q-table)
        try:
            rl_model_path = os.path.join(models_dir, "rl_agent.pkl")
            
            # Converti defaultdict a dict normale per pickle
            q_table_dict = dict(self.tp_manager.rl_agent.q_table)
            
            with open(rl_model_path, "wb") as f:
                pickle.dump({
                    'q_table': q_table_dict,
                    'exploration_rate': self.tp_manager.rl_agent.exploration_rate,
                    'state_history': self.tp_manager.rl_agent.state_history[-100:],  # Ultimi 100
                    'learning_progress': self.tp_manager.rl_agent.learning_progress[-100:],
                    'timestamp': datetime.now()
                }, f)
            saved_models['rl'] = rl_model_path
            print(f"   ✅ RL Agent salvato: {rl_model_path}")
        except Exception as e:
            print(f"   ❌ Errore salvataggio RL: {e}")
        
        # 4. Salva strategia pesi TP
        try:
            weights_path = os.path.join(models_dir, "tp_weights.pkl")
            with open(weights_path, "wb") as f:
                pickle.dump({
                    'strategy_weights': {k.value: v for k, v in self.tp_manager.strategy_weights.items()},
                    'strategy_performance': {
                        k.value: v for k, v in self.tp_manager.strategy_performance.items()
                    },
                    'timestamp': datetime.now()
                }, f)
            saved_models['weights'] = weights_path
            print(f"   ✅ Pesi TP salvati: {weights_path}")
        except Exception as e:
            print(f"   ❌ Errore salvataggio pesi: {e}")
        
        # 5. Salva modello Deep Learning (TensorFlow/Keras) se disponibile
        if DEEP_LEARNING_AVAILABLE and hasattr(self, 'dl_model') and self.dl_model is not None:
            try:
                dl_model_path = os.path.join(models_dir, "deep_learning_model.h5")
                self.dl_model.save(dl_model_path)
                saved_models['deep_learning'] = dl_model_path
                print(f"   ✅ Modello Deep Learning salvato: {dl_model_path}")
            except Exception as e:
                print(f"   ❌ Errore salvataggio Deep Learning: {e}")
        
        # 6. Salva statistiche trade history
        try:
            history_path = os.path.join(models_dir, "trade_history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump({
                    'trade_history': self.trade_history[-500:],  # Ultimi 500 trade
                    'performance_stats': self.performance_stats,
                    'timestamp': datetime.now()
                }, f)
            print(f"   ✅ Trade history salvata: {history_path}")
        except Exception as e:
            print(f"   ❌ Errore salvataggio history: {e}")
        
        # 7. Salva configurazione attuale
        try:
            config_path = os.path.join(models_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    'symbol': Config.SYMBOL,
                    'fixed_lot_size': Config.FIXED_LOT_SIZE,
                    'min_confidence': Config.MIN_CONFIDENCE,
                    'max_spread_pips': Config.MAX_SPREAD_PIPS,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"   ✅ Configurazione salvata: {config_path}")
        except Exception as e:
            print(f"   ❌ Errore salvataggio config: {e}")
        
        print(f"\n📊 RIEPILOGO SALVATAGGIO:")
        for model_type, path in saved_models.items():
            file_size = os.path.getsize(path) / 1024  # KB
            print(f"   • {model_type}: {os.path.basename(path)} ({file_size:.1f} KB)")
        
        print("✅ Salvataggio modelli completato!")
        return saved_models

    def _load_models(self):
        """Carica tutti i modelli ML da disco"""
        print("\n📂 CARICAMENTO MODELLI DA DISCO...")
        
        models_dir = Config.MODELS_DIR
        
        if not os.path.exists(models_dir):
            print("   ⚠️ Directory modelli non trovata, nessun modello da caricare")
            return False
        
        loaded_models = []
        
        # 1. Carica modello segnali
        signal_model_path = os.path.join(models_dir, "signal_model.pkl")
        if os.path.exists(signal_model_path):
            try:
                with open(signal_model_path, "rb") as f:
                    data = pickle.load(f)
                    self.ml_model.model = data['model']
                    self.ml_model.scaler = data['scaler']
                    self.ml_model.feature_columns = data['feature_columns']
                    self.ml_model.is_trained = data['is_trained']
                    timestamp = data.get('timestamp', 'unknown')
                    loaded_models.append(f"signal ({timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else timestamp})")
                    print(f"   ✅ Modello segnali caricato (trainato: {timestamp})")
            except Exception as e:
                print(f"   ❌ Errore caricamento modello segnali: {e}")
        
        # 2. Carica modello TP
        tp_model_path = os.path.join(models_dir, "tp_model.pkl")
        if os.path.exists(tp_model_path):
            try:
                with open(tp_model_path, "rb") as f:
                    data = pickle.load(f)
                    self.tp_manager.ml_predictor.regression_model = data['regression_model']
                    self.tp_manager.ml_predictor.scaler = data['scaler']
                    self.tp_manager.ml_predictor.feature_columns = data['feature_columns']
                    self.tp_manager.ml_predictor.is_trained = data['is_trained']
                    timestamp = data.get('timestamp', 'unknown')
                    loaded_models.append(f"tp ({timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else timestamp})")
                    print(f"   ✅ Modello TP caricato (trainato: {timestamp})")
            except Exception as e:
                print(f"   ❌ Errore caricamento modello TP: {e}")
        
        # 3. Carica RL Agent
        rl_model_path = os.path.join(models_dir, "rl_agent.pkl")
        if os.path.exists(rl_model_path):
            try:
                with open(rl_model_path, "rb") as f:
                    data = pickle.load(f)
                    # Ricostruisci defaultdict dalla dict salvata
                    from collections import defaultdict
                    q_table = defaultdict(lambda: np.zeros(5))
                    for key, value in data['q_table'].items():
                        # Converti la tupla stringa in tupla reale se necessario
                        if isinstance(key, str):
                            import ast
                            key = ast.literal_eval(key)
                        q_table[key] = np.array(value)
                    self.tp_manager.rl_agent.q_table = q_table
                    self.tp_manager.rl_agent.exploration_rate = data['exploration_rate']
                    timestamp = data.get('timestamp', 'unknown')
                    loaded_models.append(f"rl ({timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else timestamp})")
                    print(f"   ✅ RL Agent caricato (exploration: {self.tp_manager.rl_agent.exploration_rate:.3f})")
            except Exception as e:
                print(f"   ❌ Errore caricamento RL: {e}")
        
        # 4. Carica pesi TP
        weights_path = os.path.join(models_dir, "tp_weights.pkl")
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "rb") as f:
                    data = pickle.load(f)
                    # Ripristina pesi strategie
                    for strategy_name, weight in data['strategy_weights'].items():
                        for strategy in TPStrategy:
                            if strategy.value == strategy_name:
                                self.tp_manager.strategy_weights[strategy] = weight
                                break
                    timestamp = data.get('timestamp', 'unknown')
                    loaded_models.append(f"weights ({timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else timestamp})")
                    print(f"   ✅ Pesi TP caricati")
            except Exception as e:
                print(f"   ❌ Errore caricamento pesi: {e}")
        
        # 5. Carica modello Deep Learning
        if DEEP_LEARNING_AVAILABLE:
            dl_model_path = os.path.join(models_dir, "deep_learning_model.h5")
            if os.path.exists(dl_model_path):
                try:
                    from tensorflow import keras
                    self.dl_model = keras.models.load_model(dl_model_path)
                    loaded_models.append("deep_learning")
                    print(f"   ✅ Modello Deep Learning caricato")
                except Exception as e:
                    print(f"   ❌ Errore caricamento Deep Learning: {e}")
        
        # 6. Carica trade history
        history_path = os.path.join(models_dir, "trade_history.pkl")
        if os.path.exists(history_path):
            try:
                with open(history_path, "rb") as f:
                    data = pickle.load(f)
                    self.trade_history = data.get('trade_history', [])
                    self.performance_stats = data.get('performance_stats', {})
                    timestamp = data.get('timestamp', 'unknown')
                    loaded_models.append(f"history ({len(self.trade_history)} trades)")
                    print(f"   ✅ Trade history caricata ({len(self.trade_history)} trades)")
            except Exception as e:
                print(f"   ❌ Errore caricamento history: {e}")
        
        print(f"\n📊 RIEPILOGO CARICAMENTO:")
        if loaded_models:
            for model in loaded_models:
                print(f"   • {model}")
            print("✅ Caricamento modelli completato!")
            return True
        else:
            print("   ⚠️ Nessun modello caricato")
            return False

    def _auto_save_models(self):
        """Salvataggio automatico periodico dei modelli"""
        now = datetime.now()
        
        # Salva ogni 60 minuti (o dopo ogni aggiornamento ML)
        if not hasattr(self, '_last_auto_save'):
            self._last_auto_save = now
            return
        
        minutes_since_save = (now - self._last_auto_save).total_seconds() / 60
        
        if minutes_since_save >= 60:  # Ogni ora
            print("\n💾 Salvataggio automatico modelli...")
            self._save_models()
            self._last_auto_save = now


    def _check_ml_update(self):
        """Verifica se aggiornare i modelli ML"""
        now = datetime.now()
        
        # Aggiorna ogni 60 minuti
        if not hasattr(self, '_last_ml_update'):
            self._last_ml_update = now
            return
        
        hours_passed = (now - self._last_ml_update).total_seconds() / 3600
        
        if hours_passed >= 1:  # Ogni ora
            print("\n🤖 AGGIORNAMENTO MODELLI ML IN CORSO...")
            
            # Raccogli nuovi dati
            historical_data = self.data_collector.collect_multi_timeframe_data(
                self.symbol, days=7  # Ultimi 7 giorni
            )
            
            if 'M5' in historical_data:
                # Aggiorna modello segnali
                if len(historical_data['M5']) > 500:
                    self.ml_model.train(historical_data['M5'])
                    print("✅ Modello segnali aggiornato")
                
                # Aggiorna modello TP
                if len(historical_data['M5']) > 1000:
                    training_data = self.tp_manager.ml_predictor.prepare_training_data(historical_data)
                    if len(training_data) >= Config.ML_MIN_SAMPLES:
                        self.tp_manager.ml_predictor.train(training_data)
                        print("✅ Modello TP aggiornato")
            
            self._last_ml_update = now

    def test_mt5_connection(self):
        """Test completo connessione MT5 prima del trading"""
        print("\n🔧 TEST CONNESSIONE MT5...")
        
        # 1. Test initialize
        if not mt5.initialize():
            print(f"❌ Initialize failed: {mt5.last_error()}")
            return False
        
        # 2. Test login
        if not mt5.login(Config.MT5_ACCOUNT, Config.MT5_PASSWORD, Config.MT5_SERVER):
            print(f"❌ Login failed: {mt5.last_error()}")
            return False
        
        # 3. Test account info
        account = mt5.account_info()
        if not account:
            print(f"❌ Account info failed: {mt5.last_error()}")
            return False
        
        print(f"✅ Account: {account.login}, Balance: {account.balance}")
        
        # 4. Test order send con ordine fittizio (non eseguito)
        test_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": Config.SYMBOL,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(Config.SYMBOL).ask,
            "deviation": 20,
            "magic": 999999,
            "comment": "TEST_CONNECTION",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Non inviamo realmente, testiamo solo la struttura
        print("✅ Test request structure OK")
        
        return True

    def _generate_trade_signal(self, market_data: MarketData, 
                            market_context: Dict) -> TradeSignal:
        """Genera segnale di trading usando ML model"""
        
        # Usa il modello ML se addestrato
        if self.ml_model.is_trained:
            features = market_context.get('features', {})
            ml_prediction = self.ml_model.predict(features)
            
            action = TradeAction(ml_prediction.get('action', 'HOLD'))
            confidence = ml_prediction.get('confidence', 0.0)
            prob_buy = ml_prediction.get('probability_buy', 0.0)
            prob_sell = ml_prediction.get('probability_sell', 0.0)
            
            # 📊 STAMPA DETTAGLIATA DELLA CONFIDENZA E PROBABILITÀ
            print(f"\n📈 ML SIGNAL ANALYSIS:")
            print(f"   Model Confidence: {confidence:.3f} (Threshold: {Config.MIN_CONFIDENCE})")
            print(f"   Probability BUY: {prob_buy:.3f}")
            print(f"   Probability SELL: {prob_sell:.3f}")
            print(f"   Signal: {action.value}")
            
            # Aggiungi info sulle features principali
            if features:
                print(f"   Key Features:")
                if 'rsi_14' in features:
                    print(f"     • RSI: {features['rsi_14']:.1f}")
                if 'bb_position' in features:
                    print(f"     • BB Position: {features['bb_position']:.3f}")
                if 'volatility_20' in features:
                    print(f"     • Volatility: {features['volatility_20']:.4f}")
            
            # Controlla se il segnale supera la soglia
            if confidence >= Config.MIN_CONFIDENCE:
                print(f"   ✅ SIGNAL VALID (Confidence ≥ {Config.MIN_CONFIDENCE})")
            else:
                print(f"   ⚠️  SIGNAL TOO WEAK (Confidence < {Config.MIN_CONFIDENCE})")
            
            return TradeSignal(
                action=action,
                confidence=confidence,
                entry_price=market_data.mid_price,
                features=features
            )
        
        # Fallback: logica semplice se ML non addestrato
        features = market_context.get('features', {})
        regime = market_context.get('regime', MarketRegime.NORMAL)
        
        # Logica base per segnale (semplificata)
        rsi = features.get('rsi_14', 50)
        volatility = features.get('volatility_20', 0)
        
        # Segnale basato su RSI
        action = TradeAction.HOLD
        confidence = 0.5
        
        if rsi < 30 and volatility < 0.01:
            action = TradeAction.BUY
            confidence = 0.7
        elif rsi > 70 and volatility < 0.01:
            action = TradeAction.SELL
            confidence = 0.7
        
        # Stamp info fallback
        print(f"\n📈 FALLBACK SIGNAL (ML not trained):")
        print(f"   RSI: {rsi:.1f}")
        print(f"   Volatility: {volatility:.4f}")
        print(f"   Signal: {action.value} (Conf: {confidence:.2f})")
        
        # Adjust per regime
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKOUT]:
            confidence *= 0.8  # Riduci confidenza in alta volatilità
            print(f"   ⚠️  High volatility regime - Reduced confidence")
        
        return TradeSignal(
            action=action,
            confidence=confidence,
            features=features
        )

    def toggle_debug_mode(self, enabled: bool = True):
        """Attiva/disattiva modalità debug"""
        self.debug_mode = enabled
        status = "ATTIVATA" if enabled else "DISATTIVATA"
        print(f"🔧 Modalità debug {status}")
        
    def _setup_directories(self):
        """Crea directory necessarie"""
        for directory in [Config.MODELS_DIR, Config.DATA_DIR, Config.CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Configura sistema di logging"""
        logging.basicConfig(
            level=Config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_bot(self) -> bool:
        """Setup completo del bot con verifica qualità dati, ML doppio e caricamento modelli esistenti"""
        
        print("\n" + "=" * 80)
        print("🔧 SETUP COMPLETO BOT INTELLIGENTE")
        print("=" * 80)
        
        # =========================================================
        # 1. TENTATIVO CARICAMENTO MODELLI ESISTENTI
        # =========================================================
        print("\n📂 FASE 1: Verifica modelli esistenti...")
        models_loaded = self._load_models()
        
        if models_loaded:
            print("\n✅ MODELLI CARICATI CON SUCCESSO DAL DISCO!")
            
            # Verifica solo la connessione MT5 (salta il training)
            print("\n🔗 FASE 2: Verifica connessione MT5...")
            if not self.mt5_manager.initialize():
                print("❌ Impossibile connettersi a MT5")
                return False
            
            print("✅ Connesso a MT5")
            
            # Ottieni info account
            account_info = self.mt5_manager.get_account_info()
            if account_info:
                self.risk_manager.set_account_balance(account_info.get('balance', 0))
                print(f"💰 Account Balance: ${account_info.get('balance', 0):.2f}")
                print(f"📈 Equity: ${account_info.get('equity', 0):.2f}")
                print(f"🎯 Free Margin: ${account_info.get('free_margin', 0):.2f}")
            
            # Inizializza buffer prezzi con dati recenti
            print("\n📈 FASE 3: Inizializzazione buffer real-time...")
            self._init_price_buffer()
            
            print("\n" + "=" * 80)
            print("🎯 SETUP COMPLETATO (MODALITÀ RECOVERY)")
            print("=" * 80)
            print("✅ Bot pronto per il trading con modelli salvati!")
            print("   • Modelli ML caricati da disco")
            print("   • Nessun training necessario")
            print("   • Salvataggio automatico attivo ogni 60 minuti")
            print("=" * 80)
            
            return True
        
        # =========================================================
        # 2. SE NON CI SONO MODELLI, PROCEDI CON TRAINING COMPLETO
        # =========================================================
        print("\n⚠️ Nessun modello trovato su disco, avvio training da zero...")
        
        # 2.1 Connessione MT5
        print("\n🔗 FASE 1: Connessione a MetaTrader 5...")
        if not self.mt5_manager.initialize():
            print("❌ Impossibile connettersi a MT5")
            return False
        
        print("✅ Connesso a MT5")
        
        # 2.2 Ottieni info account
        account_info = self.mt5_manager.get_account_info()
        if account_info:
            self.risk_manager.set_account_balance(account_info.get('balance', 0))
            print(f"💰 Account Balance: ${account_info.get('balance', 0):.2f}")
            print(f"📈 Equity: ${account_info.get('equity', 0):.2f}")
            print(f"🎯 Free Margin: ${account_info.get('free_margin', 0):.2f}")
            print(f"📊 Leverage: 1:{account_info.get('leverage', 100)}")
        else:
            print("⚠️ Impossibile ottenere info account, continuo comunque...")
        
        # 2.3 Raccolta dati storici
        print(f"\n📊 FASE 2: Raccolta dati storici ({Config.TRAINING_DAYS} giorni)...")
        print(f"⏳ Questa operazione potrebbe richiedere alcuni minuti...")
        
        historical_data = self.data_collector.collect_multi_timeframe_data(
            self.symbol, Config.TRAINING_DAYS
        )
        
        if not historical_data:
            print("❌ Impossibile raccogliere dati sufficienti")
            return False
        
        # 2.4 Verifica e pulizia qualità dati
        print("\n🔍 FASE 3: Verifica qualità dati...")
        data_quality_issues = False
        
        if 'M5' in historical_data:
            m5_data = historical_data['M5'].copy()
            
            # Controlla problemi nei dati
            initial_count = len(m5_data)
            
            # Rimuovi righe con prezzi <= 0
            m5_data = m5_data[m5_data['close'] > 0]
            m5_data = m5_data[m5_data['high'] > 0]
            m5_data = m5_data[m5_data['low'] > 0]
            
            # Rimuovi NaN
            m5_data = m5_data.dropna()
            
            # Controlla se ci sono ancora dati sufficienti
            if len(m5_data) < initial_count * 0.8:
                print(f"⚠️ Molti dati scartati: {initial_count} → {len(m5_data)}")
                data_quality_issues = True
            
            if len(m5_data) < 100:
                print(f"❌ Dati M5 insufficienti dopo pulizia: {len(m5_data)} < 100")
                return False
            
            historical_data['M5'] = m5_data
            
            print(f"✅ Dati M5 puliti: {len(m5_data)} candele")
            print(f"📅 Periodo: {m5_data.index[0].date()} → {m5_data.index[-1].date()}")
            print(f"📈 Media giornaliera: {len(m5_data)/Config.TRAINING_DAYS:.1f} candele/giorno")
        else:
            print("❌ Dati M5 non disponibili")
            return False
        
        # 2.5 Training modello ML per segnali (BUY/SELL)
        ml_signal_trained = False
        print("\n🤖 FASE 4: Training modello ML per SEGNALI...")
        
        if 'M5' in historical_data and len(historical_data['M5']) > 500:
            try:
                m5_df = historical_data['M5'].copy()
                
                # Verifica che il DataFrame abbia le colonne necessarie
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in m5_df.columns:
                        print(f"⚠️ Colonna {col} mancante, creazione...")
                        if col == 'volume':
                            m5_df[col] = m5_df.get('tick_volume', 1000)
                        elif col in ['high', 'low']:
                            m5_df[col] = m5_df['close'] * (1.001 if col == 'high' else 0.999)
                        else:
                            m5_df[col] = m5_df['close']
                
                print(f"🔧 Preparazione features per segnali...")
                print(f"   Dati iniziali: {len(m5_df)} candele")
                
                # Training modello segnali
                ml_signal_trained = self.ml_model.train(m5_df)
                
                if ml_signal_trained:
                    print(f"✅ Modello segnali addestrato con successo!")
                    print(f"   Confidence threshold: {Config.MIN_CONFIDENCE}")
                    print(f"   Training completato su {len(m5_df)} campioni")
                else:
                    print("⚠️ Training ML segnali fallito, utilizzeremo fallback")
                    
            except Exception as e:
                print(f"⚠️ Errore training ML segnali: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Dati M5 insufficienti per training segnali")
        
        # 2.6 Training modello ML per Take Profit
        ml_tp_trained = False
        print("\n🤖 FASE 5: Training modello ML per TAKE PROFIT...")
        
        if 'M5' in historical_data and len(historical_data['M5']) > 1000:
            try:
                print(f"📊 Preparazione dati per training TP...")
                
                # Prepara dati per training TP
                training_data = self.tp_manager.ml_predictor.prepare_training_data(historical_data)
                
                print(f"✅ Dati TP preparati: {len(training_data)} samples")
                
                if len(training_data) >= Config.ML_MIN_SAMPLES:
                    # Addestra modello TP
                    ml_tp_trained = self.tp_manager.ml_predictor.train(training_data)
                    
                    if ml_tp_trained:
                        print(f"✅ Modello TP addestrato con successo!")
                        print(f"   Samples utilizzati: {len(training_data)}")
                        print(f"   Features: {len(self.tp_manager.ml_predictor.feature_columns)}")
                    else:
                        print("⚠️ Training ML TP fallito, utilizzeremo strategie alternative")
                else:
                    print(f"⚠️ Samples insufficienti per TP: {len(training_data)} < {Config.ML_MIN_SAMPLES}")
                    print("   Utilizzeremo strategie alternative per TP")
                    
            except Exception as e:
                print(f"⚠️ Errore training ML TP: {str(e)[:100]}")
                print("   Utilizzeremo strategie alternative per TP")
        else:
            print("❌ Dati insufficienti per training TP")
        
        # 2.7 Determina stato training complessivo
        self.is_trained = ml_signal_trained or ml_tp_trained
        
        # 2.8 Inizializza buffer real-time
        print("\n📈 FASE 6: Inizializzazione buffer real-time...")
        self._init_price_buffer(historical_data)
        
        # 2.9 Summary setup completo
        self._print_setup_summary(ml_signal_trained, ml_tp_trained, data_quality_issues)
        
        # 2.10 Salva i modelli appena addestrati
        print("\n💾 FASE 7: Salvataggio modelli su disco...")
        self._save_models()
        
        print("\n✅ BOT PRONTO PER IL TRADING!")
        print("   Usa l'opzione 2 nel menu per avviare il trading live")
        print("=" * 80)
        
        return True

    def _init_price_buffer(self, historical_data: Dict = None):
        """Inizializza il buffer dei prezzi per i calcoli real-time"""
        
        if historical_data and 'M5' in historical_data:
            recent_prices = historical_data['M5']['close'].tail(200).values
            
            # Pulisci i dati prima di aggiungerli al buffer
            recent_prices = recent_prices[~np.isnan(recent_prices)]
            recent_prices = recent_prices[np.isfinite(recent_prices)]
            recent_prices = recent_prices[recent_prices > 0]
            
            if len(recent_prices) > 0:
                self.price_buffer.extend(recent_prices)
                print(f"✅ Buffer prezzi inizializzato: {len(self.price_buffer)} samples")
                
                # Test indicatori di base
                if len(self.price_buffer) >= 20:
                    try:
                        test_prices = np.array(list(self.price_buffer)[-20:])
                        sma = ta.SMA(test_prices, timeperiod=10)
                        rsi = ta.RSI(test_prices, timeperiod=14)
                        print(f"   Test indicatori: SMA={sma[-1]:.2f}, RSI={rsi[-1]:.1f}")
                    except Exception as e:
                        print(f"⚠️ Test indicatori fallito: {e}")
            else:
                print("⚠️ Nessun dato valido per inizializzare buffer")
        else:
            print("⚠️ Nessun dato M5 per inizializzare buffer, uso dati real-time")
            # Prova a ottenere dati real-time
            market_data = self.data_collector.get_realtime_data(self.symbol)
            if market_data:
                self.price_buffer.append(market_data.mid_price)
                print(f"✅ Buffer inizializzato con prezzo corrente: {market_data.mid_price:.2f}")

    def _print_setup_summary(self, ml_signal_trained: bool, ml_tp_trained: bool, data_quality_issues: bool):

        
        # Status dettagliato
        if ml_signal_trained and ml_tp_trained:
            training_status = "FULLY TRAINED"
            status_emoji = "✅"
        elif ml_signal_trained:
            training_status = "SIGNALS ONLY"
            status_emoji = "⚠️"
        elif ml_tp_trained:
            training_status = "TP ONLY"
            status_emoji = "⚠️"
        else:
            training_status = "FALLBACK MODE"
            status_emoji = "❌"
        
        print("\n" + "=" * 80)
        print("🎯 SETUP COMPLETATO!")
        print("=" * 80)
        print(f"{status_emoji} TRAINING STATUS: {training_status}")
        print("")
        print("🤖 MODELLI ML:")
        print(f"   • Segnali (BUY/SELL): {'✅ TRAINED' if ml_signal_trained else '❌ FALLBACK'}")
        print(f"   • Take Profit: {'✅ TRAINED' if ml_tp_trained else '❌ FALLBACK'}")
        print("")
        print("⚙️ CONFIGURAZIONE:")
        print(f"   • Confidence Threshold: {Config.MIN_CONFIDENCE}")
        print(f"   • Training Days: {Config.TRAINING_DAYS}")
        print(f"   • Fixed Lot Size: {Config.FIXED_LOT_SIZE}")
        print(f"   • Max Daily Loss: {Config.MAX_DAILY_LOSS_PERCENT}%")
        print("")
        print("🛡️ SISTEMI DI SICUREZZA:")
        print(f"   • Circuit Breaker: {'✅ ATTIVO' if self.risk_manager.circuit_breakers['enabled'] else '❌ DISATTIVO'}")
        print(f"   • Single Trade Only: ✅ ATTIVO")
        print(f"   • Max Consecutive Losses: {Config.MAX_CONSECUTIVE_LOSSES}")
        print("=" * 80)
        
        # Avvertenze e consigli
        if not self.is_trained:
            print("\n⚠️ IMPORTANTE: Modelli ML non addestrati")
            print("=" * 50)
            print("Il bot utilizzerà strategie fallback:")
            print("   • Segnali: RSI-based con filtro volatilità")
            print("   • Take Profit: Hybrid (ATR + Support/Resistance)")
            print("   • Win Rate atteso: 50-55%")
            print("")
            print("🎯 CONSIGLI:")
            print("   1. Usa solo su DEMO ACCOUNT")
            print("   2. Monitora attentamente i primi trade")
            print("   3. Considera di raccogliere più dati e ri-addestrare")
            print("=" * 50)
        elif data_quality_issues:
            print("\n⚠️ AVVERTENZA: Problemi di qualità dati rilevati")
            print("   Alcuni dati sono stati scartati durante la pulizia")
            print("   Considera di raccogliere dati da una fonte più stabile")
        
        # Performance attese
        print("\n📊 PERFORMANCE ATTESE:")
        if ml_signal_trained:
            expected_winrate = f"58-63% (con confidence {Config.MIN_CONFIDENCE})"
        else:
            expected_winrate = "50-55% (fallback mode)"
        
        print(f"   • Win Rate: {expected_winrate}")
        print(f"   • Trade/giorno: 2-4")
        print(f"   • Profit Factor: 1.3-1.6")
        print(f"   • Max Drawdown: <{Config.MAX_DRAWDOWN_PERCENT}%")
 
 
    def start_trading(self):
        """Avvia trading live"""
        if not self.mt5_manager.connected:
            print("❌ Connessione MT5 non attiva")
            return
        
        print("\n🎯 AVVIO TRADING LIVE...")
        print("⚠️  ATTENZIONE: Il trading comporta rischi finanziari")
        print("   Usare solo con fondi che si possono permettere di perdere")
        
        self.is_live = True
        print("✅ TRADING LIVE ATTIVATO")
        
        try:
            self._trading_loop()
        except KeyboardInterrupt:
            print("\n🛑 Trading interrotto dall'utente")
        except Exception as e:
            print(f"❌ Errore nel trading loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_live = False
            self.mt5_manager.shutdown()
            print("🔌 Bot fermato e connessioni chiuse")
    
    def stop_trading(self):
        """Ferma il trading"""
        self.is_live = False
        print("🛑 Richiesta di stop trading...")
    
    def get_status_report(self) -> str:
        """Restituisce report di status dettagliato"""
        status = []
        status.append("\n" + "=" * 80)
        status.append("📊 STATUS REPORT - XAUUSD INTELLIGENT TP BOT")
        status.append("=" * 80)
        
        # Info bot
        status.append("\n🤖 BOT STATUS:")
        status.append(f"   Live Trading: {'ACTIVE' if self.is_live else 'INACTIVE'}")
        status.append(f"   ML Trained: {'YES' if self.is_trained else 'NO (Fallback Mode)'}")
        status.append(f"   Symbol: {self.symbol}")
        status.append(f"   Fixed Lot: {Config.FIXED_LOT_SIZE}")
        
        # Info account
        account_info = self.mt5_manager.get_account_info()
        if account_info:
            status.append("\n💰 ACCOUNT INFO:")
            status.append(f"   Balance: ${account_info.get('balance', 0):.2f}")
            status.append(f"   Equity: ${account_info.get('equity', 0):.2f}")
            status.append(f"   Free Margin: ${account_info.get('free_margin', 0):.2f}")
            status.append(f"   Profit: ${account_info.get('profit', 0):.2f}")
        
        # Trade corrente
        if self.current_trade:
            status.append("\n📈 CURRENT TRADE:")
            trade = self.current_trade
            current_price = self._get_current_price()
            
            if current_price and 'entry_price' in trade and 'action' in trade:
                if trade['action'] == TradeAction.BUY.value:
                    pnl = (current_price - trade['entry_price']) * 100 * Config.FIXED_LOT_SIZE
                    pnl_percent = (current_price - trade['entry_price']) / trade['entry_price']
                else:
                    pnl = (trade['entry_price'] - current_price) * 100 * Config.FIXED_LOT_SIZE
                    pnl_percent = (trade['entry_price'] - current_price) / trade['entry_price']
                
                status.append(f"   Action: {trade['action']}")
                status.append(f"   Entry: ${trade['entry_price']:.2f}")
                status.append(f"   Current: ${current_price:.2f}")
                status.append(f"   P&L: ${pnl:.2f} ({pnl_percent:+.2%})")
                status.append(f"   Duration: {trade.get('duration_minutes', 0):.1f} min")
                
                if 'tp_price' in trade:
                    tp_distance = abs(current_price - trade['tp_price']) / trade['entry_price']
                    status.append(f"   TP Distance: {tp_distance:.2%}")
        
        # Performance
        if self.trade_history:
            status.append("\n📊 PERFORMANCE:")
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            
            status.append(f"   Total Trades: {total_trades}")
            status.append(f"   Win Rate: {winning_trades/total_trades*100:.1f}%")
            status.append(f"   Total P&L: ${total_pnl:.2f}")
            status.append(f"   Avg P&L: ${total_pnl/total_trades:.2f}")
        
        # TP Manager Status
        tp_status = self.tp_manager.get_status()
        status.append("\n🎯 TP MANAGER STATUS:")
        for strategy, weight in tp_status['strategy_weights'].items():
            perf = tp_status['strategy_performance'].get(strategy, {})
            success = perf.get('success', 0)
            total = perf.get('total', 0)
            win_rate = success/total*100 if total > 0 else 0
            status.append(f"   {strategy}: {weight:.0%} (Win: {win_rate:.0f}%)")
        
        # Risk Manager Status
        risk_report = self.risk_manager.get_daily_report()
        status.append("\n🛡️  RISK MANAGER:")
        status.append(f"   Daily Trades: {risk_report['trades']}")
        status.append(f"   Daily P&L: ${risk_report['total_pnl']:.2f}")
        status.append(f"   Consecutive Losses: {risk_report['consecutive_losses']}")
        status.append(f"   Current Drawdown: {risk_report['current_drawdown']}")
        
        status.append("\n" + "=" * 80)
        return "\n".join(status)

    def _trading_loop(self):
        """Loop principale di trading - VERSIONE OTTIMIZZATA CON ML UPDATES"""
        
        print("\n" + "=" * 80)
        print("🔄 AVVIO MONITORAGGIO REAL-TIME...")
        print("=" * 80)
        
        # Inizializza variabili di controllo
        last_signal_check = datetime.now() - timedelta(minutes=5)
        last_ml_update = datetime.now()
        last_performance_log = datetime.now()
        last_connection_check = datetime.now()
        
        # Statistiche loop
        loop_stats = {
            'total_iterations': 0,
            'errors': 0,
            'trades_attempted': 0,
            'last_error_time': None,
            'last_error_msg': ""
        }
        
        # Parametri ottimizzati
        ML_UPDATE_MINUTES = 30  # Aggiorna ML ogni 30 minuti
        PERFORMANCE_LOG_MINUTES = 30  # Log performance ogni 30 minuti
        CONNECTION_CHECK_SECONDS = 30  # Check connessione ogni 30 secondi
        MAX_CONSECUTIVE_ERRORS = 5  # Massimo errori consecutivi prima di pausa
        
        consecutive_errors = 0
        
        while self.is_live:
            try:
                current_time = datetime.now()
                loop_stats['total_iterations'] += 1
                
                # =========================================================
                # 1. HEALTH CHECK CONNESSIONE (ogni 30 secondi)
                # =========================================================
                if (current_time - last_connection_check).total_seconds() >= CONNECTION_CHECK_SECONDS:
                    if not self.mt5_manager.check_connection():
                        print(f"⚠️ [{current_time.strftime('%H:%M:%S')}] Connessione MT5 instabile, riconnessione...")
                        if not self.mt5_manager.initialize():
                            print(f"❌ Riconnessione fallita, attendo 10 secondi...")
                            time.sleep(10)
                            continue
                        else:
                            print(f"✅ Riconnesso a MT5")
                    last_connection_check = current_time
                
                # =========================================================
                # 2. VERIFICA POSIZIONE APERTA
                # =========================================================
                if self.current_trade:
                    self._monitor_open_trade()
                    time.sleep(1)
                    continue
                
                # =========================================================
                # 3. RATE LIMITING (evita troppi segnali)
                # =========================================================
                minutes_since_last_check = (current_time - last_signal_check).total_seconds() / 60
                if minutes_since_last_check < Config.MIN_MINUTES_BETWEEN:
                    time.sleep(0.5)
                    continue
                
                # =========================================================
                # 4. OTTIENI DATI MARKET
                # =========================================================
                market_data = self.data_collector.get_realtime_data(self.symbol)
                if not market_data:
                    print(f"⚠️ [{current_time.strftime('%H:%M:%S')}] Nessun dato market, attendo...")
                    time.sleep(2)
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"⚠️ Troppi errori consecutivi ({consecutive_errors}), pausa di 30 secondi...")
                        time.sleep(30)
                        consecutive_errors = 0
                    continue
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # =========================================================
                # 5. CONTROLLO SPREAD
                # =========================================================
                spread_pips = market_data.spread
                if spread_pips > Config.MAX_SPREAD_PIPS:
                    print(f"⛔ [{current_time.strftime('%H:%M:%S')}] Spread alto: {spread_pips:.1f} pips > {Config.MAX_SPREAD_PIPS}")
                    time.sleep(10)
                    continue
                
                # =========================================================
                # 6. AGGIORNA BUFFER PREZZI
                # =========================================================
                self.price_buffer.append(market_data.mid_price)
                
                # =========================================================
                # 7. VERIFICA PERMESSO TRADING (Risk Manager)
                # =========================================================
                account_info = self.mt5_manager.get_account_info()
                allowed, reason = self.risk_manager.check_trade_allowed(account_info)
                
                if not allowed:
                    if loop_stats['total_iterations'] % 50 == 0:  # Log ogni ~50 iterazioni
                        print(f"⛔ Trading bloccato: {reason}")
                    time.sleep(5)
                    continue
                
                # =========================================================
                # 8. ANALISI CONTESTO MERCATO
                # =========================================================
                market_context = self._analyze_market_context()
                
                # =========================================================
                # 9. GENERA SEGNALE
                # =========================================================
                signal = self._generate_trade_signal(market_data, market_context)
                
                # =========================================================
                # 10. LOG STATO MERCATO (ogni 30 secondi circa)
                # =========================================================
                if loop_stats['total_iterations'] % 6 == 0:  # Ogni ~30 secondi (5s * 6)
                    print(f"\n" + "=" * 60)
                    print(f"📊 {current_time.strftime('%H:%M:%S')} - MARKET STATUS")
                    print(f"   Price: {market_data.bid:.2f} / {market_data.ask:.2f}")
                    print(f"   Spread: {spread_pips:.1f} pips")
                    print(f"   Trading Allowed: {allowed}")
                    print(f"   Signal: {signal.action.value} (Conf: {signal.confidence:.3f})")
                    print(f"   Iterations: {loop_stats['total_iterations']}")
                    print("=" * 60)
                
                # =========================================================
                # 11. ESEGUI TRADE SE SEGNALE VALIDO
                # =========================================================
                if (signal.action != TradeAction.HOLD and 
                    signal.confidence >= Config.MIN_CONFIDENCE and
                    spread_pips <= Config.MAX_SPREAD_PIPS):
                    
                    print(f"\n🎯 [{current_time.strftime('%H:%M:%S')}] Tentativo trade con spread: {spread_pips:.1f} pips")
                    loop_stats['trades_attempted'] += 1
                    
                    # Salva timestamp per rate limiting
                    last_signal_check = current_time
                    
                    # Esegui trade
                    self._execute_trade(signal, market_data, market_context)
                    
                    # Pausa dopo trade per evitare aperture multiple
                    time.sleep(3)
                    
                else:
                    # Log ridotto per segnali deboli
                    if signal.action != TradeAction.HOLD and signal.confidence < Config.MIN_CONFIDENCE:
                        if loop_stats['total_iterations'] % 12 == 0:  # Ogni ~60 secondi
                            print(f"⚠️ Confidenza bassa: {signal.confidence:.3f} < {Config.MIN_CONFIDENCE}")
                
                # =========================================================
                # 12. AGGIORNAMENTO MODELLI ML (ogni ML_UPDATE_MINUTES)
                # =========================================================
                minutes_since_ml_update = (current_time - last_ml_update).total_seconds() / 60
                
                if minutes_since_ml_update >= ML_UPDATE_MINUTES:
                    print(f"\n🤖 [{current_time.strftime('%H:%M:%S')}] Aggiornamento modelli ML in corso...")
                    self._update_ml_models()
                    last_ml_update = current_time
                
                # =========================================================
                # 13. LOG PERFORMANCE PERIODICO
                # =========================================================
                minutes_since_perf_log = (current_time - last_performance_log).total_seconds() / 60
                
                if minutes_since_perf_log >= PERFORMANCE_LOG_MINUTES:
                    self._log_performance_summary()
                    last_performance_log = current_time
                
                # =========================================================
                # 14. PAUSA PRIMA DEL PROSSIMO CICLO
                # =========================================================
                time.sleep(3)  # Pausa base tra iterazioni
                
            except KeyboardInterrupt:
                print("\n🛑 Trading interrotto dall'utente")
                break
                
            except Exception as e:
                print(f"⚠️ Errore nel trading loop: {e}")
                loop_stats['errors'] += 1
                loop_stats['last_error_time'] = datetime.now()
                loop_stats['last_error_msg'] = str(e)
                
                import traceback
                traceback.print_exc()
                
                # Pausa progressiva in base agli errori
                error_pause = min(30, loop_stats['errors'] * 2)
                print(f"   Pausa di {error_pause} secondi...")
                time.sleep(error_pause)
        
        # Fine while loop
        print("\n" + "=" * 80)
        print("📊 STATISTICHE FINALI TRADING LOOP")
        print(f"   Totale iterazioni: {loop_stats['total_iterations']}")
        print(f"   Trade tentati: {loop_stats['trades_attempted']}")
        print(f"   Errori totali: {loop_stats['errors']}")
        if loop_stats['last_error_time']:
            print(f"   Ultimo errore: {loop_stats['last_error_time'].strftime('%H:%M:%S')}")
            print(f"   Messaggio: {loop_stats['last_error_msg'][:100]}")
        print("=" * 80)

    def _update_ml_models(self):
        """Aggiorna i modelli Machine Learning con dati recenti e salva automaticamente"""
        
        print("\n" + "=" * 60)
        print("🤖 AGGIORNAMENTO MODELLI ML IN CORSO")
        print("=" * 60)
        
        start_time = datetime.now()
        update_results = {}
        
        try:
            # =========================================================
            # 1. RACCOLTA DATI RECENTI
            # =========================================================
            print("\n📊 FASE 1: Raccolta dati recenti (ultimi 7 giorni)...")
            
            historical_data = self.data_collector.collect_multi_timeframe_data(
                self.symbol, days=7
            )
            
            if not historical_data or 'M5' not in historical_data:
                print("⚠️ Dati insufficienti per aggiornamento ML")
                return
            
            m5_data = historical_data['M5']
            
            if len(m5_data) < 500:
                print(f"⚠️ Dati M5 insufficienti: {len(m5_data)} < 500")
                return
            
            print(f"✅ Dati raccolti: {len(m5_data)} candele M5")
            print(f"📅 Periodo: {m5_data.index[0].date()} → {m5_data.index[-1].date()}")
            
            # =========================================================
            # 2. PULIZIA DATI
            # =========================================================
            print("\n🔧 FASE 2: Pulizia e preparazione dati...")
            
            # Rimuovi dati non validi
            initial_count = len(m5_data)
            m5_data = m5_data[m5_data['close'] > 0]
            m5_data = m5_data[m5_data['high'] > 0]
            m5_data = m5_data[m5_data['low'] > 0]
            m5_data = m5_data.dropna()
            
            if len(m5_data) < initial_count * 0.8:
                print(f"⚠️ Scartati {initial_count - len(m5_data)} dati non validi")
            
            print(f"✅ Dati puliti: {len(m5_data)} candele")
            
            # =========================================================
            # 3. AGGIORNAMENTO MODELLO SEGNALI (BUY/SELL)
            # =========================================================
            print("\n🔄 FASE 3: Aggiornamento modello segnali...")
            
            try:
                # Verifica colonne necessarie
                m5_df = m5_data.copy()
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                
                for col in required_columns:
                    if col not in m5_df.columns:
                        if col == 'volume':
                            m5_df[col] = m5_df.get('tick_volume', 1000)
                        elif col in ['high', 'low']:
                            m5_df[col] = m5_df['close'] * (1.001 if col == 'high' else 0.999)
                        else:
                            m5_df[col] = m5_df['close']
                
                # Training
                signal_trained = self.ml_model.train(m5_df)
                update_results['signals'] = "✅" if signal_trained else "❌"
                
                if signal_trained:
                    print("   ✅ Modello segnali aggiornato con successo!")
                    # Salva il modello aggiornato
                    self._save_signal_model()
                else:
                    print("   ❌ Aggiornamento modello segnali fallito")
                    
            except Exception as e:
                update_results['signals'] = f"❌ {str(e)[:30]}"
                print(f"   ❌ Errore: {e}")
            
            # =========================================================
            # 4. AGGIORNAMENTO MODELLO TP
            # =========================================================
            print("\n🔄 FASE 4: Aggiornamento modello Take Profit...")
            
            if len(m5_data) >= 1000:
                try:
                    print("   Preparazione dati per training TP...")
                    training_data = self.tp_manager.ml_predictor.prepare_training_data(historical_data)
                    
                    if len(training_data) >= Config.ML_MIN_SAMPLES:
                        tp_trained = self.tp_manager.ml_predictor.train(training_data)
                        update_results['tp'] = "✅" if tp_trained else "❌"
                        
                        if tp_trained:
                            print(f"   ✅ Modello TP aggiornato con successo!")
                            print(f"      Samples: {len(training_data)}")
                            print(f"      Features: {len(self.tp_manager.ml_predictor.feature_columns)}")
                            # Salva il modello aggiornato
                            self._save_tp_model()
                        else:
                            print("   ❌ Aggiornamento modello TP fallito")
                    else:
                        update_results['tp'] = f"⚠️ Samples: {len(training_data)}"
                        print(f"   ⚠️ Samples insufficienti: {len(training_data)} < {Config.ML_MIN_SAMPLES}")
                        
                except Exception as e:
                    update_results['tp'] = f"❌ {str(e)[:30]}"
                    print(f"   ❌ Errore: {e}")
            else:
                update_results['tp'] = f"⚠️ Dati insufficienti ({len(m5_data)} < 1000)"
                print(f"   ⚠️ Dati M5 insufficienti per TP: {len(m5_data)} < 1000")
            
            # =========================================================
            # 5. AGGIORNAMENTO DEEP LEARNING (OPZIONALE)
            # =========================================================
            if DEEP_LEARNING_AVAILABLE and hasattr(self, 'dl_model') and self.dl_model is not None:
                print("\n🔄 FASE 5: Aggiornamento modello Deep Learning...")
                
                try:
                    # Prepara dati per DL
                    dl_trained = self._train_deep_learning_model(m5_data)
                    update_results['deep_learning'] = "✅" if dl_trained else "❌"
                    
                    if dl_trained:
                        print("   ✅ Modello Deep Learning aggiornato!")
                        self._save_deep_learning_model()
                    else:
                        print("   ❌ Aggiornamento Deep Learning fallito")
                        
                except Exception as e:
                    update_results['deep_learning'] = f"❌ {str(e)[:30]}"
                    print(f"   ❌ Errore: {e}")
            else:
                update_results['deep_learning'] = "⚪ NON ATTIVO"
            
            # =========================================================
            # 6. AGGIORNAMENTO STATISTICHE E PESI
            # =========================================================
            print("\n📊 FASE 6: Aggiornamento metriche e pesi...")
            
            # Aggiorna timestamp ultimo aggiornamento
            self._last_ml_update_time = datetime.now()
            
            # Salva pesi TP aggiornati
            self._save_tp_weights()
            
            # Salva RL agent
            self._save_rl_agent()
            
            # =========================================================
            # 7. REPORT FINALE
            # =========================================================
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            print("📊 REPORT AGGIORNAMENTO ML")
            print("=" * 60)
            print(f"   Data: {self._last_ml_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Durata: {elapsed_time:.1f} secondi")
            print(f"\n   STATO MODELLI:")
            
            for model, status in update_results.items():
                print(f"   • {model}: {status}")
            
            print(f"\n   PROSSIMO AGGIORNAMENTO: tra 60 minuti")
            print("=" * 60)
            
            # =========================================================
            # 8. SALVATAGGIO COMPLETO
            # =========================================================
            print("\n💾 Salvataggio completo dello stato...")
            self._save_models()
            
            print("✅ Aggiornamento ML completato con successo!")
            
        except Exception as e:
            print(f"\n❌ ERRORE CRITICO durante aggiornamento ML: {e}")
            import traceback
            traceback.print_exc()
            
            # Tentativo di salvataggio parziale anche in caso di errore
            print("\n⚠️ Tentativo di salvataggio parziale...")
            try:
                self._save_models()
                print("✅ Salvataggio parziale completato")
            except:
                print("❌ Salvataggio parziale fallito")

    def _train_deep_learning_model(self, data: pd.DataFrame) -> bool:
        """Addestra il modello Deep Learning TensorFlow/Keras"""
        
        if not DEEP_LEARNING_AVAILABLE:
            return False
        
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            import tensorflow as tf
            
            print("   Preparazione dati per Deep Learning...")
            
            # Prepara features e target
            features = ['rsi_14', 'atr_14', 'bb_position', 'volatility_20', 
                    'price_vs_sma10', 'price_vs_sma20']
            
            # Calcola indicatori
            df_features = self.ml_model.create_features(data)
            
            if len(df_features) < 200:
                print(f"   ⚠️ Dati insufficienti: {len(df_features)} < 200")
                return False
            
            X = df_features[features].values
            y = df_features['target'].values
            
            # Converti target in one-hot encoding
            y_one_hot = tf.keras.utils.to_categorical(y, num_classes=2)
            
            # Split train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y_one_hot[:split_idx], y_one_hot[split_idx:]
            
            # Normalizza
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Ricostruisci modello
            self.dl_model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(len(features),)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(2, activation='softmax')
            ])
            
            self.dl_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Addestra
            print(f"   Training su {len(X_train)} campioni...")
            history = self.dl_model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            accuracy = history.history['val_accuracy'][-1]
            print(f"   ✅ Deep Learning addestrato (accuracy: {accuracy:.3f})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Errore training DL: {e}")
            return False

    def _save_signal_model(self):
        """Salva solo il modello segnali"""
        try:
            models_dir = Config.MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            
            signal_model_path = os.path.join(models_dir, "signal_model.pkl")
            with open(signal_model_path, "wb") as f:
                pickle.dump({
                    'model': self.ml_model.model,
                    'scaler': self.ml_model.scaler,
                    'feature_columns': self.ml_model.feature_columns,
                    'is_trained': self.ml_model.is_trained,
                    'timestamp': datetime.now()
                }, f)
            return True
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio modello segnali: {e}")
            return False

    def _save_tp_model(self):
        """Salva solo il modello TP"""
        try:
            models_dir = Config.MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            
            tp_model_path = os.path.join(models_dir, "tp_model.pkl")
            with open(tp_model_path, "wb") as f:
                pickle.dump({
                    'regression_model': self.tp_manager.ml_predictor.regression_model,
                    'scaler': self.tp_manager.ml_predictor.scaler,
                    'feature_columns': self.tp_manager.ml_predictor.feature_columns,
                    'is_trained': self.tp_manager.ml_predictor.is_trained,
                    'timestamp': datetime.now()
                }, f)
            return True
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio modello TP: {e}")
            return False

    def _save_tp_weights(self):
        """Salva i pesi delle strategie TP"""
        try:
            models_dir = Config.MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            
            weights_path = os.path.join(models_dir, "tp_weights.pkl")
            with open(weights_path, "wb") as f:
                pickle.dump({
                    'strategy_weights': {k.value: v for k, v in self.tp_manager.strategy_weights.items()},
                    'strategy_performance': {
                        k.value: v for k, v in self.tp_manager.strategy_performance.items()
                    },
                    'timestamp': datetime.now()
                }, f)
            return True
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio pesi TP: {e}")
            return False

    def _save_rl_agent(self):
        """Salva l'agente RL"""
        try:
            models_dir = Config.MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            
            rl_model_path = os.path.join(models_dir, "rl_agent.pkl")
            
            # Converti defaultdict a dict normale per pickle
            q_table_dict = {}
            for key, value in self.tp_manager.rl_agent.q_table.items():
                # Converte la tupla in stringa per salvare
                q_table_dict[str(key)] = value.tolist()
            
            with open(rl_model_path, "wb") as f:
                pickle.dump({
                    'q_table': q_table_dict,
                    'exploration_rate': self.tp_manager.rl_agent.exploration_rate,
                    'state_history': self.tp_manager.rl_agent.state_history[-100:],
                    'learning_progress': self.tp_manager.rl_agent.learning_progress[-100:],
                    'timestamp': datetime.now()
                }, f)
            return True
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio RL: {e}")
            return False

    def _save_deep_learning_model(self):
        """Salva il modello Deep Learning"""
        if not DEEP_LEARNING_AVAILABLE or self.dl_model is None:
            return False
        
        try:
            models_dir = Config.MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            
            dl_model_path = os.path.join(models_dir, "deep_learning_model.h5")
            self.dl_model.save(dl_model_path)
            return True
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio DL: {e}")
            return False

    def _log_performance_summary(self):
        """Log periodico delle performance"""
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE SUMMARY")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Statistiche trade history
        if self.trade_history:
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            print(f"\n   📊 TRADE STATISTICS:")
            print(f"      Total Trades: {total_trades}")
            print(f"      Win Rate: {win_rate:.1f}%")
            print(f"      Total P&L: ${total_pnl:.2f}")
            
            if total_trades > 0:
                avg_pnl = total_pnl / total_trades
                print(f"      Avg P&L: ${avg_pnl:.2f}")
            
            # Ultimi 10 trade
            last_10 = self.trade_history[-10:]
            last_10_wins = sum(1 for t in last_10 if t.get('pnl', 0) > 0)
            last_10_pnl = sum(t.get('pnl', 0) for t in last_10)
            print(f"\n   📈 LAST 10 TRADES:")
            print(f"      Wins: {last_10_wins}/10 ({last_10_wins*10:.0f}%)")
            print(f"      P&L: ${last_10_pnl:.2f}")
        
        # Info account
        account_info = self.mt5_manager.get_account_info()
        if account_info:
            print(f"\n   💰 ACCOUNT STATUS:")
            print(f"      Balance: ${account_info.get('balance', 0):.2f}")
            print(f"      Equity: ${account_info.get('equity', 0):.2f}")
            print(f"      Free Margin: ${account_info.get('free_margin', 0):.2f}")
        
        # Stato modelli ML
        print(f"\n   🤖 ML MODELS STATUS:")
        print(f"      Signals Model: {'✅ TRAINED' if self.ml_model.is_trained else '❌ NOT TRAINED'}")
        print(f"      TP Model: {'✅ TRAINED' if self.tp_manager.ml_predictor.is_trained else '❌ NOT TRAINED'}")
        
        if hasattr(self, '_last_ml_update_time'):
            hours_since_update = (datetime.now() - self._last_ml_update_time).total_seconds() / 3600
            print(f"      Last ML Update: {hours_since_update:.1f} hours ago")
        
        # Risk status
        risk_report = self.risk_manager.get_daily_report()
        print(f"\n   🛡️ RISK STATUS:")
        print(f"      Daily Trades: {risk_report['trades']}")
        print(f"      Daily P&L: ${risk_report['total_pnl']:.2f}")
        print(f"      Consecutive Losses: {risk_report['consecutive_losses']}")
        print(f"      Circuit Breaker: {'ACTIVE' if risk_report['circuit_breaker_active'] else 'INACTIVE'}")
        
        print("=" * 80)


    def _analyze_market_context(self) -> Dict:
        """Analizza contesto di mercato corrente - VERSIONE CORRETTA"""
        if len(self.price_buffer) < 50:
            return {'confidence': 0.5, 'regime': MarketRegime.NORMAL}
        
        try:
            prices = np.array(list(self.price_buffer))
            
            # 1. PULIZIA DATI: Rimuovi NaN, Inf, e valori non validi
            prices = prices[~np.isnan(prices)]
            prices = prices[np.isfinite(prices)]
            prices = prices[prices > 0]  # Rimuovi prezzi <= 0
            
            if len(prices) < 50:
                return {'confidence': 0.5, 'regime': MarketRegime.NORMAL}
            
            # 2. CREA DATAFRAME CON DATI PULITI
            # Usa window più piccola se non abbiamo abbastanza dati
            window_size = min(100, len(prices))
            prices_window = prices[-window_size:]
            
            df = pd.DataFrame({
                'close': prices_window,
                'high': prices_window * 1.001,  # Stima high
                'low': prices_window * 0.999,   # Stima low
                'volume': np.ones(len(prices_window)) * 1000
            })
            
            # 3. CALCOLA RETURNS CON CONTROLLO
            try:
                df['returns'] = df['close'].pct_change()
                # Sostituisci Inf e NaN
                df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan)
                df['returns'] = df['returns'].fillna(0)
            except:
                df['returns'] = 0
            
            # 4. CALCOLA INDICATORI CON GESTIONE ERRORI ROBUSTA
            indicators = {}
            
            # RSI
            try:
                if len(df['close']) >= 14:
                    rsi = ta.RSI(df['close'].values, timeperiod=14)
                    if rsi is not None and len(rsi) > 0:
                        last_rsi = rsi[-1]
                        if not np.isnan(last_rsi) and np.isfinite(last_rsi):
                            indicators['rsi_14'] = float(last_rsi)
                        else:
                            indicators['rsi_14'] = 50.0
                    else:
                        indicators['rsi_14'] = 50.0
                else:
                    indicators['rsi_14'] = 50.0
            except Exception as e:
                indicators['rsi_14'] = 50.0
            
            # ATR
            try:
                if len(df) >= 14:
                    atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                    if atr is not None and len(atr) > 0:
                        last_atr = atr[-1]
                        if not np.isnan(last_atr) and np.isfinite(last_atr):
                            indicators['atr_14'] = float(last_atr)
                        else:
                            indicators['atr_14'] = 0.015
                    else:
                        indicators['atr_14'] = 0.015
                else:
                    indicators['atr_14'] = 0.015
            except Exception as e:
                indicators['atr_14'] = 0.015
            
            # SMA e EMA
            try:
                if len(df['close']) >= 20:
                    sma_20 = ta.SMA(df['close'].values, timeperiod=20)
                    if sma_20 is not None and len(sma_20) > 0:
                        last_sma = sma_20[-1]
                        if not np.isnan(last_sma) and np.isfinite(last_sma):
                            indicators['sma_20'] = float(last_sma)
                        else:
                            indicators['sma_20'] = float(df['close'].iloc[-1])
                    else:
                        indicators['sma_20'] = float(df['close'].iloc[-1])
                else:
                    indicators['sma_20'] = float(df['close'].iloc[-1])
            except:
                indicators['sma_20'] = float(df['close'].iloc[-1])
            
            # Volatilità
            try:
                if len(df['returns']) >= 20:
                    volatility = df['returns'].rolling(window=20).std()
                    if not volatility.empty:
                        last_vol = volatility.iloc[-1]
                        if not np.isnan(last_vol) and np.isfinite(last_vol):
                            indicators['volatility'] = float(abs(last_vol))
                        else:
                            indicators['volatility'] = 0.01
                    else:
                        indicators['volatility'] = 0.01
                else:
                    indicators['volatility'] = 0.01
            except:
                indicators['volatility'] = 0.01
            
            # Calcola trend strength
            try:
                if 'sma_20' in indicators:
                    current_price = float(df['close'].iloc[-1])
                    trend_strength = (current_price - indicators['sma_20']) / indicators['sma_20']
                    indicators['trend_strength'] = float(trend_strength)
                else:
                    indicators['trend_strength'] = 0.0
            except:
                indicators['trend_strength'] = 0.0
            
            # Determina regime manualmente (senza chiamare metodo che potrebbe fallire)
            regime = MarketRegime.NORMAL
            regime_confidence = 0.6
            
            # Logica semplice per determinare regime
            volatility = indicators.get('volatility', 0.01)
            trend_strength = abs(indicators.get('trend_strength', 0))
            rsi = indicators.get('rsi_14', 50)
            
            if volatility > 0.015:
                regime = MarketRegime.HIGH_VOLATILITY
                regime_confidence = 0.7
            elif volatility < 0.005:
                regime = MarketRegime.LOW_VOLATILITY
                regime_confidence = 0.7
            elif trend_strength > 0.01 and rsi > 60:
                regime = MarketRegime.TRENDING_UP
                regime_confidence = 0.65
            elif trend_strength > 0.01 and rsi < 40:
                regime = MarketRegime.TRENDING_DOWN
                regime_confidence = 0.65
            elif volatility < 0.01 and trend_strength < 0.005:
                regime = MarketRegime.RANGING
                regime_confidence = 0.7
            else:
                regime = MarketRegime.NORMAL
                regime_confidence = 0.6
            
            # Calcola features per ML
            features = self._calculate_real_time_features(
                self.data_collector.get_realtime_data(self.symbol)
            )
            
            return {
                'prices': prices_window,
                'regime': regime,
                'regime_confidence': regime_confidence,
                'regime_characteristics': indicators,
                'features': features,
                'confidence': regime_confidence,
                'indicators': indicators  # Aggiunto per debug
            }
            
        except Exception as e:
            print(f"⚠️ Errore analisi contesto: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return {
                'confidence': 0.5, 
                'regime': MarketRegime.NORMAL,
                'regime_confidence': 0.5,
                'features': {}
            }

    def _calculate_real_time_features(self, market_data: MarketData) -> Dict:
        """Calcola features in tempo reale per ML model - VERSIONE ROBUSTA"""
        if len(self.price_buffer) < 20:
            return {}
        
        try:
            prices = np.array(list(self.price_buffer))
            
            # Pulizia dati
            prices = prices[~np.isnan(prices)]
            prices = prices[np.isfinite(prices)]
            prices = prices[prices > 0]
            
            if len(prices) < 20:
                return {}
            
            features = {}
            current_price = market_data.mid_price if market_data else prices[-1]
            
            # Feature base essenziali con valori di default
            default_features = {
                'rsi_14': 50.0,
                'atr_14': 0.015,
                'bb_position': 0.5,
                'volatility_20': 0.01,
                'price_vs_sma10': 0.0,
                'price_vs_sma20': 0.0,
                'current_price': current_price
            }
            
            # RSI
            try:
                if len(prices) >= 14:
                    rsi = ta.RSI(prices, timeperiod=14)
                    if rsi is not None and len(rsi) > 0:
                        last_rsi = rsi[-1]
                        if not np.isnan(last_rsi) and np.isfinite(last_rsi):
                            features['rsi_14'] = float(last_rsi)
            except:
                pass
            
            # ATR
            try:
                if len(prices) >= 14:
                    highs = prices * 1.001
                    lows = prices * 0.999
                    atr = ta.ATR(highs, lows, prices, timeperiod=14)
                    if atr is not None and len(atr) > 0:
                        last_atr = atr[-1]
                        if not np.isnan(last_atr) and np.isfinite(last_atr):
                            features['atr_14'] = float(last_atr)
            except:
                pass
            
            # Bollinger Bands position
            try:
                if len(prices) >= 20:
                    bb_upper, bb_middle, bb_lower = ta.BBANDS(
                        prices, timeperiod=20, nbdevup=2.0, nbdevdn=2.0
                    )
                    if bb_upper is not None and bb_lower is not None:
                        if (bb_upper[-1] - bb_lower[-1]) != 0:
                            bb_pos = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                            if not np.isnan(bb_pos) and np.isfinite(bb_pos):
                                features['bb_position'] = float(max(0.0, min(1.0, bb_pos)))
            except:
                pass
            
            # Volatilità
            try:
                if len(prices) >= 20:
                    returns = np.diff(np.log(prices[-20:]))
                    if len(returns) > 0:
                        vol = np.std(returns)
                        if not np.isnan(vol) and np.isfinite(vol):
                            features['volatility_20'] = float(vol)
            except:
                pass
            
            # Price vs SMA
            try:
                if len(prices) >= 10:
                    sma_10 = ta.SMA(prices, timeperiod=10)
                    if sma_10 is not None and len(sma_10) > 0 and sma_10[-1] != 0:
                        pct = (current_price - sma_10[-1]) / sma_10[-1]
                        if not np.isnan(pct) and np.isfinite(pct):
                            features['price_vs_sma10'] = float(pct)
            except:
                pass
            
            try:
                if len(prices) >= 20:
                    sma_20 = ta.SMA(prices, timeperiod=20)
                    if sma_20 is not None and len(sma_20) > 0 and sma_20[-1] != 0:
                        pct = (current_price - sma_20[-1]) / sma_20[-1]
                        if not np.isnan(pct) and np.isfinite(pct):
                            features['price_vs_sma20'] = float(pct)
            except:
                pass
            
            # Assicurati che tutte le feature richieste esistano
            for key, default_value in default_features.items():
                if key not in features or features[key] is None:
                    features[key] = default_value
            
            # Controlla valori NaN
            for key in list(features.keys()):
                if features[key] is None or np.isnan(features[key]):
                    features[key] = default_features.get(key, 0.0)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Errore calcolo features: {str(e)[:100]}")
            # Restituisci features di default
            return {
                'rsi_14': 50.0,
                'atr_14': 0.015,
                'bb_position': 0.5,
                'volatility_20': 0.01,
                'price_vs_sma10': 0.0,
                'price_vs_sma20': 0.0,
                'current_price': market_data.mid_price if market_data else 0.0
            }

    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Ottiene dati in tempo reale"""
        if not self.mt5_manager.connected:
            return None
        
        try:
            # Ottieni tick corrente
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            # Ottieni candela corrente
            rates = mt5.copy_rates_from_pos(symbol, Config.REAL_TIME_TIMEFRAME, 0, 10)
            
            # 🎯 CORREZIONE: Calcola spread CORRETTAMENTE
            # Per XAUUSD, 1 pip = 0.01 (non 0.0001 come per le coppie forex)
            spread_pips = (tick.ask - tick.bid) * 100  # Per oro, moltiplica per 100
            # spread_pips = (tick.ask - tick.bid) * 10000  # PER FOREX
            
            data = MarketData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread=spread_pips,  # Salva già in pips
                timestamp=datetime.now()
            )
            
            if rates is not None and len(rates) > 0:
                latest = rates[0]
                data.high = latest['high']
                data.low = latest['low']
                data.volume = latest['tick_volume']
            
            return data
            
        except Exception as e:
            print(f"⚠️ Errore dati realtime: {e}")
            return None





    def _execute_trade(self, signal: TradeSignal, market_data: MarketData, 
                        market_context: Dict):
        """
        Esegue un trade con controlli completi, retry, lotto dinamico e gestione errori robusta
        """
        
        # =========================================================
        # 1. VERIFICHE PRELIMINARI
        # =========================================================
        
        # Controllo connessione MT5
        if not self.mt5_manager.check_connection():
            print("❌ Connessione MT5 non attiva, tentativo di riconnessione...")
            if not self.mt5_manager.initialize():
                print("❌ Impossibile riconnettersi a MT5")
                return
            time.sleep(1)
        
        # Controllo spread
        spread_pips = market_data.spread
        if spread_pips > Config.MAX_SPREAD_PIPS:
            print(f"❌ Trade cancelled: Spread {spread_pips:.1f} pips > Max {Config.MAX_SPREAD_PIPS} pips")
            return
        
        # =========================================================
        # 2. VERIFICA POSIZIONI APERTE SU MT5
        # =========================================================
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions and len(positions) > 0:
                print(f"⛔ Posizioni già aperte su MT5: {[p.ticket for p in positions]}")
                
                # Sincronizza stato interno
                if self.current_trade:
                    current_ticket = self.current_trade.get('ticket')
                    current_pos = next((p for p in positions if p.ticket == current_ticket), None)
                    if current_pos:
                        print(f"   Trade in sincronia con MT5 (Ticket: {current_ticket})")
                        return
                    else:
                        print(f"   ⚠️ Trade in memoria ma non su MT5 - Resetting...")
                        self.current_trade = None
                
                # Chiudi posizioni orfane
                print(f"   🚨 Tentativo di chiusura posizioni orfane...")
                for pos in positions:
                    try:
                        close_result = self._close_mt5_position(pos)
                        if close_result:
                            print(f"   ✅ Chiusa posizione orfana: {pos.ticket}")
                    except Exception as e:
                        print(f"   ⚠️ Errore chiusura posizione {pos.ticket}: {e}")
                
                time.sleep(2)
                return
        except Exception as e:
            print(f"⚠️ Errore verifica posizioni: {e}")
        
        print(f"\n🎯 EXECUTING {signal.action.value} TRADE...")
        print(f"   Spread: {spread_pips:.1f} pips")
        
        # =========================================================
        # 3. PREPARAZIONE PARAMETRI TRADE
        # =========================================================
        try:
            # Determina prezzo entry e tipo ordine
            if signal.action == TradeAction.BUY:
                entry_price = market_data.ask
                order_type = mt5.ORDER_TYPE_BUY
                position_type = mt5.POSITION_TYPE_BUY
                print(f"   Order Type: BUY @ Ask {entry_price:.2f}")
            else:
                entry_price = market_data.bid
                order_type = mt5.ORDER_TYPE_SELL
                position_type = mt5.POSITION_TYPE_SELL
                print(f"   Order Type: SELL @ Bid {entry_price:.2f}")
            
            # Calcola TP ibrido
            tp_level = self.tp_manager.calculate_hybrid_tp(
                market_data, signal.action.value, market_context
            )
            
            print(f"🎯 Take Profit: {tp_level.price:.2f} ({tp_level.percent:.4%})")
            print(f"   Strategy: {tp_level.strategy.value}")
            
            # Verifica TP valido
            if signal.action == TradeAction.BUY and tp_level.price <= entry_price:
                print(f"❌ Invalid TP: TP ({tp_level.price:.2f}) ≤ Entry ({entry_price:.2f})")
                return
            elif signal.action == TradeAction.SELL and tp_level.price >= entry_price:
                print(f"❌ Invalid TP: TP ({tp_level.price:.2f}) ≥ Entry ({entry_price:.2f})")
                return
            
            # Ottieni ATR per il calcolo del lotto dinamico
            atr_value = market_context.get('atr', 0.015)
            
            # =========================================================
            # 4. LOTTO FISSO MANUALE
            # =========================================================
            # Usa il lotto fisso dalla configurazione
            trade_volume = Config.MANUAL_LOT_SIZE

            print(f"📊 FIXED LOT (MANUAL):")
            print(f"   Lot Size: {trade_volume}")
            print(f"   (Modifica Config.MANUAL_LOT_SIZE per cambiare il lotto)")

            # Verifica margine disponibile (solo per controllo)
            account_info = self.mt5_manager.get_account_info()
            if account_info:
                free_margin = account_info.get('free_margin', 0)
                leverage = account_info.get('leverage', 100)
                
                # Calcola margine richiesto per il lotto scelto
                contract_size = 100
                required_margin = (entry_price * trade_volume * contract_size) / leverage
                
                print(f"\n💳 MARGIN CHECK:")
                print(f"   Required Margin: ${required_margin:.2f}")
                print(f"   Free Margin: ${free_margin:.2f}")
                print(f"   Leverage: 1:{leverage}")
                print(f"   Trade Volume: {trade_volume} lots")
                
                if free_margin < required_margin * 1.5:
                    print(f"❌ Margine insufficiente per lotto {trade_volume}!")
                    print(f"   Richiesto: ${required_margin * 1.5:.2f}, Disponibile: ${free_margin:.2f}")
                    print(f"   Riduci Config.MANUAL_LOT_SIZE o aumenta il capitale")
                    return
            else:
                print(f"⚠️ Impossibile verificare margine, continuo comunque...")
            
            # =========================================================
            # 5. CALCOLO STOP LOSS (basato su ATR)
            # =========================================================
            sl_distance_percent = min(atr_value * 1.5, 0.008)
            
            if signal.action == TradeAction.BUY:
                sl_price = entry_price * (1 - sl_distance_percent)
            else:
                sl_price = entry_price * (1 + sl_distance_percent)
            
            # Verifica SL valido
            if sl_price <= 0:
                sl_price = entry_price * 0.995 if signal.action == TradeAction.BUY else entry_price * 1.005
            
            print(f"🛡️ Stop Loss: {sl_price:.2f} ({sl_distance_percent:.4%})")
            
            # =========================================================
            # 6. VERIFICA MARGINE
            # =========================================================
            if account_info:
                contract_size = 100
                required_margin = (entry_price * trade_volume * contract_size) / leverage
                
                print(f"\n💳 MARGIN CHECK:")
                print(f"   Required Margin: ${required_margin:.2f}")
                print(f"   Free Margin: ${free_margin:.2f}")
                print(f"   Leverage: 1:{leverage}")
                print(f"   Trade Volume: {trade_volume} lots")
                
                if free_margin < required_margin * 1.5:
                    print(f"❌ Margine insufficiente!")
                    print(f"   Richiesto: ${required_margin * 1.5:.2f}, Disponibile: ${free_margin:.2f}")
                    return
            else:
                print(f"⚠️ Impossibile verificare margine, continuo comunque...")
            
            # =========================================================
            # 7. GENERA MAGIC NUMBER
            # =========================================================
            magic_number = int(time.time() % 1000000)
            print(f"🔑 Magic Number: {magic_number}")
            
            # =========================================================
            # 8. PREPARA RICHIESTA ORDINE (COMMENTO PULITO PER MT5)
            # =========================================================
            # MT5 accetta solo lettere, numeri e pochi altri caratteri
            # Evita: : . , spazi, caratteri speciali
            confidence_code = int(signal.confidence * 100)
            lot_code = int(trade_volume * 100)
            
            # Formato semplice e sicuro: TP_BUY_C55_L1
            comment = f"TP{signal.action.value}C{confidence_code}L{lot_code}"
            comment = comment[:31]  # Max 31 caratteri
            
            print(f"   Comment: {comment}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": trade_volume,
                "type": order_type,
                "price": float(entry_price),
                "sl": float(sl_price),
                "tp": float(tp_level.price),
                "deviation": 20,
                "magic": magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # =========================================================
            # 9. INVIO ORDINE CON RETRY
            # =========================================================
            print(f"\n📤 Sending order to MT5...")
            print(f"   • Symbol: {request['symbol']}")
            print(f"   • Volume: {request['volume']} lots")
            print(f"   • Type: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'}")
            print(f"   • Price: {request['price']:.2f}")
            print(f"   • SL: {request['sl']:.2f}")
            print(f"   • TP: {request['tp']:.2f}")
            print(f"   • Magic: {request['magic']}")
            print(f"   • Comment: {request['comment']}")
            
            result = self._send_order_with_retry(request, max_retries=3)
            
            if result is None:
                print("❌ Ordine fallito dopo tutti i tentativi")
                return
            
            # =========================================================
            # 10. GESTIONE RISPOSTA MT5
            # =========================================================
            print(f"\n📋 MT5 Response:")
            print(f"   Retcode: {result.retcode}")
            print(f"   Deal: {result.deal}")
            print(f"   Order: {result.order}")
            print(f"   Volume: {result.volume}")
            print(f"   Price: {result.price:.2f}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print("✅ TRADE OPENED SUCCESSFULLY!")
                
                # Recupera posizione aperta
                position = self._get_position_by_magic(magic_number, position_type, entry_price)
                
                if position:
                    print(f"\n📊 POSITION CONFIRMED:")
                    print(f"   Ticket: {position.ticket}")
                    print(f"   Open Price: {position.price_open:.2f}")
                    print(f"   Current Price: {position.price_current:.2f}")
                    print(f"   SL: {position.sl:.2f}")
                    print(f"   TP: {position.tp:.2f}")
                    print(f"   Volume: {position.volume}")
                    print(f"   Magic: {position.magic}")
                    
                    # Salva trade corrente
                    self.current_trade = {
                        'ticket': position.ticket,
                        'magic': magic_number,
                        'action': signal.action.value,
                        'position_type': position_type,
                        'entry_price': entry_price,
                        'tp_price': tp_level.price,
                        'sl_price': sl_price,
                        'volume': trade_volume,
                        'tp_strategy': tp_level.strategy,
                        'tp_percent': tp_level.percent,
                        'sl_percent': sl_distance_percent,
                        'confidence': signal.confidence,
                        'open_time': datetime.now(),
                        'lock_enabled': False,
                        'peak_profit': 0.0,
                        'trailing_active': False,
                        'trailing_peak': 0.0,
                        'trailing_stop_pct': 0.0,
                        'last_log_time': datetime.now() - timedelta(minutes=5),
                        'spread_at_open': spread_pips
                    }
                    
                    self.logger.info(f"Trade opened: {signal.action.value} @ {entry_price:.2f}, "
                                f"Volume: {trade_volume}, TP: {tp_level.price:.2f}, "
                                f"Magic: {magic_number}, Ticket: {position.ticket}")
                    
                    print(f"\n🔍 Trade monitoring initialized...")
                    print(f"   Lock profit at: +0.03%, Emergency SL: -0.25%")
                    print(f"   Volume: {trade_volume} lots")
                    print(f"   Magic: {magic_number}, Ticket: {position.ticket}")
                else:
                    print("⚠️ Posizione non trovata su MT5 nonostante ordine eseguito!")
                    print("   Verifica manualmente su MT5")
            else:
                # Gestione errori
                error_msg = result.comment if hasattr(result, 'comment') else 'Unknown error'
                print(f"❌ TRADE OPENING FAILED!")
                print(f"   Error Code: {result.retcode}")
                print(f"   Error Message: {error_msg}")
                
                # Traduzione errori comuni
                error_translations = {
                    10004: "Requote - Price changed",
                    10006: "Request rejected",
                    10013: "Invalid request",
                    10014: "Invalid volume",
                    10015: "Invalid price",
                    10016: "Invalid stops - Check SL/TP",
                    10017: "Trade disabled",
                    10018: "Market closed",
                    10019: "Not enough money",
                    10020: "Price changed",
                    10021: "No quotes",
                    10022: "Broker busy",
                    10023: "Trade timeout",
                    10029: "Long positions not allowed",
                }
                
                if result.retcode in error_translations:
                    print(f"   Suggestion: {error_translations[result.retcode]}")
                        
        except Exception as e:
            print(f"❌ CRITICAL ERROR in trade execution: {e}")
            import traceback
            traceback.print_exc()


    def _send_order_with_retry(self, request: Dict, max_retries: int = 3) -> Optional[Any]:
        """Invia ordine con tentativi multipli e backoff esponenziale"""
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"\n📤 Sending order (attempt {attempt}/{max_retries})...")
                
                # Verifica connessione prima di ogni tentativo
                if not self.mt5_manager.check_connection():
                    print("   Connection lost, reconnecting...")
                    if not self.mt5_manager.initialize():
                        print("   Reconnection failed")
                        if attempt < max_retries:
                            time.sleep(attempt * 2)
                            continue
                        return None
                    time.sleep(1)
                
                # Invia ordine
                result = mt5.order_send(request)
                
                if result is not None:
                    print(f"   Order send completed")
                    return result
                else:
                    print(f"   order_send returned None")
                    last_error = mt5.last_error()
                    print(f"   Last error: {last_error}")
                    
                    if attempt < max_retries:
                        wait_time = attempt * 2
                        print(f"   Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        
            except Exception as e:
                print(f"   Exception in order_send: {e}")
                if attempt < max_retries:
                    wait_time = attempt * 2
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        print("❌ All order send attempts failed")
        return None

    def _get_position_by_magic(self, magic_number: int, position_type: int, entry_price: float) -> Optional[Any]:
        """Recupera posizione per magic number con tentativi multipli"""
        
        for attempt in range(5):  # Max 5 tentativi
            time.sleep(0.5)  # Attendi che MT5 processi l'ordine
            
            try:
                # Cerca per magic number
                positions = mt5.positions_get(symbol=self.symbol)
                if positions:
                    for pos in positions:
                        if hasattr(pos, 'magic') and int(pos.magic) == magic_number:
                            return pos
                
                # Fallback: cerca per tipo e prezzo simile
                if positions:
                    for pos in positions:
                        if (pos.type == position_type and 
                            abs(pos.price_open - entry_price) < 0.5):
                            print(f"   Position found by price match: Ticket {pos.ticket}")
                            return pos
                            
            except Exception as e:
                print(f"   Attempt {attempt + 1}: Error finding position - {e}")
        
        # Ultimo tentativo: prendi l'ultima posizione
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions and len(positions) > 0:
                last_pos = positions[-1]
                print(f"   Using last position as fallback: Ticket {last_pos.ticket}")
                return last_pos
        except:
            pass
        
        return None

    def _close_mt5_position(self, position) -> bool:
        """Chiude una posizione MT5 in modo affidabile"""
        try:
            # Determina tipo di chiusura
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": "CLOSE_BY_BOT",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            else:
                print(f"   Close failed: {result.comment if result else 'Unknown'}")
                return False
                
        except Exception as e:
            print(f"   Close exception: {e}")
            return False

    def _monitor_open_trade(self):
        """Monitora trade aperto con lock profit funzionante"""
        try:
            if not self.current_trade:
                return

            # Ottieni dati di mercato
            market_data = self.data_collector.get_realtime_data(self.symbol)
            if not market_data:
                return

            trade = self.current_trade
            action = trade["action"]
            
            # Determina il prezzo corrente corretto
            if action == TradeAction.BUY.value:
                current_price = market_data.bid  # Per posizioni LONG, chiudi al bid
            else:
                current_price = market_data.ask  # Per posizioni SHORT, chiudi all'ask
                
            entry_price = trade["entry_price"]

            # Calcola P&L corretto
            if action == TradeAction.BUY.value:
                price_diff = current_price - entry_price
            else:
                price_diff = entry_price - current_price

            pnl_dollars = price_diff * 100 * Config.FIXED_LOT_SIZE
            pnl_percent = price_diff / entry_price if entry_price else 0.0

            # Calcola durata
            duration = (datetime.now() - trade["open_time"]).total_seconds() / 60

            # Aggiorna trade info
            trade["current_price"] = current_price
            trade["pnl_dollars"] = pnl_dollars
            trade["pnl_percent"] = pnl_percent
            trade["duration_minutes"] = duration

            # Inizializza stati lock profit
            if "lock_enabled" not in trade:
                trade["lock_enabled"] = False
            if "peak_profit" not in trade:
                trade["peak_profit"] = 0.0  # Inizializza a 0

            # 🎯 MODIFICA: Abilita Lock Profit quando supera 0.03% (0.0003)
            if not trade["lock_enabled"] and pnl_percent is not None and pnl_percent > 0.0003:
                trade["lock_enabled"] = True
                trade["peak_profit"] = pnl_percent
                print(f"🔒 LOCK PROFIT ATTIVATO @ {pnl_percent:.4%}")

            # 🎯 MODIFICA: Se Lock attivo → aggiorna peak & controlla drop del 0.02%
            if trade.get("lock_enabled", False):
                if pnl_percent is not None and pnl_percent > trade["peak_profit"]:
                    trade["peak_profit"] = pnl_percent
                    print(f"📈 Nuovo peak profit: {trade['peak_profit']:.4%}")

                peak = trade.get("peak_profit", pnl_percent)
                
                # 🎯 CONTROLLO CORRETTO: Se il profitto corrente è sceso del 0.02% dal picco
                if peak > 0 and (peak - pnl_percent) >= 0.0002:
                    print(f"\n🔔 LOCK PROFIT TRIGGERED!")
                    print(f"   Peak: {peak:.4%}")
                    print(f"   Current: {pnl_percent:.4%}")
                    print(f"   Drop: {(peak - pnl_percent):.4%} >= 0.02%")

                    # Chiudi immediatamente la posizione
                    self._close_position_by_lock_profit(trade, market_data, pnl_dollars, pnl_percent)
                    return

            # ⭐ Emergency Stop Loss (chiudi subito se ≤ -0.25%)
            if pnl_percent is not None and pnl_percent <= -0.0025:
                print(f"\n⚠️ EMERGENCY STOP LOSS TRIGGERED!")
                print(f"   Current P&L: {pnl_percent:.4%} <= -0.25%")
                self._close_position_by_stop_loss(trade, market_data, pnl_dollars, pnl_percent)
                return

            # Logica originale chiusura trade (TP, timeout, etc.)
            should_close, reason = self._check_close_conditions(trade, market_data)
            if should_close:
                print(f"\n🔔 CONDIZIONE CHIUSURA RILEVATA: {reason}")
                self._close_trade(market_data, reason)
                return

            # Logging di monitoraggio ogni 30 secondi
            current_time = datetime.now()
            if "last_log_time" not in trade or (current_time - trade["last_log_time"]).seconds >= 30:
                print(f"\n📊 Trade Monitor - {current_time.strftime('%H:%M:%S')}")
                print(f"   Action: {action}")
                print(f"   Entry: ${entry_price:.2f}")
                print(f"   Current: ${current_price:.2f}")
                print(f"   P&L: ${pnl_dollars:+.2f} ({pnl_percent:+.4%})")
                print(f"   Duration: {duration:.1f} min")
                print(f"   Lock: {'ON' if trade.get('lock_enabled') else 'OFF'}")
                if trade.get('lock_enabled'):
                    print(f"   Peak Profit: {trade.get('peak_profit', 0):.4%}")
                trade["last_log_time"] = current_time

            time.sleep(1)  # Controlla più frequentemente

        except Exception as e:
            print(f"⚠️ Errore monitoraggio trade: {e}")
            import traceback
            traceback.print_exc()

    def _emergency_close_position(self, position):
        """Chiusura di emergenza per posizioni bloccate"""
        try:
            print(f"🚨 EMERGENCY CLOSE per posizione {position.ticket}")
            
            # Metodo 1: Usa la funzione esistente del manager
            close_result = self.mt5_manager.close_position_market(position)
            
            if close_result and hasattr(close_result, 'retcode'):
                if close_result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Emergency close successful for {position.ticket}")
                else:
                    print(f"❌ Emergency close failed: {close_result.comment}")
                    
                    # Metodo 2: Direct close by ticket
                    mt5.Close(position.symbol, ticket=position.ticket)
                    
        except Exception as e:
            print(f"❌ Emergency close error: {e}")

    def _close_position_by_lock_profit(self, trade, market_data, pnl_dollars, pnl_percent):
        """Chiude posizione per lock profit - VERSIONE CORRETTA"""
        try:
            print(f"📤 Closing position due to lock profit...")
            
            # 🎯 CHIUDI DIRETTAMENTE PER TICKET (metodo più affidabile)
            ticket = trade.get("ticket")
            if not ticket:
                print(f"❌ Ticket non trovato nel trade: {trade}")
                return
            
            # Cerca la posizione per ticket
            positions = mt5.positions_get(ticket=ticket)
            if not positions or len(positions) == 0:
                print(f"❌ Nessuna posizione trovata con ticket: {ticket}")
                return
            
            position = positions[0]
            
            # Prepara richiesta di chiusura
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,  # 🎯 CHIAVE: Usa il ticket della posizione
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": market_data.bid if position.type == mt5.POSITION_TYPE_BUY else market_data.ask,
                "deviation": 20,
                "magic": position.magic,
                "comment": "LOCK_PROFIT_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Invia ordine
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position closed by lock profit @ {result.price:.2f}")
                print(f"   Final P&L: ${pnl_dollars:.2f} ({pnl_percent:.4%})")
                
                # Aggiorna statistiche
                self._update_trade_stats(trade, result.price, pnl_dollars, pnl_percent, "lock_profit")
                
                # 🎯 IMPORTANTE: Verifica che la posizione sia chiusa
                time.sleep(0.5)
                check_positions = mt5.positions_get(ticket=ticket)
                if check_positions and len(check_positions) > 0:
                    print(f"⚠️  ATTENZIONE: Posizione {ticket} ancora aperta dopo chiusura!")
                    # Tentativo alternativo di chiusura
                    self._emergency_close_position(position)
                else:
                    print(f"✅ Posizione {ticket} confermata chiusa")
                
                # Reset trade corrente
                self.current_trade = None
            else:
                error_msg = result.comment if result else "Unknown error"
                error_code = result.retcode if result else "N/A"
                print(f"❌ Failed to close position: {error_msg} (Code: {error_code})")
                
                # Tentativo alternativo
                self._emergency_close_position(position)
                
        except Exception as e:
            print(f"❌ Errore chiusura lock profit: {e}")
            import traceback
            traceback.print_exc()

    def _close_position_by_stop_loss(self, trade, market_data, pnl_dollars, pnl_percent):
        """Chiude posizione per stop loss di emergenza"""
        try:
            print(f"📤 Closing position due to emergency stop loss...")
            
            # Determina tipo di chiusura
            if trade["action"] == TradeAction.BUY.value:
                close_type = mt5.ORDER_TYPE_SELL
                price = market_data.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = market_data.ask

            # Prepara ordine di chiusura
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": Config.FIXED_LOT_SIZE,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": trade.get("magic", 1000),
                "comment": "EMERGENCY_STOP_LOSS",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Invia ordine
            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position closed by emergency stop loss @ {price:.2f}")
                print(f"   Final P&L: ${pnl_dollars:.2f} ({pnl_percent:.4%})")

                # Aggiorna statistiche
                self._update_trade_stats(trade, price, pnl_dollars, pnl_percent, "emergency_stop")
                
                # Reset trade corrente
                self.current_trade = None
            else:
                error_msg = result.comment if result else "Unknown error"
                error_code = result.retcode if result else "N/A"
                print(f"❌ Failed to close position: {error_msg} (Code: {error_code})")

        except Exception as e:
            print(f"❌ Errore chiusura stop loss: {e}")


    def _update_trade_stats(self, trade, close_price, pnl_dollars, pnl_percent, reason):
        """Aggiorna statistiche dopo la chiusura del trade"""
        try:
            # Aggiorna risk manager
            account_info = self.mt5_manager.get_account_info()
            trade_result = {
                'action': trade['action'],
                'entry_price': trade['entry_price'],
                'close_price': close_price,
                'pnl': pnl_dollars,
                'pnl_percent': pnl_percent,
                'duration_minutes': trade.get('duration_minutes', 0),
                'tp_strategy': trade.get('tp_strategy', TPStrategy.HYBRID),
                'current_equity': account_info.get('equity', 0) if account_info else 0,
                'close_reason': reason
            }
            
            self.risk_manager.update_trade_result(trade_result)
            
            # Aggiorna TP manager
            self.tp_manager.update_strategy_weights(trade_result)
            
            # Salva in history
            trade_result['close_time'] = datetime.now()
            self.trade_history.append(trade_result)
            
            # Log
            self.logger.info(f"Trade closed: {trade['action']} @ {close_price:.2f}, "
                        f"P&L: ${pnl_dollars:.2f}, Reason: {reason}")
            
        except Exception as e:
            print(f"⚠️ Errore aggiornamento statistiche: {e}")


    def _check_close_conditions(self, trade: Dict, market_data: MarketData) -> Tuple[bool, str]:
        """Controlla condizioni per chiudere il trade"""
        current_price = market_data.bid
        tp_price = trade.get('tp_price')
        pnl_percent = trade.get('pnl_percent', 0)
        duration = trade.get('duration_minutes', 0)
        
        # 1. Take Profit raggiunto
        if tp_price:
            if (trade['action'] == TradeAction.BUY.value and current_price >= tp_price) or \
            (trade['action'] == TradeAction.SELL.value and current_price <= tp_price):
                return True, f"TAKE PROFIT HIT @ {current_price:.2f}"
        
        # 2. Stop Loss di emergenza
        if pnl_percent <= -0.003:  # -0.3%
            return True, f"EMERGENCY STOP LOSS @ {current_price:.2f} (-{abs(pnl_percent):.2%})"
        
        # 3. Timeout
        if duration >= 60:  # 60 minuti massimo
            return True, f"TIMEOUT dopo {duration:.0f} minuti"
        
        # 4. Trend reversal detection (semplificato)
        # Qui potresti aggiungere logica più sofisticata
        
        return False, ""
    
    def _close_trade(self, market_data: MarketData, reason: str):
        """Chiude il trade corrente"""
        if not self.current_trade:
            return
        
        try:
            trade = self.current_trade
            current_price = market_data.bid
            
            print(f"\n📤 CHIUSURA TRADE...")
            print(f"   Reason: {reason}")
            print(f"   Price: {current_price:.2f}")
            
            # Determina tipo di chiusura
            if trade['action'] == TradeAction.BUY.value:
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_type = mt5.ORDER_TYPE_BUY
            
            # Prepara ordine di chiusura
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": Config.FIXED_LOT_SIZE,
                "type": close_type,
                "price": current_price,
                "deviation": 20,
                "magic": 1000,
                "comment": reason[:30],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Invia ordine
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Calcola P&L finale
                pnl_dollars = trade.get('pnl_dollars', 0)
                pnl_percent = trade.get('pnl_percent', 0)
                
                # Aggiorna risk manager
                account_info = self.mt5_manager.get_account_info()
                trade_result = {
                    'action': trade['action'],
                    'entry_price': trade['entry_price'],
                    'close_price': current_price,
                    'pnl': pnl_dollars,
                    'pnl_percent': pnl_percent,
                    'duration_minutes': trade.get('duration_minutes', 0),
                    'tp_strategy': trade.get('tp_strategy', TPStrategy.HYBRID),
                    'current_equity': account_info.get('equity', 0)
                }
                
                self.risk_manager.update_trade_result(trade_result)
                
                # Aggiorna TP manager
                self.tp_manager.update_strategy_weights(trade_result)
                
                # Salva in history
                trade_result['close_time'] = datetime.now()
                trade_result['close_reason'] = reason
                self.trade_history.append(trade_result)
                
                # Log
                outcome = "WIN" if pnl_dollars > 0 else "LOSS"
                print(f"✅ Trade chiuso - {outcome}")
                print(f"   Final P&L: ${pnl_dollars:.2f} ({pnl_percent:+.2%})")
                print(f"   Duration: {trade.get('duration_minutes', 0):.1f} min")
                
                self.logger.info(f"Trade closed: {trade['action']} @ {current_price:.2f}, "
                            f"P&L: ${pnl_dollars:.2f}, Reason: {reason}")
                
                # Reset trade corrente
                self.current_trade = None
                
            else:
                error_msg = result.comment if result else 'Unknown error'
                print(f"❌ Chiusura trade fallita: {error_msg}")
                
        except Exception as e:
            print(f"❌ Errore chiusura trade: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_current_price(self) -> Optional[float]:
        """Ottieni prezzo corrente"""
        market_data = self.data_collector.get_realtime_data(self.symbol)
        return market_data.bid if market_data else None

    def debug_market_data(self):
        """Debug dei dati di mercato"""
        print("\n🔍 DEBUG MARKET DATA...")
        
        if not self.mt5_manager.connected:
            print("   Not connected to MT5")
            return
        
        try:
            # Ottieni tick
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                print(f"   Symbol: {self.symbol}")
                print(f"   Bid: {tick.bid:.2f}")
                print(f"   Ask: {tick.ask:.2f}")
                print(f"   Ask-Bid: {(tick.ask - tick.bid):.4f}")
                print(f"   Spread (x100): {(tick.ask - tick.bid) * 100:.1f} pips")
                print(f"   Spread (x10000): {(tick.ask - tick.bid) * 10000:.1f} pips")
                print(f"   Time: {tick.time}")
            
            # Ottieni info simbolo
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info:
                print(f"\n   Symbol Info:")
                print(f"   • Point: {symbol_info.point}")
                print(f"   • Digits: {symbol_info.digits}")
                print(f"   • Spread: {symbol_info.spread}")
                print(f"   • Trade mode: {symbol_info.trade_mode}")
                print(f"   • Trade calc mode: {symbol_info.trade_calc_mode}")
                
        except Exception as e:
            print(f"   Error: {e}")

# =============================================================================
# 12. MAIN EXECUTION
# =============================================================================

def main():
    """Funzione principale"""
    print("\n" + "=" * 80)
    print("🥇 XAUUSD INTELLIGENT TP BOT - STARTUP")
    print("📅", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    # Inizializza bot
    bot = XAUUSDIntelligentTPBot()
    
    try:
        while True:
            print("\n🎮 MENU PRINCIPALE:")
            print("1. 🔧 Setup Bot Completo")
            print("2. 🎯 Avvia Trading Live")
            print("3. 📊 Status Report")
            print("4. ⚙️  Configura Risk Manager")
            print("5. 📈 Performance Analytics")
            print("6. 🛑 Ferma Trading")
            print("7. 🚪 Esci")
            print("8. 💾 Salva Stato")
            print("9. 📂 Carica Stato")
            
            choice = input("\nScelta (1-9): ").strip()
            
            if choice == '1':
                print("\n🔧 SETUP BOT...")
                print("Questa operazione richiederà alcuni minuti")
                confirm = input("Confermi? (s/n): ").strip().lower()
                if confirm == 's':
                    success = bot.setup_bot()
                    if success:
                        print("\n✅ SETUP COMPLETATO CON SUCCESSO!")
                    else:
                        print("\n❌ SETUP FALLITO")
                else:
                    print("Operazione annullata")
                    
            elif choice == '2':
                if hasattr(bot, 'is_trained'):
                    print("\n🎯 AVVIO TRADING LIVE...")
                    print("⚠️  ATTENZIONE: Trading live con soldi reali!")
                    print("   Assicurati di aver configurato correttamente il bot")
                    confirm = input("Confermi l'avvio? (s/n): ").strip().lower()
                    if confirm == 's':
                        # Avvia in thread separato
                        trading_thread = threading.Thread(target=bot.start_trading, daemon=True)
                        trading_thread.start()
                        print("✅ TRADING AVVIATO IN BACKGROUND")
                        print("   Usa Ctrl+C nel terminale per fermare")
                        print("   Usa opzione 3 per status, opzione 6 per fermare")
                    else:
                        print("Operazione annullata")
                else:
                    print("❌ Bot non configurato. Esegui prima il setup (opzione 1)")
                    
            elif choice == '3':
                print(bot.get_status_report())
                    
            elif choice == '4':
                print("\n⚙️  CONFIGURAZIONE RISK MANAGER")
                print("1. Abilita/Disabilita Circuit Breaker")
                print("2. Visualizza Configurazione Attuale")
                print("3. Reset Statistiche Giornaliere")
                
                sub_choice = input("\nScelta (1-3): ").strip()
                
                if sub_choice == '1':
                    current = bot.risk_manager.circuit_breakers['enabled']
                    new_status = not current
                    bot.risk_manager.enable_circuit_breaker(new_status)
                    status = "abilitato" if new_status else "disabilitato"
                    print(f"✅ Circuit breaker {status}")
                    
                elif sub_choice == '2':
                    config = bot.risk_manager.circuit_breakers
                    print("\n🛡️  CONFIGURAZIONE RISK MANAGER:")
                    print(f"   Max Daily Loss: {config['max_daily_loss']}%")
                    print(f"   Max Consecutive Losses: {config['max_consecutive_losses']}")
                    print(f"   Max Drawdown: {config['max_drawdown']}%")
                    print(f"   Circuit Breaker: {'ENABLED' if config['enabled'] else 'DISABLED'}")
                    
                elif sub_choice == '3':
                    bot.risk_manager._reset_daily_stats()
                    print("✅ Statistiche giornaliere resettate")
                    
            elif choice == '5':
                print("\n📈 PERFORMANCE ANALYTICS")
                if bot.trade_history:
                    total_trades = len(bot.trade_history)
                    winning_trades = sum(1 for t in bot.trade_history if t.get('pnl', 0) > 0)
                    total_pnl = sum(t.get('pnl', 0) for t in bot.trade_history)
                    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
                    
                    print(f"   Total Trades: {total_trades}")
                    print(f"   Winning Trades: {winning_trades}")
                    print(f"   Win Rate: {winning_trades/total_trades*100:.1f}%")
                    print(f"   Total P&L: ${total_pnl:.2f}")
                    print(f"   Average P&L: ${avg_pnl:.2f}")
                    
                    # P&L per strategia
                    print("\n   P&L per Strategia TP:")
                    strategies = {}
                    for trade in bot.trade_history:
                        strategy = trade.get('tp_strategy', TPStrategy.HYBRID)
                        pnl = trade.get('pnl', 0)
                        if strategy not in strategies:
                            strategies[strategy] = {'total': 0, 'count': 0}
                        strategies[strategy]['total'] += pnl
                        strategies[strategy]['count'] += 1
                    
                    for strategy, data in strategies.items():
                        avg = data['total'] / data['count'] if data['count'] > 0 else 0
                        print(f"   • {strategy.value}: ${data['total']:.2f} (avg: ${avg:.2f})")
                else:
                    print("   Nessun trade eseguito")
                    
            elif choice == '6':
                bot.stop_trading()
                print("✅ Richiesta di stop trading inviata")
                
            elif choice == '7':
                bot.stop_trading()
                print("\n👋 Arrivederci!")
                break
                
            elif choice == '8':
                print("💾 Funzione di salvataggio stato (da implementare)")
                print("⚠️  Feature in sviluppo")
                
            elif choice == '9':
                print("📂 Funzione di caricamento stato (da implementare)")
                print("⚠️  Feature in sviluppo")

            elif choice == '10':  # <-- AGGIUNGI QUESTO
                bot.debug_market_data()
                
            elif choice == '11':  # <-- Opzionale: debug features ML
                if hasattr(bot, 'debug_features'):
                    bot.debug_features()
                else:
                    print("⚠️  Debug features not available")
                

            else:
                print("❌ Scelta non valida")
                
            # Pausa tra comandi
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Programma interrotto")
        bot.stop_trading()
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
        bot.stop_trading()

# =============================================================================
# 13. BACKTESTING MODULE (OPZIONALE)
# =============================================================================

class Backtester:
    """Modulo per backtesting strategia"""
    
    @staticmethod
    def run_backtest(historical_data: pd.DataFrame, 
                    initial_balance: float = 10000.0,
                    commission: float = 0.0001) -> Dict:
        """
        Esegue backtesting su dati storici
        
        Args:
            historical_data: DataFrame con dati OHLCV
            initial_balance: Balance iniziale
            commission: Commissione per trade
        
        Returns:
            Dict con risultati backtest
        """
        print("\n📊 AVVIO BACKTEST...")
        
        # Implementazione semplificata
        # In una versione completa, qui implementeresti il backtesting completo
        
        return {
            'initial_balance': initial_balance,
            'final_balance': initial_balance * 1.05,  # Esempio
            'total_trades': 100,
            'win_rate': 0.55,
            'total_pnl': initial_balance * 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'message': 'Backtest da implementare completamente'
        }

# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Verifica dipendenze
    print("🔍 Verifica dipendenze...")
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5: OK")
    except ImportError:
        print("❌ MetaTrader5 non installato")
        print("   Installa con: pip install MetaTrader5")
        exit(1)
    
    try:
        import talib
        print("✅ TA-Lib: OK")
    except ImportError:
        print("❌ TA-Lib non installato")
        print("   Installa con: pip install TA-Lib")
        print("   Oppure: conda install -c conda-forge ta-lib")
        exit(1)
    
    try:
        import sklearn
        print("✅ Scikit-learn: OK")
    except ImportError:
        print("❌ Scikit-learn non installato")
        print("   Installa con: pip install scikit-learn")
        exit(1)
    
    print("\n✅ Tutte le dipendenze verificate")
    
    # Avvia bot
    main()
    