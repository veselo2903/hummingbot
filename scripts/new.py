# -*- coding: utf-8 -*-

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.core.data_type.common import OrderType
from decimal import Decimal
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Tuple

# =================================================================================================
# ||                                                                                             ||
# ||          СКРИПТ МНОГОТАЙМФРЕЙМОВОЙ СТРАТЕГИИ ДЛЯ HUMMINGBOT (PAPER TRADING)                 ||
# ||                                                                                             ||
# =================================================================================================
# || ОПИСАНИЕ:                                                                                   ||
# || Этот скрипт реализует long-only стратегию, оптимизированную для тестирования                ||
# || на бумажном счете (paper trading). Решения о входе принимаются на основе взвешенной        ||
# || системы баллов, которая анализирует технические индикаторы на трех таймфреймах.             ||
# ||                                                                                             ||
# || ВАЖНО ДЛЯ PAPER TRADING:                                                                    ||
# || Для работы 'paper_trade_exchange' необходимо настроить коннектор к реальной бирже           ||
# || в файле conf/conf_global.yml, который будет служить источником цен.                         ||
# || Например: paper_trade_exchange_market: mexc_spot                                            ||
# =================================================================================================


class MultiTFLongPaperTrade(ScriptStrategyBase):
    """
    Класс стратегии, реализующий всю торговую логику для бумажной торговли.
    """

    # --- НАСТРОЙКИ СТРАТЕГИИ ---
    # Биржа и торговая пара
    # Установлен paper_trade_exchange для безопасного тестирования
    exchange = "paper_trade_exchange"
    trading_pair = "BTC-USDT"
    
    # Настройки таймфреймов
    primary_interval = "4h"
    secondary_intervals = ["1h", "1d"]
    
    # Настройки индикаторов
    ema_periods = [20, 50, 200]
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    volume_sma_period = 20

    # Настройки управления позицией
    order_amount_pct = Decimal("0.70")  # 70% от баланса quote_asset
    tp_strong = Decimal("0.017")       # 1.7% для сильного сигнала
    tp_medium = Decimal("0.012")       # 1.2% для среднего сигнала
    tp_base = Decimal("0.007")         # 0.7% для базового сигнала
    min_profit_for_signal_exit = Decimal("0.05") # 5% мин. прибыль для выхода по шорт-сигналу

    # Настройки защиты от переторговли
    enable_cooldown = True
    cooldown_period = 3  # Минимальное количество баров (4H) между сделками

    # --- ВНУТРЕННИЕ ПЕРЕМЕННЫЕ СТРАТЕГИИ ---
    candles = { "4h": None, "1h": None, "1d": None }
    in_position = False
    entry_price = Decimal("0")
    current_signal_strength = None
    take_profit_price = Decimal("0")
    last_trade_close_timestamp = 0

    # Автоматическое определение базового и квотируемого актива
    @property
    def base_asset(self) -> str:
        return self.trading_pair.split("-")[0]

    @property
    def quote_asset(self) -> str:
        return self.trading_pair.split("-")[1]

    @property
    def all_markets(self) -> List[str]:
        """
        Возвращает список всех рынков, необходимых для работы стратегии.
        """
        return [f"{self.exchange}_{self.trading_pair}"]

    def on_tick(self):
        """
        Основной метод, который вызывается на каждом тике (обновлении) рынка.
        """
        if not self.update_all_candles():
            self.logger().info("Данные по свечам еще не готовы. Ожидание...")
            return

        current_price = self.connectors[self.exchange].get_price(self.trading_pair, True)
        if current_price <= 0:
            return

        if self.in_position:
            self.manage_open_position(current_price)
        else:
            self.check_entry_conditions(current_price)

    def update_all_candles(self) -> bool:
        """
        Обновляет и обрабатывает данные по свечам для всех необходимых таймфреймов.
        """
        all_intervals = [self.primary_interval] + self.secondary_intervals
        for interval in all_intervals:
            try:
                candles_df = self.get_candles_df(f"{self.exchange}_{self.trading_pair}", interval, limit=300)
                self.candles[interval] = self.calculate_indicators(candles_df.copy())
            except Exception as e:
                self.logger().error(f"Ошибка при получении или обработке свечей для {interval}: {e}")
                return False
        return all(df is not None and not df.empty for df in self.candles.values())

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает все необходимые технические индикаторы для заданного DataFrame.
        """
        for period in self.ema_periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        macd = ta.macd(df["close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df["macd"] = macd[f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"]
        df["macdh"] = macd[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"]
        df["macds"] = macd[f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"]
        df["volume_sma"] = ta.sma(df["volume"], length=self.volume_sma_period)
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["is_bullish_candle"] = df["close"] > df["open"]
        df["is_bearish_candle"] = df["close"] < df["open"]
        return df.dropna()

    def calculate_long_points(self) -> float:
        """
        Рассчитывает баллы для входа в LONG позицию на основе текущих данных.
        """
        points = 0.0
        df4h = self.candles["4h"].iloc[-1]
        df1d = self.candles["1d"].iloc[-1]
        df1h = self.candles["1h"].iloc[-1]
        
        # EMA сигналы
        if df4h["ema_20"] > df4h["ema_50"] and self.candles["4h"]["ema_20"].iloc[-2] <= self.candles["4h"]["ema_50"].iloc[-2]:
            points += 2.0
        elif df4h["ema_20"] > df4h["ema_50"]:
            points += 1.0

        # MACD сигналы
        if df4h["macd"] > df4h["macds"] and self.candles["4h"]["macd"].iloc[-2] <= self.candles["4h"]["macds"].iloc[-2] and df4h["macd"] < 0:
            points += 1.0
        elif df4h["macd"] > df4h["macds"]:
            points += 0.5
            
        # RSI сигналы
        if df4h["rsi"] < 35:
            points += 1.0
        elif 40 < df4h["rsi"] < 70:
            points += 0.5

        # Volume/OBV сигналы
        volume_spike = df4h["volume"] > (df4h["volume_sma"] * 1.2)
        obv_growth = df4h["obv"] > self.candles["4h"]["obv"].iloc[-2]
        if volume_spike and df4h["is_bullish_candle"] and obv_growth:
            points += 1.0
        elif (volume_spike and df4h["is_bullish_candle"]) or obv_growth:
            points += 0.5

        # Фильтры
        if df1d["close"] > df1d["ema_50"] and df1d["rsi"] > 45:
            points += 0.5
        if df1h["ema_20"] > df1h["ema_50"] and df1h["rsi"] > 35:
            points += 0.5
            
        return points

    def calculate_short_points(self) -> float:
        """
        Рассчитывает баллы для выхода из позиции (шорт-сигнал).
        """
        points = 0.0
        df4h = self.candles["4h"].iloc[-1]
        df1d = self.candles["1d"].iloc[-1]
        df1h = self.candles["1h"].iloc[-1]
        
        # EMA сигналы
        if df4h["ema_20"] < df4h["ema_50"] and self.candles["4h"]["ema_20"].iloc[-2] >= self.candles["4h"]["ema_50"].iloc[-2]:
            points += 2.0
        elif df4h["ema_20"] < df4h["ema_50"]:
            points += 1.0

        # MACD сигналы
        if df4h["macd"] < df4h["macds"] and self.candles["4h"]["macd"].iloc[-2] >= self.candles["4h"]["macds"].iloc[-2] and df4h["macd"] > 0:
            points += 1.0
        elif df4h["macd"] < df4h["macds"]:
            points += 0.5
            
        # RSI сигналы
        if df4h["rsi"] > 65:
            points += 1.0
        elif 30 < df4h["rsi"] < 60:
            points += 0.5

        # Volume/OBV сигналы
        volume_spike = df4h["volume"] > (df4h["volume_sma"] * 1.2)
        obv_fall = df4h["obv"] < self.candles["4h"]["obv"].iloc[-2]
        if volume_spike and df4h["is_bearish_candle"] and obv_fall:
            points += 1.0
        elif (volume_spike and df4h["is_bearish_candle"]) or obv_fall:
            points += 0.5

        # Фильтры
        if df1d["close"] < df1d["ema_50"] and df1d["rsi"] < 55:
            points += 0.5
        if df1h["ema_20"] < df1h["ema_50"] and df1h["rsi"] < 65:
            points += 0.5
            
        return points

    def get_signal_strength(self, long_points: float) -> Tuple[str, Decimal]:
        """
        Определяет силу сигнала и соответствующую цель Take Profit.
        """
        if long_points >= 4.0:
            return "Сильный", self.tp_strong
        elif 3.0 <= long_points < 4.0:
            return "Средний", self.tp_medium
        elif 2.5 <= long_points < 3.0:
            return "Базовый", self.tp_base
        else:
            return None, None

    def manage_open_position(self, current_price: Decimal):
        """
        Логика управления уже открытой позицией.
        """
        if current_price >= self.take_profit_price:
            self.logger().info(f"Цель Take Profit ({self.take_profit_price}) достигнута. Продажа.")
            self.sell_and_reset_position()
            return

        current_profit = (current_price - self.entry_price) / self.entry_price
        if current_profit >= self.min_profit_for_signal_exit:
            short_points = self.calculate_short_points()
            if short_points >= 2.5:
                self.logger().info(f"Обнаружен шорт-сигнал ({short_points:.2f} баллов) с достаточной прибылью ({current_profit:.2%}). Продажа.")
                self.sell_and_reset_position()
                return

    def check_entry_conditions(self, current_price: Decimal):
        """
        Проверяет условия для открытия новой LONG позиции.
        """
        if self.enable_cooldown:
            last_candle_ts = self.candles[self.primary_interval].iloc[-1]["timestamp"]
            if last_candle_ts <= self.last_trade_close_timestamp + (self.cooldown_period * 4 * 60 * 60):
                return

        if current_price < self.candles["4h"].iloc[-1]["ema_200"]:
            return
            
        long_points = self.calculate_long_points()
        signal_strength, tp_level = self.get_signal_strength(long_points)

        if signal_strength:
            self.logger().info(f"Обнаружен LONG сигнал! Тип: {signal_strength} ({long_points:.2f} баллов). Вход в позицию.")
            self.buy_and_set_position(current_price, signal_strength, tp_level)

    def buy_and_set_position(self, price: Decimal, signal: str, tp_level: Decimal):
        """
        Размещает ордер на покупку и устанавливает внутренние переменные состояния.
        """
        quote_balance = self.connectors[self.exchange].get_available_balance(self.quote_asset)
        if quote_balance <= 10:
            self.logger().warning(f"Недостаточно {self.quote_asset} для открытия позиции.")
            return
            
        amount_to_buy = (quote_balance * self.order_amount_pct) / price
        
        self.buy(connector_name=self.exchange, trading_pair=self.trading_pair, amount=amount_to_buy, order_type=OrderType.MARKET)
        
        self.in_position = True
        self.entry_price = price
        self.current_signal_strength = signal
        self.take_profit_price = price * (1 + tp_level)
        self.logger().info(f"Позиция открыта. Вход: {self.entry_price}, TP: {self.take_profit_price}, Сигнал: {self.current_signal_strength}")

    def sell_and_reset_position(self):
        """
        Размещает ордер на продажу и сбрасывает состояние позиции.
        """
        base_balance = self.connectors[self.exchange].get_available_balance(self.base_asset)
        if base_balance > 0:
            self.sell(connector_name=self.exchange, trading_pair=self.trading_pair, amount=base_balance, order_type=OrderType.MARKET)
        
        self.in_position = False
        self.entry_price = Decimal("0")
        self.current_signal_strength = None
        self.take_profit_price = Decimal("0")
        self.last_trade_close_timestamp = self.candles[self.primary_interval].iloc[-1]["timestamp"]
        self.logger().info("Позиция закрыта. Состояние сброшено.")

    def format_status(self) -> str:
        """
        Форматирует и выводит статус стратегии в интерфейс Hummingbot.
        """
        if not self.ready_to_trade:
            return "Стратегия не готова. Проверьте подключение к бирже."

        lines = []
        lines.append("--- БАЛАНСЫ ---")
        for asset in [self.base_asset, self.quote_asset]:
            balance = self.connectors[self.exchange].get_balance(asset)
            lines.append(f"  {asset}: {balance:.8f}")
        lines.append("-" * 15)

        if self.in_position:
            lines.append("--- ПОЗИЦИЯ: ОТКРЫТА ---")
            current_price = self.connectors[self.exchange].get_price(self.trading_pair, True)
            pnl = ((current_price - self.entry_price) / self.entry_price) * 100 if self.entry_price > 0 else 0
            lines.append(f"  Сигнал входа: {self.current_signal_strength}")
            lines.append(f"  Цена входа: {self.entry_price:.4f}")
            lines.append(f"  Текущая цена: {current_price:.4f}")
            lines.append(f"  Цель (TP): {self.take_profit_price:.4f}")
            lines.append(f"  Текущий PnL: {pnl:.2f}%")
        else:
            lines.append("--- ПОЗИЦИЯ: ЗАКРЫТА ---")
            lines.append("  Ожидание сигнала для входа...")
        lines.append("-" * 15)
        
        try:
            long_points = self.calculate_long_points()
            short_points = self.calculate_short_points()
            lines.append("--- АНАЛИЗ РЫНКА ---")
            lines.append(f"  Баллы для LONG: {long_points:.2f}")
            lines.append(f"  Баллы для SHORT: {short_points:.2f}")
            df4h = self.candles["4h"].iloc[-1]
            lines.append(f"  Цена {self.base_asset} (4H): {df4h['close']:.2f} | RSI(14): {df4h['rsi']:.2f}")
            lines.append(f"  EMA(200) 4H: {df4h['ema_200']:.2f}")
        except Exception:
            lines.append("--- АНАЛИЗ РЫНКА: ожидание данных ---")

        return "\n".join(lines)

