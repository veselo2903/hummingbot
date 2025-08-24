import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple

import pandas as pd

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

class OneOne(ScriptStrategyBase):

    markets = {"paper_trade": {"BTC-USDT"}}
    trading_pair = "BTC-USDT"

    def on_start(self):
    connector = self.connectors.get(self.exchange)
    if connector is not None:
        available_pairs = connector.trading_pairs
        self.logger().info(f"Доступные пары {self.exchange}: {available_pairs}")
    else:
        self.logger().warning(f"Коннектор {self.exchange} не найден!")

class MexcMultiTfBtc(ScriptStrategyBase):
    """
    Многотаймфреймовая стратегия для BTC/USDT на бирже MEXC.
    """
    markets = {"paper_trade": {"BTC-USDT"}}

    # Встроенные параметры конфигурации
    exchange = "paper_trade"
    trading_pair = "BTC-USDT"
    order_amount_pct = Decimal("70.0")  # 70% от баланса USDT
    tp_strong = Decimal("1.7")  # Take profit для сильного сигнала (%)
    tp_medium = Decimal("1.2")  # Take profit для среднего сигнала (%)
    tp_base = Decimal("0.7")  # Take profit для базового сигнала (%)
    min_profit_for_signal_exit = Decimal("5.0")  # Минимальная прибыль для выхода по шорт-сигналу (%)
    min_bars_between_trades = 3  # Минимальное количество 4H баров между сделками
    ema_short_len = 20
    ema_medium_len = 50
    ema_long_len = 200
    rsi_len = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    volume_ma_len = 20

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        """
        Инициализация стратегии без внешнего конфига.
        """
        super().__init__(connectors)
        self.market_info = self._market_trading_pair_tuple(self.exchange, self.trading_pair)

        self.in_position = False
        self.entry_price = Decimal("0")
        self.position_amount = Decimal("0")
        self.take_profit_price = Decimal("0")
        self.last_trade_bar = -1
        self.active_order_id = None

        self._tp_pct_to_set: Optional[Decimal] = None
        self._last_bar_time_to_set: Optional[pd.Timestamp] = None
        self.last_trade_time: Optional[pd.Timestamp] = None

        self.logger().setLevel(logging.INFO)
        self.logger().info(f"Стратегия инициализирована: exchange={self.exchange}, trading_pair={self.trading_pair}")

    def _has_recent_nans(self, df: pd.DataFrame, cols, lookback: int = 3) -> bool:
        cols = [c for c in cols if c in df.columns]
        return df[cols].tail(lookback).isnull().any().any()


def on_tick(self):
    if not self.ready_to_trade:
        return
    if "BTC-USDT" not in self.connectors["mexc"].trading_pairs:
        self.logger().error(f"Пара BTC-USDT не распознана MEXC. Доступные пары: {self.connectors['mexc'].trading_pairs}")
        return

        if self.active_order_id:
            return

        candles_4h = self.get_candles(interval="4h", limit=500)
        candles_1d = self.get_candles(interval="1d", limit=200)
        candles_1h = self.get_candles(interval="1h", limit=100)

        min_4h_candles = self.ema_long_len + 5
        if (candles_4h is None or len(candles_4h) < min_4h_candles or
                candles_1d is None or len(candles_1d) < self.ema_medium_len + 5 or
                candles_1h is None or len(candles_1h) < self.ema_medium_len + 5):
            self.logger().warning("Недостаточно данных свечей для анализа. Пропускаем тик.")
            return

        self.add_indicators(candles_4h)
        self.add_indicators(candles_1d, is_main_tf=False)
        self.add_indicators(candles_1h, is_main_tf=False)

        # время последнего закрытого 4H-бара
        if "timestamp" in candles_4h.columns:
            last_closed_time_4h = pd.to_datetime(candles_4h.iloc[-2]["timestamp"], unit="ms", utc=True)
        else:
            last_closed_time_4h = pd.to_datetime(candles_4h.index[-2], utc=True)

        if self._has_recent_nans(candles_4h, ["open","close","volume","ema_short","ema_medium","ema_long",
                                              "volume_ma","obv","macd","macds","macdh","rsi"]) \
           or self._has_recent_nans(candles_1d, ["close","ema_medium","rsi"]) \
           or self._has_recent_nans(candles_1h, ["close","ema_short","ema_medium","rsi"]):
            self.logger().warning("Обнаружены NaN в последних барах. Пропускаем тик.")
            return

        # кулдаун в барах 4H
        min_delta = pd.Timedelta(hours=4 * self.min_bars_between_trades)
        if self.last_trade_time is not None and (last_closed_time_4h - self.last_trade_time) < min_delta:
            return

        last_candle_4h = candles_4h.iloc[-2]
        prev_candle_4h = candles_4h.iloc[-3]
        last_candle_1d = candles_1d.iloc[-2]
        last_candle_1h = candles_1h.iloc[-2]
        current_price = Decimal(str(candles_4h.iloc[-1]['close']))

        if not self.in_position:
            self.handle_entry_logic(last_candle_4h, prev_candle_4h, last_candle_1d, last_candle_1h, last_closed_time_4h)
        else:
            self.handle_exit_logic(current_price, last_candle_4h, prev_candle_4h, last_candle_1d, last_candle_1h)

    def handle_entry_logic(self, candle_4h: pd.Series, prev_candle_4h: pd.Series,
                           candle_1d: pd.Series, candle_1h: pd.Series, last_closed_time_4h: pd.Timestamp):
        long_points, _ = self.calculate_points(candle_4h, prev_candle_4h, candle_1d, candle_1h)
        signal_type = self.get_signal_type(long_points, candle_4h, is_long=True)
        if signal_type:
            tp_pct = self.get_tp_for_signal(signal_type)
            quote_balance = self.connectors[self.exchange].get_available_balance(self.market_info.quote_asset)
            order_amount_quote = quote_balance * (self.order_amount_pct / Decimal("100"))

            if order_amount_quote > Decimal("10"):
                price = self.connectors[self.exchange].get_price(self.trading_pair, is_buy=True)
                amount = (order_amount_quote / price) * Decimal("0.999")
                quantized_amount = self.market_info.market.quantize_order_amount(self.trading_pair, amount)

                if quantized_amount > Decimal("0"):
                    self.logger().info(f"Получен {signal_type} сигнал на покупку. Баллы: {long_points:.2f}. Размещаем ордер.")
                    self.place_order(is_buy=True, amount=quantized_amount, tp_pct=tp_pct,
                                     last_bar_time=last_closed_time_4h)

    def handle_exit_logic(self, current_price: Decimal, candle_4h: pd.Series, prev_candle_4h: pd.Series,
                          candle_1d: pd.Series, candle_1h: pd.Series):
        if self.take_profit_price > 0 and current_price >= self.take_profit_price:
            self.logger().info(f"Цена достигла тейк-профита ({self.take_profit_price}). Продаем.")
            self.place_order(is_buy=False, amount=self.position_amount)
            return

        current_profit_pct = (current_price - self.entry_price) / self.entry_price * Decimal("100")
        if current_profit_pct >= self.min_profit_for_signal_exit:
            _, short_points = self.calculate_points(candle_4h, prev_candle_4h, candle_1d, candle_1h)
            short_signal_type = self.get_signal_type(short_points, candle_4h, is_long=False)
            if short_signal_type:
                self.logger().info(f"Получен {short_signal_type} сигнал на продажу с прибылью {current_profit_pct:.2f}%. Продаем.")
                self.place_order(is_buy=False, amount=self.position_amount)

    def place_order(self, is_buy: bool, amount: Decimal, tp_pct: Optional[Decimal] = None, last_bar_time: Optional[pd.Timestamp] = None):
        order_type = OrderType.MARKET
        side = "BUY" if is_buy else "SELL"

        try:
            if is_buy:
                self.active_order_id = self.buy(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=amount,
                    order_type=order_type
                )
                self._tp_pct_to_set = tp_pct
                self._last_bar_time_to_set = last_bar_time
            else:
                self.active_order_id = self.sell(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=amount,
                    order_type=order_type
                )
            self.logger().info(f"Размещен {side} ордер {self.active_order_id} на {amount} {self.market_info.base_asset}.")
        except Exception as e:
            self.logger().error(f"Ошибка при размещении {side} ордера: {e}")
            self.active_order_id = None

    def did_fill_order(self, event: OrderFilledEvent):
        if event.client_order_id == self.active_order_id:
            self.logger().info(f"Ордер {event.client_order_id} исполнен: {event.trade_type.name} {event.amount} @ {event.price}")

            if event.trade_type == TradeType.BUY:
                if not self.in_position:
                    self.in_position = True
                    self.entry_price = Decimal(str(event.price))
                    self.position_amount = Decimal("0")
                self.position_amount += Decimal(str(event.amount))
                if self.take_profit_price == 0:
                    self.take_profit_price = self.entry_price * (1 + self._tp_pct_to_set / 100)
                if self._last_bar_time_to_set is not None:
                    self.last_trade_time = self._last_bar_time_to_set

            elif event.trade_type == TradeType.SELL:
                self.position_amount -= Decimal(str(event.amount))
                if self.position_amount <= Decimal("0"):
                    self.in_position = False
                    self.entry_price = Decimal("0")
                    self.position_amount = Decimal("0")
                    self.take_profit_price = Decimal("0")

            self._tp_pct_to_set = None
            self._last_bar_time_to_set = None

    def get_candles(self, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            return self.market_info.market.get_candles_df(
                trading_pair=self.trading_pair,
                interval=interval,
                max_records=limit
            )
        except Exception as e:
            self.logger().error(f"Не удалось получить свечи для {interval}: {e}")
            return None

    def rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = pd.Series(100 - (100 / (1 + rs)), index=close.index)
        return rsi

    def macd(self, close: pd.Series, fast: int, slow: int, signal: int):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, histogram, signal_line

    def obv(self, df: pd.DataFrame) -> pd.Series:
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    def add_indicators(self, df: pd.DataFrame, is_main_tf: bool = True):
        df["ema_short"] = df["close"].ewm(span=self.ema_short_len, adjust=False).mean()
        df["ema_medium"] = df["close"].ewm(span=self.ema_medium_len, adjust=False).mean()
        df["rsi"] = self.rsi(df["close"], self.rsi_len)
        if is_main_tf:
            df["ema_long"] = df["close"].ewm(span=self.ema_long_len, adjust=False).mean()
            df["volume_ma"] = df["volume"].ewm(span=self.volume_ma_len, adjust=False).mean()
            df["obv"] = self.obv(df)
            df["macd"], df["macdh"], df["macds"] = self.macd(df["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        df.dropna(inplace=True)

    def calculate_points(self, candle_4h: pd.Series, prev_candle_4h: pd.Series, candle_1d: pd.Series,
                         candle_1h: pd.Series) -> Tuple[float, float]:
        long_points = 0.0
        short_points = 0.0

        # --- ЛОНГ БАЛЛЫ ---
        if candle_4h["ema_short"] > candle_4h["ema_medium"] and prev_candle_4h["ema_short"] <= prev_candle_4h["ema_medium"]:
            long_points += 2.0
        elif candle_4h["ema_short"] > candle_4h["ema_medium"]:
            long_points += 1.0

        if (candle_4h["macd"] > candle_4h["macds"] and prev_candle_4h["macd"] <= prev_candle_4h["macds"] and
                candle_4h["macd"] < 0):
            long_points += 1.0
        elif candle_4h["macd"] > candle_4h["macds"]:
            long_points += 0.5

        if candle_4h["rsi"] < 35:
            long_points += 1.0
        elif 40 < candle_4h["rsi"] < 70:
            long_points += 0.5

        is_bullish_candle = candle_4h["close"] > candle_4h["open"]
        is_high_volume = candle_4h["volume"] > candle_4h["volume_ma"] * 1.2
        is_obv_rising = candle_4h["obv"] > prev_candle_4h["obv"]
        factor1_long = is_high_volume and is_bullish_candle
        factor2_long = is_obv_rising
        if factor1_long and factor2_long:
            long_points += 1.0
        elif factor1_long or factor2_long:
            long_points += 0.5

        if candle_1d["close"] > candle_1d["ema_medium"] and candle_1d["rsi"] > 45:
            long_points += 0.5
        if candle_1h["ema_short"] > candle_1h["ema_medium"] and candle_1h["rsi"] > 35:
            long_points += 0.5

        # --- ШОРТ БАЛЛЫ ---
        if candle_4h["ema_short"] < candle_4h["ema_medium"] and prev_candle_4h["ema_short"] >= prev_candle_4h["ema_medium"]:
            short_points += 2.0
        elif candle_4h["ema_short"] < candle_4h["ema_medium"]:
            short_points += 1.0

        if (candle_4h["macd"] < candle_4h["macds"] and prev_candle_4h["macd"] >= prev_candle_4h["macds"] and
                candle_4h["macd"] > 0):
            short_points += 1.0
        elif candle_4h["macd"] < candle_4h["macds"]:
            short_points += 0.5

        if candle_4h["rsi"] > 65:
            short_points += 1.0
        elif 30 < candle_4h["rsi"] < 60:
            short_points += 0.5

        is_bearish_candle = candle_4h["close"] < candle_4h["open"]
        is_obv_falling = candle_4h["obv"] < prev_candle_4h["obv"]
        factor1_short = is_high_volume and is_bearish_candle
        factor2_short = is_obv_falling
        if factor1_short and factor2_short:
            short_points += 1.0
        elif factor1_short or factor2_short:
            short_points += 0.5

        if candle_1d["close"] < candle_1d["ema_medium"] and candle_1d["rsi"] < 55:
            short_points += 0.5
        if candle_1h["ema_short"] < candle_1h["ema_medium"] and candle_1h["rsi"] < 65:
            short_points += 0.5

        return long_points, short_points

    def get_signal_type(self, points: float, candle_4h: pd.Series, is_long: bool) -> Optional[str]:
        if is_long and candle_4h["close"] < candle_4h["ema_long"]:
            return None
        if not is_long and candle_4h["close"] > candle_4h["ema_long"]:
            return None

        if points >= 4.0:
            return "Сильный"
        elif 3.0 <= points < 4.0:
            return "Средний"
        elif 2.5 <= points < 3.0:
            return "Базовый"
        else:
            return None

    def get_tp_for_signal(self, signal_type: str) -> Decimal:
        if signal_type == "Сильный":
            return self.tp_strong
        elif signal_type == "Средний":
            return self.tp_medium
        else:
            return self.tp_base

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Рынки не готовы."
        lines = []
        balance_df = self.get_balance_df()
        lines.extend(["", "  Балансы:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        lines.append("\n  --- Статус Стратегии ---")
        if self.in_position:
            lines.append(f"    В позиции:      ДА")
            lines.append(f"    Размер:         {self.position_amount:.6f} {self.market_info.base_asset}")
            lines.append(f"    Цена входа:     {self.entry_price:.4f} {self.market_info.quote_asset}")
            try:
                current_price = self.connectors[self.exchange].get_price(self.trading_pair, is_buy=False)
                pnl = (current_price - self.entry_price) / self.entry_price * Decimal("100")
                lines.append(f"    Текущая цена:   {current_price:.4f} {self.market_info.quote_asset}")
                lines.append(f"    P/L:            {pnl:.2f}%")
            except Exception as e:
                lines.append(f"    Не удалось получить текущую цену: {e}")
            lines.append(f"    Тейк-профит:    {self.take_profit_price:.4f} {self.market_info.quote_asset}")
        else:
            lines.append(f"    В позиции:      НЕТ")
            lines.append(f"    Ожидание сигнала на покупку...")

        try:
            candles_4h = self.get_candles(interval="4h", limit=500)
            if candles_4h is not None and len(candles_4h) > self.ema_long_len:
                candles_1d = self.get_candles(interval="1d", limit=200)
                candles_1h = self.get_candles(interval="1h", limit=100)
                self.add_indicators(candles_4h)
                self.add_indicators(candles_1d, is_main_tf=False)
                self.add_indicators(candles_1h, is_main_tf=False)
                if not (self._has_recent_nans(candles_4h, ["open","close","volume","ema_short","ema_medium","ema_long",
                                              "volume_ma","obv","macd","macds","macdh","rsi"]) \
           or self._has_recent_nans(candles_1d, ["close","ema_medium","rsi"]) \
           or self._has_recent_nans(candles_1h, ["close","ema_short","ema_medium","rsi"])):
                    last_candle_4h, prev_candle_4h = candles_4h.iloc[-2], candles_4h.iloc[-3]
                    last_candle_1d, last_candle_1h = candles_1d.iloc[-2], candles_1h.iloc[-2]
                    long_points, short_points = self.calculate_points(last_candle_4h, prev_candle_4h, last_candle_1d,
                                                                     last_candle_1h)
                    lines.append("\n  --- Текущие сигналы (на основе последней закрытой свечи) ---")
                    lines.append(f"    Лонг баллы:     {long_points:.2f}")
                    lines.append(f"    Шорт баллы:     {short_points:.2f}")
                    long_signal = self.get_signal_type(long_points, last_candle_4h, is_long=True) or "Нет"
                    short_signal = self.get_signal_type(short_points, last_candle_4h, is_long=False) or "Нет"
                    lines.append(f"    Сигнал на вход: {long_signal}")
                    lines.append(f"    Сигнал на выход:{short_signal}")
        except Exception as e:
            lines.append(f"\n  Не удалось рассчитать текущие сигналы: {e}")
        return "\n".join(lines)