# advanced_analysis/control_metrics.py
"""制御性能評価モジュール

心拍フィードバック制御の性能を定量的に評価するための指標を計算する。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class ControlMetrics:
    """制御性能指標"""
    rmse: float  # Root Mean Square Error
    mae: float  # Mean Absolute Error
    steady_state_error: float  # 定常偏差
    rise_time: Optional[float]  # 立ち上がり時間（秒）
    settling_time: Optional[float]  # 整定時間（秒）
    overshoot: Optional[float]  # オーバーシュート（%）
    control_rate: float  # 制御率（目標±5BPM以内の割合）
    convergence_rate: float  # 収束率
    stability_index: float  # 安定性指標（収束後の標準偏差）

    def to_dict(self) -> Dict:
        return {
            'RMSE (BPM)': self.rmse,
            'MAE (BPM)': self.mae,
            '定常偏差 (BPM)': self.steady_state_error,
            '立ち上がり時間 (秒)': self.rise_time,
            '整定時間 (秒)': self.settling_time,
            'オーバーシュート (%)': self.overshoot,
            '制御率 (%)': self.control_rate * 100,
            '収束率 (%)': self.convergence_rate * 100,
            '安定性指標 (BPM)': self.stability_index,
        }


class ControlMetricsAnalyzer:
    """制御性能解析クラス"""

    def __init__(
        self,
        tolerance_bpm: float = 5.0,
        settling_threshold: float = 0.05,
        rise_threshold: float = 0.9,
        steady_state_ratio: float = 0.2
    ):
        """
        Args:
            tolerance_bpm: 制御率計算時の許容誤差 (BPM)
            settling_threshold: 整定判定の閾値（目標値に対する割合）
            rise_threshold: 立ち上がり時間の閾値（90%到達）
            steady_state_ratio: 定常偏差計算に使用する終盤の割合
        """
        self.tolerance_bpm = tolerance_bpm
        self.settling_threshold = settling_threshold
        self.rise_threshold = rise_threshold
        self.steady_state_ratio = steady_state_ratio

    def calculate_metrics(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: float,
        hr_initial: Optional[float] = None
    ) -> ControlMetrics:
        """制御性能指標を計算

        Args:
            time: 時間配列（秒）
            hr_actual: 実測心拍数配列
            hr_target: 目標心拍数
            hr_initial: 初期心拍数（Noneの場合は最初の値を使用）

        Returns:
            ControlMetrics: 制御性能指標
        """
        if hr_initial is None:
            hr_initial = hr_actual[0]

        # 基本誤差指標
        error = hr_actual - hr_target
        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))

        # 定常偏差（終盤の平均誤差）
        n_steady = max(1, int(len(hr_actual) * self.steady_state_ratio))
        steady_state_error = np.mean(np.abs(error[-n_steady:]))

        # 制御率（目標±tolerance_bpm以内の割合）
        within_tolerance = np.abs(error) <= self.tolerance_bpm
        control_rate = np.mean(within_tolerance)

        # 収束率（初期値から目標への到達度）
        if hr_initial != hr_target:
            final_hr = np.mean(hr_actual[-n_steady:])
            convergence_rate = 1 - abs(final_hr - hr_target) / abs(hr_initial - hr_target)
            convergence_rate = max(0, min(1, convergence_rate))
        else:
            convergence_rate = 1.0

        # 安定性指標（収束後の標準偏差）
        stability_index = np.std(hr_actual[-n_steady:])

        # 立ち上がり時間
        rise_time = self._calculate_rise_time(time, hr_actual, hr_target, hr_initial)

        # 整定時間
        settling_time = self._calculate_settling_time(time, hr_actual, hr_target)

        # オーバーシュート
        overshoot = self._calculate_overshoot(hr_actual, hr_target, hr_initial)

        return ControlMetrics(
            rmse=rmse,
            mae=mae,
            steady_state_error=steady_state_error,
            rise_time=rise_time,
            settling_time=settling_time,
            overshoot=overshoot,
            control_rate=control_rate,
            convergence_rate=convergence_rate,
            stability_index=stability_index
        )

    def _calculate_rise_time(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: float,
        hr_initial: float
    ) -> Optional[float]:
        """立ち上がり時間を計算（目標値の90%到達時間）"""
        if hr_initial == hr_target:
            return 0.0

        threshold = hr_initial + (hr_target - hr_initial) * self.rise_threshold

        if hr_target > hr_initial:
            # 心拍数を上げる場合
            idx = np.where(hr_actual >= threshold)[0]
        else:
            # 心拍数を下げる場合
            idx = np.where(hr_actual <= threshold)[0]

        if len(idx) > 0:
            return time[idx[0]] - time[0]
        return None

    def _calculate_settling_time(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: float
    ) -> Optional[float]:
        """整定時間を計算（目標値±5%以内に収束する時間）"""
        tolerance = hr_target * self.settling_threshold
        within_band = np.abs(hr_actual - hr_target) <= tolerance

        # 最後から連続して収束している区間を探す
        settled_idx = None
        for i in range(len(within_band) - 1, -1, -1):
            if within_band[i]:
                settled_idx = i
            else:
                break

        if settled_idx is not None and settled_idx < len(within_band) - 1:
            # 最初に収束した時点を探す
            for i in range(settled_idx + 1):
                if all(within_band[i:settled_idx + 1]):
                    return time[i] - time[0]

        return None

    def _calculate_overshoot(
        self,
        hr_actual: np.ndarray,
        hr_target: float,
        hr_initial: float
    ) -> Optional[float]:
        """オーバーシュートを計算"""
        if hr_initial == hr_target:
            return 0.0

        if hr_target > hr_initial:
            # 心拍数を上げる場合
            max_hr = np.max(hr_actual)
            if max_hr > hr_target:
                return (max_hr - hr_target) / (hr_target - hr_initial) * 100
        else:
            # 心拍数を下げる場合
            min_hr = np.min(hr_actual)
            if min_hr < hr_target:
                return (hr_target - min_hr) / (hr_initial - hr_target) * 100

        return 0.0

    def calculate_tracking_error(
        self,
        time: np.ndarray,
        hr_actual: np.ndarray,
        hr_target: np.ndarray
    ) -> Dict:
        """時変目標値に対する追従誤差を計算

        Args:
            time: 時間配列
            hr_actual: 実測心拍数
            hr_target: 目標心拍数（時系列）

        Returns:
            追従誤差の統計量
        """
        error = hr_actual - hr_target

        return {
            'tracking_error_mean': np.mean(error),
            'tracking_error_std': np.std(error),
            'tracking_error_max': np.max(np.abs(error)),
            'tracking_rmse': np.sqrt(np.mean(error ** 2)),
            'cumulative_error': np.sum(np.abs(error)),
        }

    def calculate_delay(
        self,
        time: np.ndarray,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
        max_lag: int = 60
    ) -> Tuple[float, np.ndarray]:
        """相互相関による遅延時間を推定

        Args:
            time: 時間配列
            input_signal: 入力信号（例：韻律パラメータ）
            output_signal: 出力信号（例：心拍数）
            max_lag: 最大ラグ（サンプル数）

        Returns:
            (推定遅延時間, 相互相関関数)
        """
        # 正規化
        input_norm = (input_signal - np.mean(input_signal)) / (np.std(input_signal) + 1e-10)
        output_norm = (output_signal - np.mean(output_signal)) / (np.std(output_signal) + 1e-10)

        # 相互相関
        correlation = np.correlate(output_norm, input_norm, mode='full')
        lags = np.arange(-len(input_signal) + 1, len(input_signal))

        # 正のラグのみを考慮
        positive_mask = (lags >= 0) & (lags <= max_lag)
        positive_lags = lags[positive_mask]
        positive_corr = correlation[positive_mask]

        # 最大相関のラグを取得
        if len(positive_corr) > 0:
            max_idx = np.argmax(positive_corr)
            delay_samples = positive_lags[max_idx]
            dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0
            delay_time = delay_samples * dt
        else:
            delay_time = 0.0

        return delay_time, correlation

    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = 'Time',
        hr_col: str = 'HR',
        target_hr: float = 70.0
    ) -> ControlMetrics:
        """DataFrameから制御性能を解析

        Args:
            df: 解析対象のDataFrame
            time_col: 時間列名
            hr_col: 心拍数列名
            target_hr: 目標心拍数

        Returns:
            ControlMetrics: 制御性能指標
        """
        time = df[time_col].values
        hr_actual = df[hr_col].values

        return self.calculate_metrics(time, hr_actual, target_hr)

    def compare_conditions(
        self,
        condition_data: Dict[str, pd.DataFrame],
        target_hr: float,
        time_col: str = 'Time',
        hr_col: str = 'HR'
    ) -> pd.DataFrame:
        """複数条件の制御性能を比較

        Args:
            condition_data: {条件名: DataFrame} の辞書
            target_hr: 目標心拍数
            time_col: 時間列名
            hr_col: 心拍数列名

        Returns:
            条件ごとの制御性能指標をまとめたDataFrame
        """
        results = []

        for condition, df in condition_data.items():
            metrics = self.analyze_from_dataframe(df, time_col, hr_col, target_hr)
            metrics_dict = metrics.to_dict()
            metrics_dict['Condition'] = condition
            results.append(metrics_dict)

        result_df = pd.DataFrame(results)
        # Conditionを最初の列に
        cols = ['Condition'] + [c for c in result_df.columns if c != 'Condition']
        return result_df[cols]
