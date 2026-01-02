# advanced_analysis/correlation.py
"""相関分析モジュール

自律神経活動と主観評価の関連性を分析する。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CorrelationResult:
    """相関分析結果"""
    r: float  # 相関係数
    p_value: float
    n: int
    ci_lower: float  # 95%信頼区間下限
    ci_upper: float  # 95%信頼区間上限
    r_squared: float  # 決定係数
    is_significant: bool

    def to_dict(self) -> Dict:
        return {
            '相関係数 (r)': self.r,
            'p値': self.p_value,
            'n': self.n,
            '95%CI下限': self.ci_lower,
            '95%CI上限': self.ci_upper,
            'R²': self.r_squared,
            '有意 (p<0.05)': self.is_significant,
        }


@dataclass
class RegressionResult:
    """回帰分析結果"""
    slope: float  # 傾き
    intercept: float  # 切片
    r_squared: float  # 決定係数
    adjusted_r_squared: float  # 自由度調整済みR²
    f_statistic: float
    p_value: float
    se_slope: float  # 傾きの標準誤差
    coefficients: Dict[str, float]  # 重回帰の場合

    def to_dict(self) -> Dict:
        return {
            '傾き': self.slope,
            '切片': self.intercept,
            'R²': self.r_squared,
            '調整済みR²': self.adjusted_r_squared,
            'F統計量': self.f_statistic,
            'p値': self.p_value,
            '傾きSE': self.se_slope,
        }


class CorrelationAnalyzer:
    """相関分析クラス"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def pearson_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> CorrelationResult:
        """Pearson相関係数

        Args:
            x: 変数1
            y: 変数2

        Returns:
            CorrelationResult: 相関分析結果
        """
        x = np.array(x)
        y = np.array(y)

        # 欠損値を除外
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        n = len(x)
        if n < 3:
            raise ValueError("データ数が不足しています（最低3点必要）")

        r, p_value = stats.pearsonr(x, y)

        # Fisher's z変換で信頼区間計算
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)

        return CorrelationResult(
            r=r,
            p_value=p_value,
            n=n,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            r_squared=r ** 2,
            is_significant=p_value < self.alpha
        )

    def spearman_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> CorrelationResult:
        """Spearman順位相関係数

        Args:
            x: 変数1
            y: 変数2

        Returns:
            CorrelationResult: 相関分析結果
        """
        x = np.array(x)
        y = np.array(y)

        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        n = len(x)
        if n < 3:
            raise ValueError("データ数が不足しています")

        r, p_value = stats.spearmanr(x, y)

        # 近似信頼区間
        se = 1 / np.sqrt(n - 3)
        z = np.arctanh(r)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)

        return CorrelationResult(
            r=r,
            p_value=p_value,
            n=n,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            r_squared=r ** 2,
            is_significant=p_value < self.alpha
        )

    def partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> CorrelationResult:
        """偏相関係数（zの影響を制御）

        Args:
            x: 変数1
            y: 変数2
            z: 制御変数

        Returns:
            CorrelationResult: 相関分析結果
        """
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        n = len(x)
        if n < 4:
            raise ValueError("データ数が不足しています")

        # 残差を計算
        slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
        slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)

        residual_x = x - (slope_xz * z + intercept_xz)
        residual_y = y - (slope_yz * z + intercept_yz)

        # 残差間の相関
        r, p_value = stats.pearsonr(residual_x, residual_y)

        # 信頼区間
        df = n - 3
        se = 1 / np.sqrt(df)
        z_val = np.arctanh(r)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z_val - z_crit * se)
        ci_upper = np.tanh(z_val + z_crit * se)

        return CorrelationResult(
            r=r,
            p_value=p_value,
            n=n,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            r_squared=r ** 2,
            is_significant=p_value < self.alpha
        )

    def correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """相関行列とp値行列を計算

        Args:
            data: データフレーム
            method: 'pearson' or 'spearman'

        Returns:
            (相関行列, p値行列)
        """
        columns = data.columns
        n = len(columns)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    x = data.iloc[:, i].values
                    y = data.iloc[:, j].values

                    mask = ~(np.isnan(x) | np.isnan(y))
                    if np.sum(mask) < 3:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan
                    else:
                        if method == 'pearson':
                            r, p = stats.pearsonr(x[mask], y[mask])
                        else:
                            r, p = stats.spearmanr(x[mask], y[mask])
                        corr_matrix[i, j] = r
                        p_matrix[i, j] = p

        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)

        return corr_df, p_df

    def simple_regression(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> RegressionResult:
        """単回帰分析

        Args:
            x: 説明変数
            y: 目的変数

        Returns:
            RegressionResult: 回帰分析結果
        """
        x = np.array(x)
        y = np.array(y)

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        n = len(x)
        if n < 3:
            raise ValueError("データ数が不足しています")

        slope, intercept, r, p_value, se = stats.linregress(x, y)

        # R²と調整済みR²
        r_squared = r ** 2
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

        # F統計量
        ss_reg = np.sum((slope * x + intercept - np.mean(y)) ** 2)
        ss_res = np.sum((y - (slope * x + intercept)) ** 2)
        f_stat = (ss_reg / 1) / (ss_res / (n - 2)) if ss_res > 0 else 0

        return RegressionResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            f_statistic=f_stat,
            p_value=p_value,
            se_slope=se,
            coefficients={'slope': slope, 'intercept': intercept}
        )

    def multiple_regression(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> RegressionResult:
        """重回帰分析

        Args:
            X: 説明変数（複数列）
            y: 目的変数

        Returns:
            RegressionResult: 回帰分析結果
        """
        from numpy.linalg import lstsq, inv

        y = np.array(y).flatten()

        # 欠損値除去
        mask = ~np.isnan(y)
        for col in X.columns:
            mask &= ~np.isnan(X[col].values)

        X_clean = X[mask].values
        y_clean = y[mask]

        n, k = X_clean.shape
        if n < k + 2:
            raise ValueError("データ数が説明変数の数に対して不足しています")

        # 定数項を追加
        X_with_const = np.column_stack([np.ones(n), X_clean])

        # 最小二乗法
        coeffs, residuals, rank, s = lstsq(X_with_const, y_clean, rcond=None)

        # 予測値と残差
        y_pred = X_with_const @ coeffs
        residuals = y_clean - y_pred

        # R²
        ss_total = np.sum((y_clean - np.mean(y_clean)) ** 2)
        ss_res = np.sum(residuals ** 2)
        r_squared = 1 - ss_res / ss_total if ss_total > 0 else 0

        # 調整済みR²
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # F統計量
        ss_reg = ss_total - ss_res
        f_stat = (ss_reg / k) / (ss_res / (n - k - 1)) if ss_res > 0 else 0
        p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)

        # 係数の標準誤差
        mse = ss_res / (n - k - 1)
        try:
            var_coeff = mse * inv(X_with_const.T @ X_with_const)
            se_coeffs = np.sqrt(np.diag(var_coeff))
        except Exception:
            se_coeffs = np.zeros(k + 1)

        # 係数を辞書に
        coeff_dict = {'intercept': coeffs[0]}
        for i, col in enumerate(X.columns):
            coeff_dict[col] = coeffs[i + 1]

        return RegressionResult(
            slope=coeffs[1] if k == 1 else np.nan,
            intercept=coeffs[0],
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            f_statistic=f_stat,
            p_value=p_value,
            se_slope=se_coeffs[1] if k == 1 else np.nan,
            coefficients=coeff_dict
        )

    def correlation_interpretation(self, r: float) -> str:
        """相関係数の解釈

        Args:
            r: 相関係数

        Returns:
            解釈文字列
        """
        r = abs(r)
        if r < 0.1:
            return 'ほぼ無相関'
        elif r < 0.3:
            return '弱い相関'
        elif r < 0.5:
            return '中程度の相関'
        elif r < 0.7:
            return 'やや強い相関'
        elif r < 0.9:
            return '強い相関'
        else:
            return '非常に強い相関'

    def analyze_hrv_subjective_relation(
        self,
        hrv_data: pd.DataFrame,
        subjective_data: pd.DataFrame,
        hrv_cols: List[str],
        subjective_cols: List[str],
        merge_on: str = 'Subject'
    ) -> pd.DataFrame:
        """HRV指標と主観評価の関連性を分析

        Args:
            hrv_data: HRVデータ
            subjective_data: 主観評価データ
            hrv_cols: HRV指標の列名リスト
            subjective_cols: 主観評価の列名リスト
            merge_on: 結合キー

        Returns:
            相関分析結果のDataFrame
        """
        # データ結合
        merged = pd.merge(hrv_data, subjective_data, on=merge_on, how='inner')

        results = []
        for hrv_col in hrv_cols:
            for subj_col in subjective_cols:
                if hrv_col not in merged.columns or subj_col not in merged.columns:
                    continue

                try:
                    # Pearson
                    pearson = self.pearson_correlation(
                        merged[hrv_col].values,
                        merged[subj_col].values
                    )

                    # Spearman
                    spearman = self.spearman_correlation(
                        merged[hrv_col].values,
                        merged[subj_col].values
                    )

                    results.append({
                        'HRV指標': hrv_col,
                        '主観評価': subj_col,
                        'Pearson r': pearson.r,
                        'Pearson p': pearson.p_value,
                        'Spearman ρ': spearman.r,
                        'Spearman p': spearman.p_value,
                        'n': pearson.n,
                        '解釈': self.correlation_interpretation(pearson.r),
                    })
                except Exception as e:
                    print(f"Warning: {hrv_col} vs {subj_col} の分析に失敗: {e}")

        return pd.DataFrame(results)
