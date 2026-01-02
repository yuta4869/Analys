# AnalysisECG.py
"""ECG解析モジュール

ECGデータからRRIを算出し、HRV指標（LF/HF, RMSSD, SDNN）を計算する。
HCS_ver4.0の解析機能を参考にブラッシュアップ。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, find_peaks, filtfilt
from scipy import interpolate
import scipy.signal
from itertools import accumulate


# ---------------------------------------------------------
# 解析用の定数
# ---------------------------------------------------------
ECG_CONDITIONS = [
    "Fixed",
    "HRF",
    "Sin",
    "HRF2_PID",
    "HRF2_Adaptive",
    "HRF2_GS",
    "HRF2_Robust",
]

CONDITION_LABELS = {
    'Fixed': '固定会話',
    'Sin': '正弦波',
    'HRF': '調整会話',
    'HRF2_PID': 'HRF2 (PID)',
    'HRF2_Adaptive': 'HRF2 (Adaptive)',
    'HRF2_GS': 'HRF2 (GS)',
    'HRF2_Robust': 'HRF2 (Robust)',
}

CONDITION_COLORS = {
    'Fixed': 'lightcoral',
    'Sin': 'lightblue',
    'HRF': 'lightyellow',
    'HRF2_PID': '#ffcc99',
    'HRF2_Adaptive': '#b2d8b2',
    'HRF2_GS': '#c9c3ff',
    'HRF2_Robust': '#f4b6c2',
}


# ---------------------------------------------------------
# 1. 信号処理・RRI算出ロジック
# ---------------------------------------------------------

def bandpass_filter(signal_data, fs):
    """バンドパスフィルタ (0.5Hz - 50Hz)

    Args:
        signal_data: ECG信号データ
        fs: サンプリング周波数

    Returns:
        フィルタ適用後の信号

    Raises:
        ValueError: データが短すぎる場合
    """
    # 5次バターワースフィルタには最低 3 * (5 * 2 + 1) = 33 サンプル必要
    min_samples = 34
    if len(signal_data) < min_samples:
        raise ValueError(
            f"データが短すぎます: {len(signal_data)}サンプル（最低{min_samples}サンプル≒{min_samples/fs:.2f}秒必要）"
        )

    lowcut = 0.5
    highcut = 50
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal_data)


def ecg_to_rri(
    file_path,
    fs=130,
    analysis_start_offset=None,
    analysis_end_offset=None
):
    """ECGデータからRRI（RR間隔）を計算する

    Args:
        file_path: ECGデータのファイルパス
        fs: サンプリング周波数（デフォルト: 130Hz）
        analysis_start_offset: 解析開始オフセット（秒）。Noneの場合は自動判定
        analysis_end_offset: 解析終了オフセット（秒）。Noneの場合は自動判定

    Returns:
        tuple: (RRIデータ (ms単位), 解析開始時刻, 解析終了時刻) or (空の配列, None, None)
    """
    try:
        data = pd.read_csv(
            file_path,
            delimiter=',',
            encoding="utf-8",
            skiprows=1,
            header=None,
            usecols=[0, 2],
            names=['timestamp', 'ecg']
        )
    except ValueError:
        print(f"エラー: {os.path.basename(file_path)} の読み込みに失敗しました。")
        return np.array([]), None, None
    except Exception as e:
        print(f"エラー: ファイル読み込み中に問題が発生しました: {e}")
        return np.array([]), None, None

    # データが空の場合のチェック
    if data.empty or len(data) == 0:
        print(f"エラー: {os.path.basename(file_path)} にデータがありません。")
        return np.array([]), None, None

    # タイムスタンプ変換
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    start_time_real = data['timestamp'].iloc[0]
    end_time_real = data['timestamp'].iloc[-1]
    duration_seconds = (end_time_real - start_time_real).total_seconds()

    # --- 解析区間の決定 ---
    custom_window = analysis_start_offset is not None or analysis_end_offset is not None

    if custom_window:
        # カスタム区間指定
        start_offset = max(0.0, analysis_start_offset or 0.0)
        analysis_start_time = start_time_real + pd.Timedelta(seconds=start_offset)

        if analysis_end_offset is not None:
            if analysis_end_offset <= start_offset:
                print("  -> エラー: 解析終了時刻が開始時刻以下です。")
                return np.array([]), None, None
            analysis_end_time = start_time_real + pd.Timedelta(seconds=analysis_end_offset)
        else:
            analysis_end_time = end_time_real

        # 範囲チェック
        if analysis_start_time >= end_time_real:
            print("  -> エラー: 指定された解析開始時刻がデータ範囲外です。")
            return np.array([]), None, None
        if analysis_end_time > end_time_real:
            analysis_end_time = end_time_real
        if analysis_end_time <= analysis_start_time:
            print("  -> エラー: 指定された解析区間が無効です。")
            return np.array([]), None, None

        data_filtered = data[
            (data['timestamp'] >= analysis_start_time) &
            (data['timestamp'] <= analysis_end_time)
        ]
        actual_end_offset = (analysis_end_time - start_time_real).total_seconds()
        print(f"  -> 指定区間で解析: {start_offset:.1f}秒 〜 {actual_end_offset:.1f}秒")
    else:
        # 自動判定
        if duration_seconds < 60:
            # 1分未満の短いデータは全期間を使用
            print(f"  -> 短いデータを検出 ({duration_seconds:.1f}秒): 全期間を解析します")
            analysis_start_time = start_time_real
            analysis_end_time = end_time_real
            data_filtered = data
        else:
            # 十分な長さがある場合は、30秒後から5分間を使用
            print(f"  -> 長いデータを検出 ({duration_seconds:.1f}秒): 30秒後から5分間を解析します")
            analysis_start_time = start_time_real + pd.Timedelta(seconds=30)
            analysis_end_time = start_time_real + pd.Timedelta(minutes=5, seconds=30)
            data_filtered = data[
                (data['timestamp'] >= analysis_start_time) &
                (data['timestamp'] <= analysis_end_time)
            ]

    if data_filtered.empty:
        print("  -> エラー: 解析対象期間のデータが空です。")
        return np.array([]), None, None

    # ECGデータ抽出
    ecg_data = data_filtered['ecg'].values

    # バンドパスフィルタ適用
    try:
        filtered_ecg = bandpass_filter(ecg_data, fs)
    except ValueError as e:
        print(f"  -> エラー: {e}")
        return np.array([]), None, None

    # R波ピーク検出
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')

    height_threshold = np.mean(integrated_ecg) * 0.4
    distance = fs * 0.3
    peaks, _ = find_peaks(integrated_ecg, distance=distance, height=height_threshold)

    rri_data = np.diff(peaks) * 1000 / fs  # ms単位

    return rri_data, analysis_start_time, analysis_end_time


# ---------------------------------------------------------
# 2. HRV指標計算ロジック
# ---------------------------------------------------------

def calculate_hrv_indices(
    file_path,
    label=None,
    fs=130,
    output_excel_path=None,
    output_csv_path=None,
    analysis_start_offset=None,
    analysis_end_offset=None,
    resampling_freq=1.0,
    quantile_low=0.038,
    quantile_high=0.962,
    min_hr=45.0,
    max_hr=210.0,
    analysis_window_seconds=30.0,
):
    """ECGデータからHRV指標を計算する

    Args:
        file_path: ECGデータのファイルパス
        label: 解析ラベル（ログ出力用）
        fs: サンプリング周波数
        output_excel_path: スライディングウィンドウ結果を保存するExcelファイルパス
        output_csv_path: 全体結果を保存するCSVファイルパス
        analysis_start_offset: 解析開始オフセット（秒）
        analysis_end_offset: 解析終了オフセット（秒）
        resampling_freq: リサンプリング周波数
        quantile_low: 外れ値除去の下限パーセンタイル
        quantile_high: 外れ値除去の上限パーセンタイル
        min_hr: 最小心拍数制限
        max_hr: 最大心拍数制限
        analysis_window_seconds: スライディングウィンドウの秒数

    Returns:
        tuple: (時系列DataFrame, 全体LF/HF値) or (None, None)
    """
    if label is None:
        label = os.path.basename(file_path)

    print(f"--- 解析開始: {label} ({os.path.basename(file_path)}) ---")

    rri_data, start_time, end_time = ecg_to_rri(
        file_path,
        fs,
        analysis_start_offset=analysis_start_offset,
        analysis_end_offset=analysis_end_offset
    )

    if len(rri_data) == 0:
        print("  -> RRIデータが取得できなかったため、計算を中止します。")
        return None, None

    # パラメータのバリデーション
    if resampling_freq <= 0:
        resampling_freq = 1.0
    quantile_low = max(0.0, min(quantile_low, 1.0))
    quantile_high = max(0.0, min(quantile_high, 1.0))
    if quantile_high <= quantile_low:
        quantile_low, quantile_high = 0.038, 0.962
    if min_hr <= 0:
        min_hr = 45.0
    if max_hr <= min_hr:
        max_hr = min_hr + 1.0
    if analysis_window_seconds <= 0:
        analysis_window_seconds = 30.0

    # 時間軸の作成
    time_data = list(accumulate(rri_data / 1000))
    if not time_data:
        print("  -> RRIデータから時間軸を作成できませんでした。")
        return None, None

    duration_total = time_data[-1]
    time_axis = np.arange(0, duration_total, 1 / resampling_freq)

    # スプライン補間
    if len(time_data) < 4:
        print("  -> データ点が少なすぎるためスキップします。")
        return None, None

    spline_func = interpolate.interp1d(time_data, rri_data, fill_value="extrapolate", kind='cubic')
    rri = spline_func(time_axis)

    # --- 前処理・フィルタリング ---
    # 外れ値除去
    min_samples_for_quantile = int(max(resampling_freq * 10, 10))
    if len(rri) > min_samples_for_quantile:
        low_threshold = np.quantile(rri, quantile_low)
        high_threshold = np.quantile(rri, quantile_high)
        rri[(rri < low_threshold) | (rri > high_threshold)] = np.nan

    # 欠損値補間
    df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    # 心拍数制限
    rri[(rri > 60000 / min_hr) | (rri < 60000 / max_hr)] = np.nan
    df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    # 急激な変化の抑制
    if len(rri) > 1:
        prerri = np.roll(rri, 1)
        prerri[0] = rri[0]
        safe_prerri = np.where(prerri == 0, 1, prerri)
        change_ratio = rri / safe_prerri
        rri[(change_ratio < 0.7) | (change_ratio > 1.3)] = np.nan
        df_temp = pd.DataFrame(data=rri, index=time_axis, columns=["rri"])
        df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
        rri = df_temp["rri"].values

    # --- 全体のLF/HF計算 ---
    N_total = len(rri)
    dt_total = 1 / resampling_freq
    window_total = scipy.signal.windows.hann(N_total)
    F_total = np.fft.fft(rri * window_total)
    freq_total = np.fft.fftfreq(N_total, d=dt_total)
    Amp_total = np.abs(F_total / (N_total / 2))

    lf_mask_total = (freq_total >= 0.04) & (freq_total < 0.15)
    hf_mask_total = (freq_total >= 0.15) & (freq_total < 0.4)
    LF_total = np.sum(Amp_total[lf_mask_total])
    HF_total = np.sum(Amp_total[hf_mask_total])

    overall_lf_hf = LF_total / HF_total if HF_total != 0 else 0

    print(f"  -> 全体LF/HF値: {overall_lf_hf:.4f}")

    # 全体結果をCSVに保存
    if output_csv_path and start_time is not None:
        overall_result_data = {
            'ファイル名': [os.path.basename(file_path)],
            '解析開始時刻': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
            '解析終了時刻': [end_time.strftime('%Y-%m-%d %H:%M:%S')],
            'LF/HF (全体)': [f"{overall_lf_hf:.4f}"]
        }
        overall_df = pd.DataFrame(overall_result_data)
        try:
            overall_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"  -> 全体結果を {output_csv_path} に保存しました。")
        except IOError as e:
            print(f"  -> エラー: CSVファイルの書き込みに失敗しました。 {e}")

    # --- スライディングウィンドウ解析 ---
    target_window_samples = max(int(round(analysis_window_seconds * resampling_freq)), 1)
    min_window_samples = max(int(round(10 * resampling_freq)), 1)

    if len(rri) >= target_window_samples:
        analysis_window = target_window_samples
    elif len(rri) >= min_window_samples:
        analysis_window = len(rri)
        actual_seconds = analysis_window / resampling_freq
        print(f"  -> データが短いため、ウィンドウサイズを {actual_seconds:.1f}秒 に短縮して解析します。")
    else:
        print("  -> データが短すぎて解析できません（約10秒未満）。")
        return None, None

    LF_HF_sliding = []
    RMSSD_sliding = []
    SDNN_sliding = []
    time_points = []

    i = 0
    while i <= (len(rri) - analysis_window):
        rri_window = rri[i:analysis_window + i]

        # LF/HF計算
        N = len(rri_window)
        dt = 1 / resampling_freq
        window = scipy.signal.windows.hann(N)
        F = np.fft.fft(rri_window * window)
        freq = np.fft.fftfreq(N, d=dt)
        Amp = np.abs(F / (N / 2))

        lf_mask = (freq >= 0.04) & (freq < 0.15)
        hf_mask = (freq >= 0.15) & (freq < 0.4)
        LF = np.sum(Amp[lf_mask])
        HF = np.sum(Amp[hf_mask])
        lf_hf_val = LF / HF if HF != 0 else 0
        LF_HF_sliding.append(lf_hf_val)

        # RMSSD計算
        if len(rri_window) > 1:
            diff_rri = np.diff(rri_window)
            mssd = np.mean(np.square(diff_rri))
            rmssd_val = np.sqrt(mssd)
            RMSSD_sliding.append(rmssd_val)

            # SDNN計算（新機能）
            sdnn_val = np.std(rri_window)
            SDNN_sliding.append(sdnn_val)
        else:
            RMSSD_sliding.append(np.nan)
            SDNN_sliding.append(np.nan)

        time_points.append(i)
        i += 1

    # 結果をDataFrameに格納
    sliding_result_df = pd.DataFrame({
        'Time': time_points,
        'LF/HF': LF_HF_sliding,
        'RMSSD': RMSSD_sliding,
        'SDNN': SDNN_sliding
    })

    # Excelに保存
    if output_excel_path:
        try:
            sliding_result_df.to_excel(output_excel_path, index=False)
            print(f"  -> スライディングウィンドウの結果を {output_excel_path} に保存しました。")
        except Exception as e:
            print(f"  -> エラー: Excelファイルの書き込みに失敗しました。 {e}")

    return sliding_result_df, overall_lf_hf


# ---------------------------------------------------------
# 3. 可視化関数
# ---------------------------------------------------------

def plot_hrv_indices(time_points, lf_hf, rmssd, sdnn=None, save_path_base=None):
    """LF/HF, RMSSD, SDNNの時間変化をプロットする

    Args:
        time_points: 時間点のリスト
        lf_hf: LF/HFのリスト
        rmssd: RMSSDのリスト
        sdnn: SDNNのリスト（オプション）
        save_path_base: 保存パスのベース名（拡張子なし）
    """
    if not time_points:
        print("プロットするデータがありません。")
        return

    n_plots = 3 if sdnn is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)

    if n_plots == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    # LF/HFのプロット
    ax1.plot(time_points, lf_hf, 'b-', linewidth=2, label='LF/HF')
    ax1.set_ylabel('LF/HF')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # RMSSDのプロット
    ax2.plot(time_points, rmssd, 'g-', linewidth=2, label='RMSSD')
    ax2.set_ylabel('RMSSD (ms)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # SDNNのプロット（存在する場合）
    if ax3 is not None and sdnn is not None:
        ax3.plot(time_points, sdnn, 'r-', linewidth=2, label='SDNN')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('SDNN (ms)')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
    else:
        ax2.set_xlabel('Time (s)')

    plt.tight_layout()

    if save_path_base:
        save_path = f"{save_path_base}_hrv_plot.png"
        plt.savefig(save_path, dpi=300)
        print(f"プロットを {save_path} に保存しました。")

    plt.show()


def plot_hrv_from_dataframe(df, label, output_dir):
    """DataFrameから時系列プロットを作成して保存する

    Args:
        df: 時系列データのDataFrame
        label: 解析ラベル
        output_dir: 出力ディレクトリ
    """
    print(f"  -> Creating time-series plot for {label}...")

    has_sdnn = 'SDNN' in df.columns
    n_plots = 3 if has_sdnn else 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    fig.suptitle(f'Time-Series Analysis: {label}', fontsize=16)

    if n_plots == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    # LF/HFのプロット
    ax1.plot(df['Time'], df['LF/HF'], label='LF/HF', color='dodgerblue')
    ax1.set_ylabel('LF/HF')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # RMSSDのプロット
    ax2.plot(df['Time'], df['RMSSD'], label='RMSSD', color='limegreen')
    ax2.set_ylabel('RMSSD (ms)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # SDNNのプロット
    if ax3 is not None and has_sdnn:
        ax3.plot(df['Time'], df['SDNN'], label='SDNN', color='coral')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('SDNN (ms)')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
    else:
        ax2.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(output_dir, f"{label}_timeseries.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  -> Time-series plot saved: {os.path.basename(save_path)}")


# ---------------------------------------------------------
# 使用例
# ---------------------------------------------------------
if __name__ == "__main__":
    current_name = "kawato2"
    ecg_file_path = "/Users/user/Documents/exp0707/kawato/h10_ecg_session_20250728_162632.csv"
    output_excel = f"/Users/user/Documents/exp0707/result/{current_name}_hrv_result.xlsx"
    output_csv = f"/Users/user/Documents/exp0707/result/{current_name}_lf_hf_overall_result.csv"
    plot_path_base = f"/Users/user/Documents/exp0707/result/{current_name}_analysis_result"

    # HRV指標の計算
    sliding_df, overall_lfhf = calculate_hrv_indices(
        ecg_file_path,
        label=current_name,
        output_excel_path=output_excel,
        output_csv_path=output_csv
    )

    # 結果をプロット
    if sliding_df is not None:
        plot_hrv_indices(
            sliding_df['Time'].tolist(),
            sliding_df['LF/HF'].tolist(),
            sliding_df['RMSSD'].tolist(),
            sliding_df['SDNN'].tolist() if 'SDNN' in sliding_df.columns else None,
            plot_path_base
        )
