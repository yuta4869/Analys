import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, find_peaks, filtfilt, windows
import scipy.signal
from itertools import accumulate
from scipy import interpolate
import os

# ---------------------------------------------------------
# 1. 信号処理・RRI算出ロジック
# ---------------------------------------------------------

def bandpass_filter(signal, fs):
    """バンドパスフィルタ (0.5Hz - 50Hz)"""
    lowcut = 0.5
    highcut = 50
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal)

def ecg_to_rri(file_path, fs=130):
    """
    ECGデータからRRIを算出する。
    データの長さに応じて、全期間解析するか5分間切り出しを行うかを自動判定する。
    """
    try:
        # ヘッダーなし、1列目(timestamp)と3列目(ecg)を使用
        data = pd.read_csv(file_path, delimiter=',', encoding="utf-8", skiprows=1, header=None, usecols=[0, 2], names=['timestamp', 'ecg'])
    except ValueError:
        print(f"エラー: {os.path.basename(file_path)} の読み込みに失敗しました。")
        return np.array([]), None, None

    # タイムスタンプ変換
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    start_time_real = data['timestamp'].iloc[0]
    end_time_real = data['timestamp'].iloc[-1]
    duration_seconds = (end_time_real - start_time_real).total_seconds()

    # --- データ長に応じた期間設定 ---
    if duration_seconds < 60:
        # 1分未満の短いサンプルデータの場合は、全期間を使用
        print(f"  -> 短いデータを検出 ({duration_seconds:.1f}秒): 全期間を解析します")
        analysis_start_time = start_time_real
        analysis_end_time = end_time_real
        data_filtered = data
    else:
        # 十分な長さがある場合は、安定する30秒後から5分間を使用（従来ロジック）
        print(f"  -> 長いデータを検出 ({duration_seconds:.1f}秒): 30秒後から5分間を解析します")
        analysis_start_time = start_time_real + pd.Timedelta(seconds=30)
        analysis_end_time = start_time_real + pd.Timedelta(minutes=5, seconds=30)
        data_filtered = data[(data['timestamp'] >= analysis_start_time) & (data['timestamp'] <= analysis_end_time)]

    if data_filtered.empty:
        print("  -> エラー: 解析対象期間のデータが空です。")
        return np.array([]), None, None

    ecg_data = data_filtered['ecg'].values

    # ピーク検出処理
    filtered_ecg = bandpass_filter(ecg_data, fs)
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
    
    # しきい値設定
    height_threshold = np.mean(integrated_ecg) * 0.4
    distance = fs * 0.3
    peaks, _ = find_peaks(integrated_ecg, distance=distance, height=height_threshold)
    
    rri_data = np.diff(peaks) * 1000 / fs # ms単位

    return rri_data, analysis_start_time, analysis_end_time

# ---------------------------------------------------------
# 2. HRV指標計算ロジック
# ---------------------------------------------------------

def calculate_hrv_indices(file_path, label, fs=130):
    """
    指定されたファイルを解析し、
    1. 時系列データのDataFrame（Time, LF/HF, RMSSD）
    2. 全体（5分間）のLF/HF値
    を返す。
    """
    print(f"--- 解析開始: {label} ({os.path.basename(file_path)}) ---")
    rri_data, start_time, end_time = ecg_to_rri(file_path, fs)

    if len(rri_data) == 0:
        return None, None

    # 時間軸の作成とリサンプリング (1Hz)
    time_data = list(accumulate(rri_data / 1000))
    if not time_data:
        return None, None
    
    resampling_freq = 1
    duration_total = int(time_data[-1])
    time = np.arange(0, duration_total, 1 / resampling_freq)

    # スプライン補間
    if len(time_data) < 4: # データ点が少なすぎる場合の保護
        print("  -> データ点が少なすぎるためスキップします。")
        return None, None
        
    spline_func = interpolate.interp1d(time_data, rri_data, fill_value="extrapolate", kind='cubic')
    rri = spline_func(time)

    # --- 前処理・フィルタリング ---
    # 外れ値除去
    if len(rri) > 10:
        low_threshold = np.quantile(rri, 0.038)
        high_threshold = np.quantile(rri, 0.962)
        rri[(rri < low_threshold) | (rri > high_threshold)] = np.nan

    # 欠損値補間
    df_temp = pd.DataFrame(data=rri, index=time, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    # 心拍数制限 (45-210 bpm)
    MinHR, MaxHR = 45, 210
    rri[(rri > 60000 / MinHR) | (rri < 60000 / MaxHR)] = np.nan
    df_temp = pd.DataFrame(data=rri, index=time, columns=["rri"])
    df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df_temp["rri"].values

    # 急激な変化の抑制
    if len(rri) > 1:
        prerri = np.roll(rri, 1)
        prerri[0] = rri[0]
        # 0除算防止
        safe_prerri = np.where(prerri == 0, 1, prerri)
        change_ratio = rri / safe_prerri
        rri[(change_ratio < 0.7) | (change_ratio > 1.3)] = np.nan
        df_temp = pd.DataFrame(data=rri, index=time, columns=["rri"])
        df_temp.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
        rri = df_temp["rri"].values

    # --- [A] 全体（5分間）のLF/HF計算 ---
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
    
    overall_lf_hf_val = LF_total / HF_total if HF_total != 0 else 0
    print(f"  -> 全体LF/HF値: {overall_lf_hf_val:.4f}")

    # --- [B] スライディングウィンドウ解析 ---
    # データ長に応じてウィンドウサイズを調整（通常は30秒）
    if len(rri) >= 30:
        analysis_window = 30
    elif len(rri) >= 10:
        analysis_window = 10 # 短いサンプル用
        print(f"  -> データが短いため、ウィンドウサイズを {analysis_window}秒 に短縮して解析します。")
    else:
        print("  -> データが短すぎて解析できません（10秒未満）。")
        return None, None

    LF_HF_sliding = []
    RMSSD_sliding = []
    time_points = []
    
    i = 0
    # 1秒ずつスライド
    while i <= (len(rri) - analysis_window):
        rri_window = rri[i : analysis_window + i]
        
        # --- LF/HF計算 ---
        N = len(rri_window)
        dt = 1 / resampling_freq
        
        # ハニング窓
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

        # --- RMSSD計算 ---
        if len(rri_window) > 1:
            diff_rri = np.diff(rri_window)
            mssd = np.mean(np.square(diff_rri))
            rmssd_val = np.sqrt(mssd)
            RMSSD_sliding.append(rmssd_val)
        else:
            RMSSD_sliding.append(np.nan)
            
        time_points.append(i)
        i += 1

    # 結果をDataFrameに格納
    sliding_result_df = pd.DataFrame({
        'Time': time_points,
        'LF/HF': LF_HF_sliding,
        'RMSSD': RMSSD_sliding
    })
    
    return sliding_result_df, overall_lf_hf_val

# ---------------------------------------------------------
# 3. バッチ解析実行ロジック
# ---------------------------------------------------------
def save_timeseries_plot(df, label, output_dir):
    """
    Creates and saves a time-series plot for LF/HF and RMSSD.
    """
    print(f"  -> Creating time-series plot for {label}...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Time-Series Analysis: {label}', fontsize=16)

    # Plot LF/HF
    axes[0].plot(df['Time'], df['LF/HF'], label='LF/HF', color='dodgerblue')
    axes[0].set_ylabel('LF/HF')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Plot RMSSD
    axes[1].plot(df['Time'], df['RMSSD'], label='RMSSD', color='limegreen')
    axes[1].set_ylabel('RMSSD (ms)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    save_path = os.path.join(output_dir, f"{label}_timeseries.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  -> Time-series plot saved: {os.path.basename(save_path)}")


def run_batch_analysis(files_map, output_dir):
    """
    指定されたファイルマップに基づいてバッチ解析を実行し、結果をExcelファイルに保存する。
    
    :param files_map: 解析対象のファイルを {label: file_path} の形式で格納した辞書
    :param output_dir: 結果を出力するディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    combined_df = None

    print("=== バッチ解析を開始します ===")

    for label, filename in files_map.items():
        file_path = filename
        
        if not os.path.exists(file_path):
            print(f"警告: ファイルが見つかりません -> {file_path}")
            continue
            
        # 解析実行
        sliding_df, overall_lfhf = calculate_hrv_indices(file_path, label)
        
        if sliding_df is not None and not sliding_df.empty:
            # 時系列データの個別保存
            sliding_output_path = os.path.join(output_dir, f"{label}_result.xlsx")
            sliding_df.to_excel(sliding_output_path, index=False)
            print(f"  -> 時系列結果を保存: {os.path.basename(sliding_output_path)}")

            # 時系列グラフの保存
            save_timeseries_plot(sliding_df, label, output_dir)

            # 全体LF/HFの個別保存
            overall_output_path = os.path.join(output_dir, f"{label}_resultLFHF5min.xlsx")
            overall_df = pd.DataFrame({'File Name': [filename], 'LF/HF (Overall)': [overall_lfhf]})
            overall_df.to_excel(overall_output_path, index=False)
            print(f"  -> 全体平均結果を保存: {os.path.basename(overall_output_path)}")
            
            # 結合用データの準備
            df_renamed = sliding_df.copy()
            df_renamed.columns = ['Time', f'{label}_LF/HF', f'{label}_RMSSD']
            
            if combined_df is None:
                combined_df = df_renamed
            else:
                combined_df = pd.merge(combined_df, df_renamed, on='Time', how='outer')
        else:
            print(f"  -> {label} の解析結果が得られませんでした。")

    # 結合ファイルの保存
    if combined_df is not None:
        combined_df.sort_values('Time', inplace=True)
        cols = ['Time']
        for label in files_map.keys():
            if f'{label}_LF/HF' in combined_df.columns:
                cols.append(f'{label}_LF/HF')
                cols.append(f'{label}_RMSSD')
        
        combined_df = combined_df[cols]
        combined_output_path = os.path.join(output_dir, "Combined_HRV_Analysis.xlsx")
        combined_df.to_excel(combined_output_path, index=False)
        print(f"\n=== 全データの結合ファイルを保存しました ===")
        print(f"保存先: {combined_output_path}")
    else:
        print("\n有効な解析結果が1つもありませんでした。")

    print("\n処理完了。")

# ---------------------------------------------------------
# 4. メイン実行ブロック
# ---------------------------------------------------------

if __name__ == "__main__":
    # このスクリプトを直接実行した場合のデフォルト動作
    # ▼▼▼ 環境設定: ここを自分のPCのパスに合わせて変更してください ▼▼▼
    base_dir_main = os.path.dirname(os.path.abspath(__file__))
    
    # 入力ファイル名（フォルダ内にあるファイル名を指定）
    default_files_map = {
        "Fixed": "/Users/user/Documents/MHS2025/kawato/h10_ecg_session_20250728_162632.csv",
        "HRF":   "/Users/user/Documents/MHS2025/kawato/h10_ecg_session_20250728_161705.csv",
        "Sin":   "/Users/user/Documents/MHS2025/kawato/h10_ecg_session_20250728_160807.csv"
    }
    # ▲▲▲ 設定ここまで ▲▲▲

    default_output_dir = os.path.join(base_dir_main, "result_batch")
    run_batch_analysis(default_files_map, default_output_dir)
