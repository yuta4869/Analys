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
    指定されたファイルを解析し、DataFrame（Time, LF/HF, RMSSD）を返す。
    """
    print(f"--- 解析開始: {label} ({os.path.basename(file_path)}) ---")
    rri_data, start_time, end_time = ecg_to_rri(file_path, fs)

    if len(rri_data) == 0:
        return None

    # 時間軸の作成とリサンプリング (1Hz)
    time_data = list(accumulate(rri_data / 1000))
    if not time_data:
        return None
    
    resampling_freq = 1
    duration_total = int(time_data[-1])
    time = np.arange(0, duration_total, 1 / resampling_freq)

    # スプライン補間
    if len(time_data) < 4: # データ点が少なすぎる場合の保護
        print("  -> データ点が少なすぎるためスキップします。")
        return None
        
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

    # --- スライディングウィンドウ解析 ---
    # データ長に応じてウィンドウサイズを調整（通常は30秒）
    if len(rri) >= 30:
        analysis_window = 30
    elif len(rri) >= 10:
        analysis_window = 10 # 短いサンプル用
        print(f"  -> データが短いため、ウィンドウサイズを {analysis_window}秒 に短縮して解析します。")
    else:
        print("  -> データが短すぎて解析できません（10秒未満）。")
        return None

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
    result_df = pd.DataFrame({
        'Time': time_points,
        'LF/HF': LF_HF_sliding,
        'RMSSD': RMSSD_sliding
    })
    
    return result_df

# ---------------------------------------------------------
# 3. メイン実行ブロック
# ---------------------------------------------------------

if __name__ == "__main__":
    # ▼▼▼ 環境設定: ここを自分のPCのパスに合わせて変更してください ▼▼▼
    base_dir = "/Users/user/Documents/exp0707"  # ファイルがあるフォルダ
    output_dir = os.path.join(base_dir, "result_batch") # 結果を出力するフォルダ
    
    # 入力ファイル名（フォルダ内にあるファイル名を指定）
    files_map = {
        "Fixed": "h10_ecg_session_20251204_015103_Fixed.csv",
        "HRF":   "h10_ecg_session_20251204_015139_HRF.csv",
        "Sin":   "h10_ecg_session_20251204_015223_Sin.csv"
    }
    # ▲▲▲ 設定ここまで ▲▲▲

    # 出力フォルダ作成
    os.makedirs(output_dir, exist_ok=True)
    
    combined_df = None
    processed_files = []

    print("=== バッチ解析を開始します ===")

    for label, filename in files_map.items():
        file_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"警告: ファイルが見つかりません -> {file_path}")
            continue
            
        # 解析実行
        df_result = calculate_hrv_indices(file_path, label)
        
        if df_result is not None and not df_result.empty:
            # 1. 個別ファイルの保存 (Time, LF/HF, RMSSD)
            individual_output_path = os.path.join(output_dir, f"{label}_result.xlsx")
            df_result.to_excel(individual_output_path, index=False)
            processed_files.append(individual_output_path)
            print(f"  -> 個別結果を保存: {os.path.basename(individual_output_path)}")
            
            # 2. 結合用データの準備 (列名を変更: LF/HF -> Fixed_LF/HF)
            df_renamed = df_result.copy()
            df_renamed.columns = ['Time', f'{label}_LF/HF', f'{label}_RMSSD']
            
            # マージ処理 (Timeをキーにして外部結合)
            if combined_df is None:
                combined_df = df_renamed
            else:
                combined_df = pd.merge(combined_df, df_renamed, on='Time', how='outer')
        else:
            print(f"  -> {label} の解析結果が得られませんでした。")

    # 3. 結合ファイルの保存
    if combined_df is not None:
        # Timeでソート
        combined_df.sort_values('Time', inplace=True)
        
        # 列の並び順を整理 (Time, Fixed..., HRF..., Sin...)
        # 存在する列だけを抽出して並べ替え
        cols = ['Time']
        for label in files_map.keys():
            if f'{label}_LF/HF' in combined_df.columns:
                cols.append(f'{label}_LF/HF')
                cols.append(f'{label}_RMSSD')
        
        combined_df = combined_df[cols]
        
        combined_output_path = os.path.join(output_dir, "Combined_HRV_Analysis_All.xlsx")
        combined_df.to_excel(combined_output_path, index=False)
        print(f"\n=== 全データの結合ファイルを保存しました ===")
        print(f"保存先: {combined_output_path}")
    else:
        print("\n有効な解析結果が1つもありませんでした。")

    print("\n処理完了。")