import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, find_peaks, filtfilt
import os
from itertools import accumulate
from scipy import interpolate
import scipy
import math

def ecg_to_rri(file_path, fs=130):
    """
    ECGデータからRRI（RR間隔）を計算する
    指定された5分間のデータのみを対象とする

    Parameters:
    file_path (str): ECGデータのファイルパス
    fs (int): サンプリング周波数（デフォルト: 130Hz）

    Returns:
    tuple: (RRIデータ (ms単位), 解析開始時刻, 解析終了時刻) or (空の配列, None, None)
    """
    # バンドパスフィルタ関数
    def bandpass_filter(signal, fs):
        lowcut = 0.5   # 低周波カットオフ
        highcut = 50   # 高周波カットオフ
        nyq = 0.5 * fs  # ナイキスト周波数
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, signal)

    # pandasでCSVを読み込み、列に名前を付ける（ヘッダーは無いものとする）
    try:
        data = pd.read_csv(file_path, delimiter=',', encoding="utf-8", skiprows=1, header=None, usecols=[0, 2], names=['timestamp', 'ecg'])
    except ValueError:
        print("エラー: CSVファイルの列数が想定と異なります。タイムスタンプとECGデータがそれぞれ1列目と3列目にあるか確認してください。")
        return np.array([]), None, None

    # タイムスタンプの列をdatetimeオブジェクトに変換
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # 最初のデータ時刻を取得
    start_time_real = data['timestamp'].iloc[0]

    # 解析対象の開始時刻と終了時刻を定義（記録開始から1分後～6分後）
    analysis_start_time = start_time_real + pd.Timedelta(seconds=30)
    analysis_end_time = start_time_real + pd.Timedelta(minutes=5, seconds=30)

    # 指定した5分間のデータを抽出
    data_filtered = data[(data['timestamp'] >= analysis_start_time) & (data['timestamp'] <= analysis_end_time)]

    if data_filtered.empty:
        print("エラー: 指定された時間範囲（1分後～6分後）にデータがありません。データ長が6分以上あるか確認してください。")
        return np.array([]), None, None

    print(f"解析対象期間: {analysis_start_time} から {analysis_end_time} まで")
    print(f"抽出されたデータ数: {len(data_filtered)} 件")

    if not data_filtered.empty:
        original_start_index = data_filtered.index[0] + 2
        original_end_index = data_filtered.index[-1] + 2
        print(f"元のCSVファイルにおける開始行: {original_start_index}行目")
        print(f"元のCSVファイルにおける終了行: {original_end_index}行目")

    ecg_data = data_filtered['ecg'].values

    filtered_ecg = bandpass_filter(ecg_data, fs)
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
    height_threshold = np.mean(integrated_ecg) * 0.4
    distance = fs * 0.3
    peaks, _ = find_peaks(integrated_ecg, distance=distance, height=height_threshold)
    rri_data = np.diff(peaks) * 1000 / fs

    return rri_data, analysis_start_time, analysis_end_time

def calculate_hrv_indices(file_path, output_excel_path=None, output_csv_path=None):
    """
    ECGデータからLF/HFとRMSSDを計算し、結果を保存する

    Parameters:
    file_path (str): ECGデータのファイルパス
    output_excel_path (str, optional): スライディングウィンドウ結果を保存するExcelファイルパス
    output_csv_path (str, optional): 全体結果を保存するCSVファイルパス

    Returns:
    tuple: (時間点, LF/HFリスト, RMSSDリスト)
    """
    rri_data, start_time, end_time = ecg_to_rri(file_path)

    if len(rri_data) == 0:
        print("RRIデータが取得できなかったため、計算を中止します。")
        return [], [], []

    time_data = list(accumulate(rri_data / 1000))

    resampling_freq = 1
    if not time_data:
        print("RRIデータから時間軸を作成できませんでした。")
        return [], [], []
    time = np.arange(0, int(time_data[-1]), 1 / resampling_freq)

    def resample(time_index, rri):
        spline_func = interpolate.interp1d(time_index, rri, fill_value="extrapolate", kind='cubic')
        return spline_func(time)

    rri = resample(time_data, rri_data)

    # 外れ値除去・補間
    low_threshold = np.quantile(rri, 0.038)
    high_threshold = np.quantile(rri, 0.962)
    rri[(rri < low_threshold) | (rri > high_threshold)] = np.nan
    df = pd.DataFrame(data=rri, index=time, columns=["rri"])
    df.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df["rri"].values

    # 心拍数制限
    MinHR, MaxHR = 45, 210
    rri[(rri > 60000 / MinHR) | (rri < 60000 / MaxHR)] = np.nan
    df = pd.DataFrame(data=rri, index=time, columns=["rri"])
    df.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df["rri"].values

    # 急激な変化の修正
    prerri = np.roll(rri, 1)
    prerri[0] = rri[0]
    change_ratio = rri / prerri
    rri[(change_ratio < 0.7) | (change_ratio > 1.3)] = np.nan
    df = pd.DataFrame(data=rri, index=time, columns=["rri"])
    df.interpolate(method='spline', order=3, inplace=True, limit_direction='both')
    rri = df["rri"].values

    # --- 5分間全体のLF/HF計算 ---
    if len(rri) > 0:
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
        LF_HF_total = LF_total / HF_total if HF_total != 0 else 0

        print("\n--- 解析結果 (5分間全体) ---")
        print(f"LF/HF (全体): {LF_HF_total:.4f}")
        print("---------------------------\n")

        if output_csv_path:
            overall_result_data = {
                'ファイル名': [os.path.basename(file_path)],
                '解析開始時刻': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
                '解析終了時刻': [end_time.strftime('%Y-%m-%d %H:%M:%S')],
                'LF/HF (全体)': [f"{LF_HF_total:.4f}"]
            }
            overall_df = pd.DataFrame(overall_result_data)
            try:
                overall_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"全体結果を {output_csv_path} に保存しました。")
            except IOError as e:
                print(f"エラー: CSVファイルの書き込みに失敗しました。 {e}")

    # --- スライディングウィンドウでのLF/HFおよびRMSSD計算 ---
    analysis_time = 30
    LF_HF_sliding = []
    RMSSD_sliding = [] # ★追加：RMSSDの結果を保存するリスト
    time_points = []
    i = 0
    while i <= (len(rri) - analysis_time):
        rri_i = rri[i:analysis_time + i]
        
        # --- LF/HF計算 ---
        N = len(rri_i)
        dt = 1 / resampling_freq
        window = scipy.signal.windows.hann(N)
        F = np.fft.fft(rri_i * window)
        freq = np.fft.fftfreq(N, d=dt)
        Amp = np.abs(F / (N / 2))
        lf_mask = (freq >= 0.04) & (freq < 0.15)
        hf_mask = (freq >= 0.15) & (freq < 0.4)
        LF = np.sum(Amp[lf_mask])
        HF = np.sum(Amp[hf_mask])
        LF_HF_t = LF / HF if HF != 0 else 0
        LF_HF_sliding.append(LF_HF_t)

        # --- ★ここからRMSSD計算を追加 ---
        if len(rri_i) > 1:
            diff_rri = np.diff(rri_i)           # 隣り合うRR間隔の差
            mssd = np.mean(np.square(diff_rri)) # 差の二乗平均 (MSSD)
            rmssd_t = np.sqrt(mssd)             # MSSDの平方根 (RMSSD)
            RMSSD_sliding.append(rmssd_t)
        else:
            RMSSD_sliding.append(np.nan) # データが少ない場合はNaNを追加
        # --- ★ここまで追加 ---
            
        time_points.append(i)
        i += 1

    if output_excel_path:
        # ★変更: DataFrameにRMSSD列を追加
        results_df = pd.DataFrame({
            'Time': time_points,
            'LF/HF': LF_HF_sliding,
            'RMSSD': RMSSD_sliding
        })
        results_df.to_excel(output_excel_path, index=False)
        print(f"スライディングウィンドウの結果を {output_excel_path} に保存しました。")

    return time_points, LF_HF_sliding, RMSSD_sliding

def plot_hrv_indices(time_points, lf_hf, rmssd, save_path_base=None):
    """LF/HFとRMSSDの時間変化をプロットする"""
    if not time_points:
        print("プロットするデータがありません。")
        return

    # 2つのグラフを縦に並べて表示
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # LF/HFのプロット
    ax1.plot(time_points, lf_hf, 'b-', linewidth=2, label='LF/HF')
    ax1.set_title('')
    ax1.set_ylabel('LF/HF')
    ax1.grid(True)
    ax1.legend()

    # RMSSDのプロット
    ax2.plot(time_points, rmssd, 'g-', linewidth=2, label='RMSSD')
    ax2.set_xlabel('seconds')
    ax2.set_ylabel('RMSSD')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout() # グラフの重なりを防ぐ
    
    if save_path_base:
        save_path = f"{save_path_base}_hrv_plot.png"
        plt.savefig(save_path)
        print(f"プロットを {save_path} に保存しました。")
        
    plt.show()

# --- 使用例 ---
if __name__ == "__main__":
    current_name = "kawato2"
    # ▼▼▼ ファイルパスを環境に合わせて変更してください ▼▼▼
    ecg_file_path = "/Users/user/Documents/exp0707/kawato/h10_ecg_session_20250728_162632.csv"
    # スライディングウィンドウ結果（LF/HF と RMSSD）
    output_excel = f"/Users/user/Documents/exp0707/result/{current_name}_lfhfrmssd_result.xlsx" 
    # 5分間全体のLF/HF結果
    output_csv = f"/Users/user/Documents/exp0707/result/{current_name}_lf_hf_overall_result.csv"
    # プロット画像の保存パス（拡張子なしで指定）
    plot_path_base = f"/Users/user/Documents/exp0707/result/{current_name}_analysis_result"
    # ▲▲▲ ファイルパスを環境に合わせて変更してください ▲▲▲
    
    # HRV指標（LF/HF, RMSSD）の計算
    time_points, lf_hf, rmssd = calculate_hrv_indices(ecg_file_path, output_excel, output_csv)
    
    # 結果をプロット
    if time_points:
        plot_hrv_indices(time_points, lf_hf, rmssd, plot_path_base)