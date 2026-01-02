# Analyzer.py
"""バッチ解析モジュール

複数のECGファイルを一括で解析し、結果を統合・保存する。
HCS_ver4.0の解析機能を参考にブラッシュアップ。
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
import seaborn as sns
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

from AnalysisECG import (
    calculate_hrv_indices,
    plot_hrv_from_dataframe,
    ECG_CONDITIONS,
    CONDITION_LABELS,
    CONDITION_COLORS,
)


# ---------------------------------------------------------
# ファイル名パターン定義
# ---------------------------------------------------------
_FILENAME_COND_PATTERN = "|".join(ECG_CONDITIONS)
FILENAME_PATTERN = re.compile(
    rf"h10_ecg_session_No(?P<subject>\d+)_\d{{8}}_\d{{6}}_(?P<condition>{_FILENAME_COND_PATTERN})\.csv$",
    re.IGNORECASE,
)

# 条件名の正規化マップ
CONDITION_MAP = {
    "sin": "Sin",
    "fixed": "Fixed",
    "hrf": "HRF",
    "hrf2_pid": "HRF2_PID",
    "hrf2_adaptive": "HRF2_Adaptive",
    "hrf2_gs": "HRF2_GS",
    "hrf2_gainscheduled": "HRF2_GS",
    "hrf2_robust": "HRF2_Robust",
}


# ---------------------------------------------------------
# バッチ解析関数
# ---------------------------------------------------------

def run_batch_analysis(
    files_map,
    output_dir,
    analysis_start_offset=None,
    analysis_end_offset=None,
    sensor_sample_rate=130.0,
    resampling_freq=1.0,
    quantile_low=0.038,
    quantile_high=0.962,
    min_hr=45.0,
    max_hr=210.0,
    analysis_window_seconds=30.0,
    subject_id=None,
    save_plots=True,
):
    """バッチ解析を実行し、結果をExcelファイルに保存する

    Args:
        files_map: 解析対象のファイルを {label: file_path} の形式で格納した辞書
        output_dir: 結果を出力するディレクトリ
        analysis_start_offset: 解析開始オフセット（秒）
        analysis_end_offset: 解析終了オフセット（秒）
        sensor_sample_rate: センサーのサンプリング周波数
        resampling_freq: リサンプリング周波数
        quantile_low: 外れ値除去の下限パーセンタイル
        quantile_high: 外れ値除去の上限パーセンタイル
        min_hr: 最小心拍数制限
        max_hr: 最大心拍数制限
        analysis_window_seconds: スライディングウィンドウの秒数
        subject_id: 被験者ID（出力ファイル名に含める。Noneの場合はフォルダ名から推測）
        save_plots: 時系列グラフを保存するか
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_df = None

    print("=== バッチ解析を開始します ===")

    # subject_idが指定されていない場合、出力フォルダ名から推測
    if subject_id is None:
        folder_name = os.path.basename(output_dir.rstrip('/\\'))
        # No1, No2 などのパターンをチェック
        match = re.match(r'(No\d+)', folder_name, re.IGNORECASE)
        if match:
            subject_id = match.group(1)
        else:
            subject_id = folder_name

    for label, filename in files_map.items():
        file_path = filename

        if not os.path.exists(file_path):
            print(f"警告: ファイルが見つかりません -> {file_path}")
            continue

        # 解析実行
        sliding_df, overall_lfhf = calculate_hrv_indices(
            file_path,
            label,
            fs=int(sensor_sample_rate),
            analysis_start_offset=analysis_start_offset,
            analysis_end_offset=analysis_end_offset,
            resampling_freq=resampling_freq,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            min_hr=min_hr,
            max_hr=max_hr,
            analysis_window_seconds=analysis_window_seconds,
        )

        if sliding_df is not None and not sliding_df.empty:
            # 時系列データの個別保存（被験者番号を含むファイル名）
            sliding_output_path = os.path.join(output_dir, f"{subject_id}_{label}_result.xlsx")
            sliding_df.to_excel(sliding_output_path, index=False)
            print(f"  -> 時系列結果を保存: {os.path.basename(sliding_output_path)}")

            # 時系列グラフの保存
            if save_plots:
                plot_hrv_from_dataframe(sliding_df, f"{subject_id}_{label}", output_dir)

            # 全体LF/HFの個別保存
            overall_output_path = os.path.join(output_dir, f"{subject_id}_{label}_resultLFHF5min.xlsx")
            overall_df_file = pd.DataFrame({
                'File Name': [filename],
                'LF/HF (Overall)': [overall_lfhf]
            })
            overall_df_file.to_excel(overall_output_path, index=False)
            print(f"  -> 全体平均結果を保存: {os.path.basename(overall_output_path)}")

            # 結合用データの準備（SDNN列も追加）
            df_renamed = sliding_df.copy()
            if 'SDNN' in df_renamed.columns:
                df_renamed.columns = ['Time', f'{label}_LF/HF', f'{label}_RMSSD', f'{label}_SDNN']
            else:
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
                if f'{label}_SDNN' in combined_df.columns:
                    cols.append(f'{label}_SDNN')

        combined_df = combined_df[cols]
        combined_output_path = os.path.join(output_dir, f"{subject_id}_Combined_HRV_Analysis.xlsx")
        combined_df.to_excel(combined_output_path, index=False)
        print(f"\n=== 全データの結合ファイルを保存しました ===")
        print(f"保存先: {combined_output_path}")
    else:
        print("\n有効な解析結果が1つもありませんでした。")

    print("\n処理完了。")


# ---------------------------------------------------------
# 箱ひげ図生成関数
# ---------------------------------------------------------

def generate_box_plots(
    input_file_path,
    output_dir,
    condition_labels=None,
    condition_order=None,
):
    """箱ひげ図を生成する

    Args:
        input_file_path: 入力ファイルパス（結合済みHRV解析結果のExcel）
        output_dir: 出力ディレクトリ
        condition_labels: 条件ラベルのマッピング辞書
        condition_order: 条件の表示順序リスト

    Returns:
        保存されたファイルパスのリスト
    """
    condition_labels = condition_labels or CONDITION_LABELS
    condition_order = condition_order or ECG_CONDITIONS
    colors = CONDITION_COLORS

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(input_file_path)
    print("データの読み込みに成功しました。")

    saved_files = []

    for metric_suffix, title, output_filename in [
        ('_LF/HF', 'LF/HFの比較（時系列分布）', 'LFHF_Boxplot.png'),
        ('_RMSSD', 'RMSSDの比較（時系列分布）', 'RMSSD_Boxplot.png'),
        ('_SDNN', 'SDNNの比較（時系列分布）', 'SDNN_Boxplot.png')
    ]:
        print(f"--- {title} のグラフを作成中 ---")
        plot_data = pd.DataFrame()
        found_cols = False

        for eng_key, display_label in condition_labels.items():
            col_name = f"{eng_key}{metric_suffix}"
            if col_name in df.columns:
                label = display_label or eng_key
                plot_data[label] = df[col_name]
                found_cols = True

        if not found_cols:
            print(f"  -> {metric_suffix} に関するデータが見つかりませんでした。スキップします。")
            continue

        df_melted = plot_data.melt(var_name='Condition', value_name='Value')
        df_melted = df_melted.dropna()

        # Aggバックエンドを使用（スレッドセーフ）
        fig = Figure(figsize=(10, 7))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # 表示順序の決定
        available_labels = set(df_melted['Condition'])
        order_labels = []
        for cond in condition_order:
            label = condition_labels.get(cond)
            if label and label in available_labels:
                order_labels.append(label)
        if not order_labels:
            order_labels = list(df_melted['Condition'].unique())

        # パレットの設定
        palette = {}
        for cond in condition_order:
            label = condition_labels.get(cond)
            if label and label in order_labels:
                palette[label] = colors.get(cond, 'lightgray')
        for label in order_labels:
            if label not in palette:
                palette[label] = 'lightgray'

        sns.boxplot(
            x='Condition',
            y='Value',
            data=df_melted,
            palette=palette,
            ax=ax,
            showfliers=False,
            width=0.5,
            order=order_labels
        )

        # 凡例
        legend_patches = [
            mpatches.Patch(color=palette[label], label=label)
            for label in order_labels
        ]
        ax.legend(handles=legend_patches, title="条件", loc='upper right')
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(metric_suffix.replace('_', ''), fontsize=14)
        ax.set_xlabel("条件", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        save_path = os.path.join(output_dir, output_filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"  -> 保存完了: {save_path}")
        saved_files.append(save_path)

    print("\nすべてのグラフ作成が完了しました。")
    return saved_files


# ---------------------------------------------------------
# 複数被験者の統計比較
# ---------------------------------------------------------

def compare_subjects(
    data_dir,
    output_dir,
    conditions=None,
    metrics=None,
):
    """複数被験者のデータを比較する

    Args:
        data_dir: 被験者データを含むディレクトリ
        output_dir: 出力ディレクトリ
        conditions: 比較する条件のリスト（Noneの場合は全条件）
        metrics: 比較する指標のリスト（Noneの場合は['LF/HF', 'RMSSD', 'SDNN']）

    Returns:
        比較結果のDataFrame
    """
    conditions = conditions or ECG_CONDITIONS
    metrics = metrics or ['LF/HF', 'RMSSD', 'SDNN']

    os.makedirs(output_dir, exist_ok=True)

    # 結合ファイルを検索
    combined_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('_Combined_HRV_Analysis.xlsx'):
                combined_files.append(os.path.join(root, f))

    if not combined_files:
        print("結合ファイルが見つかりませんでした。")
        return None

    # 各被験者のデータを集約
    summary_data = []

    for file_path in combined_files:
        # ファイル名から被験者IDを抽出
        filename = os.path.basename(file_path)
        match = re.match(r'(No\d+)', filename, re.IGNORECASE)
        if match:
            subject_id = match.group(1)
        else:
            subject_id = os.path.splitext(filename)[0].replace('_Combined_HRV_Analysis', '')

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"警告: {file_path} の読み込みに失敗しました: {e}")
            continue

        row = {'Subject': subject_id}

        for cond in conditions:
            for metric in metrics:
                col_name = f"{cond}_{metric}"
                if col_name in df.columns:
                    # 平均値と標準偏差を計算
                    values = df[col_name].dropna()
                    if len(values) > 0:
                        row[f"{cond}_{metric}_mean"] = values.mean()
                        row[f"{cond}_{metric}_std"] = values.std()

        summary_data.append(row)

    if not summary_data:
        print("有効なデータが見つかりませんでした。")
        return None

    summary_df = pd.DataFrame(summary_data)
    summary_df.sort_values('Subject', inplace=True)

    # 結果を保存
    output_path = os.path.join(output_dir, "Subject_Comparison.xlsx")
    summary_df.to_excel(output_path, index=False)
    print(f"被験者比較結果を保存しました: {output_path}")

    return summary_df


# ---------------------------------------------------------
# 統計サマリー生成
# ---------------------------------------------------------

def generate_summary_statistics(
    input_file_path,
    output_dir,
    conditions=None,
):
    """統計サマリーを生成する

    Args:
        input_file_path: 結合済みHRV解析結果のExcelファイル
        output_dir: 出力ディレクトリ
        conditions: 比較する条件のリスト

    Returns:
        サマリーのDataFrame
    """
    conditions = conditions or ECG_CONDITIONS

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(input_file_path)

    summary_data = []

    for cond in conditions:
        for metric_suffix in ['_LF/HF', '_RMSSD', '_SDNN']:
            col_name = f"{cond}{metric_suffix}"
            if col_name in df.columns:
                values = df[col_name].dropna()
                if len(values) > 0:
                    summary_data.append({
                        'Condition': cond,
                        'Metric': metric_suffix.replace('_', ''),
                        'N': len(values),
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Median': values.median(),
                        'Q1': values.quantile(0.25),
                        'Q3': values.quantile(0.75),
                    })

    if not summary_data:
        print("統計サマリーを生成するデータがありませんでした。")
        return None

    summary_df = pd.DataFrame(summary_data)

    # 結果を保存
    output_path = os.path.join(output_dir, "Statistics_Summary.xlsx")
    summary_df.to_excel(output_path, index=False)
    print(f"統計サマリーを保存しました: {output_path}")

    return summary_df


# ---------------------------------------------------------
# メイン実行ブロック
# ---------------------------------------------------------

if __name__ == "__main__":
    base_dir_main = os.path.dirname(os.path.abspath(__file__))

    # 入力ファイル設定
    default_files_map = {
        "Fixed": "/Users/user/Documents/MHS2025/kawato/h10_ecg_session_20250728_162632.csv",
        "HRF": "/Users/user/Documents/MHS2025/kawato/h10_ecg_session_20250728_161705.csv",
        "Sin": "/Users/user/Documents/MHS2025/kawato/h10_ecg_session_20250728_160807.csv"
    }

    default_output_dir = os.path.join(base_dir_main, "result_batch")

    # バッチ解析実行
    run_batch_analysis(
        default_files_map,
        default_output_dir,
        subject_id="No1",
    )

    # 箱ひげ図生成（結合ファイルが存在する場合）
    combined_file = os.path.join(default_output_dir, "No1_Combined_HRV_Analysis.xlsx")
    if os.path.exists(combined_file):
        generate_box_plots(combined_file, default_output_dir)
        generate_summary_statistics(combined_file, default_output_dir)
