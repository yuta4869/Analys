import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib # 日本語表示用
import matplotlib.patches as mpatches
import os

# ---------------------------------------------------------
# 設定項目
# ---------------------------------------------------------

# 読み込むデータファイルのパス
# ※ 前回のスクリプトで生成された "Combined_HRV_Analysis.xlsx" のパスを指定してください
DEFAULT_INPUT_FILE_PATH = "/Users/user/Research/Analys/result_batch/Combined_HRV_Analysis.xlsx"

# 保存先のディレクトリ（指定がなければ入力ファイルと同じ場所）
DEFAULT_OUTPUT_DIR = os.path.dirname(DEFAULT_INPUT_FILE_PATH)

# 条件と列名の対応定義
# Excelの列名に含まれるキーワード (key) と、グラフに表示する日本語ラベル (label)
conditions = {
    'Fixed': '固定会話',
    'HRF': '調整会話',
    'Sin': '正弦波'
}
CONDITION_ORDER = ['Fixed', 'HRF', 'Sin']

# グラフの色設定
colors = {
    '固定会話': 'lightcoral', 
    '調整会話': 'lightyellow', 
    '正弦波': 'lightblue'
}

# ---------------------------------------------------------
# グラフ描画関数
# ---------------------------------------------------------

def _condition_palette(labels):
    palette = {}
    for label in labels:
        palette[label] = colors.get(label, 'lightgray')
    return palette


def create_boxplot_for_metric(df, metric_suffix, title, output_filename, output_dir):
    """
    指定された指標（LF/HF または RMSSD）の箱ひげ図を作成して保存する
    
    Parameters:
    df: データフレーム
    metric_suffix: 列名の末尾 (例: '_LF/HF', '_RMSSD')
    title: グラフのタイトル
    output_filename: 保存するファイル名
    """
    print(f"--- {title} のグラフを作成中 ---")
    
    # プロット用のデータフレームを作成
    plot_data = pd.DataFrame()
    
    found_cols = False
    for eng_key, jp_label in conditions.items():
        # 列名を構築 (例: Fixed_LF/HF)
        col_name = f"{eng_key}{metric_suffix}"
        
        if col_name in df.columns:
            plot_data[jp_label] = df[col_name]
            found_cols = True
        else:
            print(f"警告: 列 '{col_name}' が見つかりません。スキップします。")
    
    if not found_cols:
        print(f"エラー: {metric_suffix} に関するデータが見つかりませんでした。")
        return

    # Seaborn用にデータを縦持ち（Long format）に変換
    # columns: Condition (条件), Value (値)
    df_melted = plot_data.melt(var_name='Condition', value_name='Value')
    
    # 欠損値を除去
    df_melted = df_melted.dropna()

    # --- 描画 ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. 箱ひげ図
    # showfliers=False: 箱ひげ図には外れ値を表示せず、全体傾向を強調する
    order_labels = [conditions[key] for key in CONDITION_ORDER if conditions[key] in set(df_melted['Condition'])]
    if not order_labels:
        order_labels = list(df_melted['Condition'].unique())
    palette = _condition_palette(order_labels)
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
    
    # 凡例の作成
    legend_patches = [mpatches.Patch(color=palette[label], label=label) for label in order_labels]
    ax.legend(handles=legend_patches, title="条件", loc='upper right')

    # ラベルとタイトルの設定
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(metric_suffix.replace('_', ''), fontsize=14)
    ax.set_xlabel("条件", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"保存完了: {save_path}")
    plt.close() # メモリ解放
    return save_path


def generate_box_plots(input_file_path=None, output_dir=None):
    """
    Combined_HRV_Analysis.xlsx を読み込み、LF/HF と RMSSD の箱ひげ図を生成する。
    
    :param input_file_path: 結合済みExcelファイルのパス
    :param output_dir: 図を保存するディレクトリ
    :return: 作成したファイルのパスリスト
    """
    input_file_path = input_file_path or DEFAULT_INPUT_FILE_PATH
    output_dir = output_dir or os.path.dirname(input_file_path)

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {input_file_path}")

    os.makedirs(output_dir, exist_ok=True)

    # データの読み込み
    df = pd.read_excel(input_file_path)
    print("データの読み込みに成功しました。")

    saved_files = []

    lf_path = create_boxplot_for_metric(
            df,
            metric_suffix='_LF/HF',
            title='LF/HFの比較（時系列分布）',
            output_filename='LFHF_Boxplot.png',
            output_dir=output_dir
        )
    if lf_path:
        saved_files.append(lf_path)

    rmssd_path = create_boxplot_for_metric(
            df,
            metric_suffix='_RMSSD',
            title='RMSSDの比較（時系列分布）',
            output_filename='RMSSD_Boxplot.png',
            output_dir=output_dir
        )
    if rmssd_path:
        saved_files.append(rmssd_path)

    print("\nすべてのグラフ作成が完了しました。")
    return saved_files


def _create_boxplot_for_long_df(df, metric_col, title, output_filename, output_dir):
    if 'Condition' not in df.columns or metric_col not in df.columns:
        print(f"警告: 長形式データに必要な列がありません ({metric_col})")
        return None

    plot_df = df[['Condition', metric_col]].dropna().copy()
    if plot_df.empty:
        print(f"警告: 長形式データに {metric_col} の有効な値がありません。")
        return None

    plot_df['ConditionLabel'] = plot_df['Condition'].map(conditions).fillna(plot_df['Condition'])
    available_conditions = [key for key in CONDITION_ORDER if key in set(plot_df['Condition'])]
    order_labels = [conditions[key] for key in available_conditions]
    if not order_labels:
        order_labels = list(plot_df['ConditionLabel'].unique())
    palette = _condition_palette(order_labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        x='ConditionLabel',
        y=metric_col,
        data=plot_df,
        palette=palette,
        ax=ax,
        showfliers=False,
        width=0.5,
        order=order_labels
    )

    legend_patches = [mpatches.Patch(color=palette[label], label=label) for label in order_labels]
    ax.legend(handles=legend_patches, title="条件", loc='upper right')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("条件", fontsize=14)
    ax.set_ylabel(metric_col, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存完了: {save_path}")
    return save_path


def generate_box_plots_from_long_df(df, output_dir, filename_prefix="AllSubjects"):
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    lf = _create_boxplot_for_long_df(
        df,
        metric_col='LF/HF',
        title='LF/HFの比較（全被験者）',
        output_filename=f'{filename_prefix}_LFHF_Boxplot.png',
        output_dir=output_dir
    )
    if lf:
        saved.append(lf)

    rmssd = _create_boxplot_for_long_df(
        df,
        metric_col='RMSSD',
        title='RMSSDの比較（全被験者）',
        output_filename=f'{filename_prefix}_RMSSD_Boxplot.png',
        output_dir=output_dir
    )
    if rmssd:
        saved.append(rmssd)

    return saved

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

if __name__ == "__main__":
    try:
        generate_box_plots(DEFAULT_INPUT_FILE_PATH, DEFAULT_OUTPUT_DIR)
    except Exception as exc:
        print(f"箱ひげ図の作成に失敗しました: {exc}")
