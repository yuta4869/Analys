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
input_file_path = "/Users/user/Research/Analys/result_batch/Combined_HRV_Analysis.xlsx"

# 保存先のディレクトリ（指定がなければ入力ファイルと同じ場所）
output_dir = os.path.dirname(input_file_path)

# 条件と列名の対応定義
# Excelの列名に含まれるキーワード (key) と、グラフに表示する日本語ラベル (label)
conditions = {
    'Fixed': '固定会話',
    'HRF': '調整会話',
    'Sin': '正弦波'
}

# グラフの色設定
colors = {
    '固定会話': 'lightcoral', 
    '調整会話': 'lightyellow', 
    '正弦波': 'lightblue'
}

# ---------------------------------------------------------
# グラフ描画関数
# ---------------------------------------------------------

def create_boxplot_for_metric(df, metric_suffix, title, output_filename):
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
    # showfliers=False: 外れ値を箱ひげ図自体では非表示にする（ストリッププロットで重ねるため）
    sns.boxplot(
        x='Condition', 
        y='Value', 
        data=df_melted, 
        palette=colors, 
        ax=ax, 
        showfliers=False,
        width=0.5
    )
    
    # 2. ストリッププロット（データ点を重ねて表示）
    # jitter=True: 点が重ならないように少し散らす
    sns.stripplot(
        x='Condition', 
        y='Value', 
        data=df_melted, 
        color='black', 
        alpha=0.3, # 透明度
        jitter=True, 
        size=3,
        ax=ax
    )

    # 凡例の作成
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in colors.items() if label in plot_data.columns]
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

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

if __name__ == "__main__":
    # ファイルの存在確認
    if not os.path.exists(input_file_path):
        print(f"エラー: ファイルが見つかりません -> {input_file_path}")
        print("パスが正しいか、またはBatchAnalysisECG.pyを実行してファイルが生成されているか確認してください。")
        exit()

    # データの読み込み
    try:
        df = pd.read_excel(input_file_path)
        print("データの読み込みに成功しました。")
    except Exception as e:
        print(f"データの読み込みに失敗しました: {e}")
        exit()

    # 1. LF/HF の箱ひげ図作成
    # BatchAnalysisECG.pyの出力列名は 'Fixed_LF/HF' などになっているため、サフィックスは '_LF/HF'
    create_boxplot_for_metric(
        df, 
        metric_suffix='_LF/HF', 
        title='LF/HFの比較（時系列分布）', 
        output_filename='LFHF_Boxplot.png'
    )

    # 2. RMSSD の箱ひげ図作成
    create_boxplot_for_metric(
        df, 
        metric_suffix='_RMSSD', 
        title='RMSSDの比較（時系列分布）', 
        output_filename='RMSSD_Boxplot.png'
    )

    print("\nすべてのグラフ作成が完了しました。")