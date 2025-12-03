import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib # 日本語表示に対応させる
import matplotlib.patches as mpatches # 凡例を手動で作成するためにインポート

# CSVファイルを読み込む
# このスクリプトと同じ階層に 'data.csv' を配置してください

# try:
#     df = pd.read_csv('data.csv')
# except FileNotFoundError:
#     print("エラー: 'data.csv' が見つかりません。")
#     print("スクリプトと同じディレクトリにCSVファイルがあるか確認してください。")
#     exit()


# Data extracted from the LaTeX table
data = {

    '固定会話': [0.87, 0.78, 1.54, 2.03, 1.10, 1.27],
    '調整会話': [1.00, 0.60, 1.82, 1.53, 1.03, 1.05],
    '正弦波': [1.01, 0.61, 1.16, 1.52, 1.00, 1.18]
}
# Create a DataFrame
df = pd.DataFrame(data)


# グラフの描画設定
# axオブジェクトを取得して、グラフの各要素を直接操作します
fig, ax = plt.subplots(figsize=(8, 6))

# カラーパレットの定義
colors = {'固定会話': 'lightcoral', '調整会話': 'lightyellow', '正弦波': 'lightblue'}

# # データフレームから直接箱ひげ図を作成
# sns.boxplot(data=df, palette=colors)



# データを「縦長」の形式に変換
# これにより、seabornが凡例を自動的に作成しやすくなります。
df_melted = df.melt(var_name='種別', value_name='値')

# データフレームから箱ひげ図を作成
# ax=ax を指定して、操作対象の軸を明示します
sns.boxplot(x='種別', y='値', hue='種別', data=df_melted, palette=colors, dodge=False, ax=ax)

# x軸のラベルを削除
ax.set_xlabel('')
ax.set_xticklabels([])



# タイトルとy軸ラベルの設定
ax.set_title('LF/HF（5分間）')
ax.set_ylabel('Scale')



# 【修正点】凡例を手動で作成し、表示します
# 元のデータフレームの列の順序で凡例項目を作成します
legend_patches = [mpatches.Patch(color=colors[label], label=label) for label in df.columns]
ax.legend(handles=legend_patches, title='')



# グリッドを追加して見やすくする
ax.grid(axis='y', linestyle='--', alpha=0.7)

# プロットを画像として保存
plt.savefig('box_plot_from_csv.png')

print("CSVファイルから箱ひげ図を正常に作成し、'box_plot_from_csv.png' として保存しました。")

# グラフを表示（ローカル環境で実行する場合）
# plt.show()
