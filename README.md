# HRV Analysis from ECG Data

心拍フィードバック制御研究のためのECGデータからHRV（心拍変動）を解析するツール群です。

## 起動方法

### 統合GUI（推奨）

すべての解析機能を統合したGUIアプリケーションです。

```bash
python integrated_gui.py
```

### 従来のGUI

基本的なECG解析のみのシンプルなGUIです。

```bash
python Main.py
```

---

## 統合GUI の機能

### 1. ECG解析タブ

単一のECGファイルからHRV指標を算出します。

**入力データ形式:**
```csv
Timestamp,Device Timestamp (ns),ECG Value (uV)
2024-12-25 12:31:36.000,123456789,150.5
```

**パラメータ:**
- サンプリング周波数: ECGセンサーのサンプリングレート（デフォルト: 130Hz）
- 解析開始/終了(秒): 解析区間を限定する場合に指定
- ラベル: 出力ファイル名に使用

**出力:**
- LF/HF時系列グラフ
- RMSSD時系列グラフ
- SDNN時系列グラフ
- Excelファイル（時系列データ + 全体統計）

---

### 2. バッチ解析タブ

複数条件のECGファイルを一括処理します。

**使い方:**
1. 条件ごとにECGファイルを選択（Fixed, HRF, Sin, HRF2_PID, HRF2_Adaptive, HRF2_GS）
2. または「フォルダから自動検出」ボタンでファイル名から自動マッピング
3. 被験者IDと解析パラメータを設定
4. 出力ディレクトリを指定して実行

**出力:**
- 各条件のHRV解析結果（Excel）
- 統合結果ファイル（Combined_HRV_Analysis.xlsx）
- 時系列グラフ（PNG）

---

### 3. 箱ひげ図タブ

バッチ解析結果から条件間比較の箱ひげ図を生成します。

**入力:**
- Combined_HRV_Analysis.xlsx（バッチ解析の出力）

**出力:**
- LF/HF箱ひげ図
- RMSSD箱ひげ図
- SDNN箱ひげ図
- 記述統計サマリー

---

### 4. 制御性能評価タブ

心拍数制御の追従性能を評価します。

**入力データ形式:**
```csv
Timestamp,Heart Rate (BPM)
2024-12-25 12:31:36.000,72.5
```

**パラメータ:**
- 目標心拍数 (BPM): 制御の目標値
- 許容誤差 (BPM): 目標達成と判定する許容範囲（デフォルト: ±5BPM）

**算出指標:**
| 指標 | 説明 |
|------|------|
| RMSE | 目標値からの二乗平均平方根誤差 |
| MAE | 平均絶対誤差 |
| 制御率 | 許容範囲内にあった時間の割合 |
| 収束率 | 時間経過に伴う誤差減少率 |
| 立ち上がり時間 | 目標の90%に達するまでの時間 |
| 整定時間 | 許容範囲内に収束するまでの時間 |
| オーバーシュート | 最大超過量 |

---

### 5. 高度HRV解析タブ

非線形HRV指標を含む詳細な解析を行います。

**入力:**
- RR間隔データ（ms単位のCSV）
- または ECGデータ（「ECG Value (uV)」列を含むCSV）

**算出指標:**

| カテゴリ | 指標 |
|----------|------|
| 時間領域 | Mean RR, Mean HR, SDNN, RMSSD, pNN50, pNN20 |
| 周波数領域 | Total Power, LF Power, HF Power, LF/HF, LF nu, HF nu |
| 非線形 | SD1, SD2, SD2/SD1 (Poincaré), Sample Entropy, DFA α1, DFA α2 |

---

### 6. 統計検定タブ

条件間の統計的有意差を検定します。

**入力データ形式:**
長形式（Long format）のデータ:
```csv
Subject,Condition,Value
No1,Fixed,1.25
No1,HRF,0.98
No2,Fixed,1.32
```

**検定タイプ:**
- 反復測定一元配置ANOVA: 同一被験者の条件間比較
- 一元配置ANOVA: 独立群の比較
- Friedman検定: ノンパラメトリックな反復測定検定

**出力:**
- 記述統計（各条件のn, M, SD）
- 検定統計量（F値/χ²値）
- p値
- 効果量

---

### 7. 相関分析タブ

HRV指標と主観評価の相関を分析します。

**入力:**
1. HRVデータファイル: RMSSD, SDNN, LF/HF等の列を含むExcel/CSV
2. 主観評価ファイル: アンケート結果のExcel/CSV

**設定:**
- 結合キー: 両ファイルを紐付ける列名（例: Subject）
- HRV列: 分析対象のHRV指標（カンマ区切り）
- 主観評価列: 分析対象の主観評価項目（カンマ区切り）

**出力:**
- Pearson相関係数とp値
- Spearman順位相関係数とp値
- 相関の解釈（弱い/中程度/強い）

---

## サンプルデータ

以下のサンプルデータで動作確認できます:

- ECGデータ: `/Users/user/Research/HCS_ver4.0/logs20251225/h10_ecg_session_*.csv`
- HRデータ: `/Users/user/Research/HCS_ver4.0/logs20251225/h10_hr_session_*.csv`
- 主観評価: `/Users/user/Research/HCS_ver4.0/実験後アンケート20251225.xlsx`

---

## ファイル構成

```
Analys/
├── integrated_gui.py    # 統合GUI（メイン）
├── AnalysisECG.py       # ECG解析コアモジュール
├── Analyzer.py          # バッチ解析モジュール
├── advanced_gui.py      # 高度解析GUI（単体）
├── Main.py              # 従来のシンプルGUI
├── advanced_analysis/   # 高度解析モジュール群
│   ├── __init__.py
│   ├── control_metrics.py  # 制御性能評価
│   ├── advanced_hrv.py     # 非線形HRV解析
│   ├── statistics.py       # 統計検定
│   └── correlation.py      # 相関分析
└── result_batch/        # 解析結果出力先
```

---

## 依存ライブラリ

```bash
pip install numpy pandas scipy matplotlib openpyxl
pip install japanize-matplotlib  # 日本語フォント対応（オプション）
```

---

## 条件ラベルと色

| 条件 | 日本語ラベル | グラフ色 |
|------|-------------|---------|
| Fixed | 固定会話 | lightcoral |
| Sin | 正弦波 | lightblue |
| HRF | HRフィードバック | lightgreen |
| HRF2_PID | PID制御 | lightyellow |
| HRF2_Adaptive | 適応制御 | plum |
| HRF2_GS | ゲインスケジューリング | lightsalmon |
| HRF2_Robust | ロバスト制御 | lightcyan |

---

## トラブルシューティング

### 日本語が文字化けする
```bash
pip install japanize-matplotlib
```

### ECGデータが読み込めない
- ファイルの列名を確認: `ECG Value (uV)` または `ECG` が必要
- サンプリング周波数がECGセンサーと一致しているか確認

### 統計検定でエラー
- データが長形式（Long format）になっているか確認
- 被験者ID、条件名、値の列名が正しく設定されているか確認

---

## ライセンス

Private use only.
