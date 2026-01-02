# advanced_gui.py
"""高度解析GUI

心拍フィードバック制御研究のための統合解析GUIアプリケーション
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional, Dict, List
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

from advanced_analysis import (
    ControlMetricsAnalyzer,
    AdvancedHRVAnalyzer,
    StatisticalAnalyzer,
    CorrelationAnalyzer,
)


class AdvancedAnalysisGUI:
    """高度解析GUIアプリケーション"""

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("高度解析ツール - 心拍フィードバック制御研究")
        self.master.geometry("1200x800")

        # 解析器の初期化
        self.control_analyzer = ControlMetricsAnalyzer()
        self.hrv_analyzer = AdvancedHRVAnalyzer()
        self.stat_analyzer = StatisticalAnalyzer()
        self.corr_analyzer = CorrelationAnalyzer()

        # データ保持
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, any] = {}

        self._build_ui()

    def _build_ui(self):
        """UIを構築"""
        # メインフレーム
        main_frame = ttk.Frame(self.master, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タブコントロール
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 各タブを作成
        self._create_control_metrics_tab()
        self._create_advanced_hrv_tab()
        self._create_statistics_tab()
        self._create_correlation_tab()

    def _create_control_metrics_tab(self):
        """制御性能評価タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="制御性能評価")

        # 左側: 入力設定
        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # ファイル選択
        ttk.Label(left_frame, text="HRデータファイル:").pack(anchor="w")
        self.ctrl_file_var = tk.StringVar()
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.ctrl_file_var, width=40).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="参照", command=self._ctrl_browse_file).pack(side=tk.LEFT, padx=5)

        # パラメータ
        param_frame = ttk.LabelFrame(left_frame, text="パラメータ", padding="5")
        param_frame.pack(fill=tk.X, pady=10)

        ttk.Label(param_frame, text="目標心拍数 (BPM):").grid(row=0, column=0, sticky="w", pady=2)
        self.ctrl_target_hr_var = tk.DoubleVar(value=70.0)
        ttk.Entry(param_frame, textvariable=self.ctrl_target_hr_var, width=10).grid(row=0, column=1, pady=2)

        ttk.Label(param_frame, text="許容誤差 (BPM):").grid(row=1, column=0, sticky="w", pady=2)
        self.ctrl_tolerance_var = tk.DoubleVar(value=5.0)
        ttk.Entry(param_frame, textvariable=self.ctrl_tolerance_var, width=10).grid(row=1, column=1, pady=2)

        ttk.Label(param_frame, text="時間列:").grid(row=2, column=0, sticky="w", pady=2)
        self.ctrl_time_col_var = tk.StringVar(value="Time")
        ttk.Entry(param_frame, textvariable=self.ctrl_time_col_var, width=10).grid(row=2, column=1, pady=2)

        ttk.Label(param_frame, text="HR列:").grid(row=3, column=0, sticky="w", pady=2)
        self.ctrl_hr_col_var = tk.StringVar(value="HR")
        ttk.Entry(param_frame, textvariable=self.ctrl_hr_col_var, width=10).grid(row=3, column=1, pady=2)

        # 実行ボタン
        ttk.Button(left_frame, text="解析実行", command=self._run_control_analysis).pack(pady=10)

        # 右側: 結果表示
        right_frame = ttk.LabelFrame(tab, text="解析結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ctrl_result_text = scrolledtext.ScrolledText(right_frame, height=20, width=60)
        self.ctrl_result_text.pack(fill=tk.BOTH, expand=True)

        # グラフエリア
        self.ctrl_fig_frame = ttk.Frame(right_frame)
        self.ctrl_fig_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def _create_advanced_hrv_tab(self):
        """高度HRV解析タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="高度HRV解析")

        # 左側: 入力設定
        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # ファイル選択
        ttk.Label(left_frame, text="RR間隔/ECGデータ:").pack(anchor="w")
        self.hrv_file_var = tk.StringVar()
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.hrv_file_var, width=40).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="参照", command=self._hrv_browse_file).pack(side=tk.LEFT, padx=5)

        # データタイプ
        ttk.Label(left_frame, text="データタイプ:").pack(anchor="w", pady=(10, 0))
        self.hrv_data_type_var = tk.StringVar(value="rr")
        ttk.Radiobutton(left_frame, text="RR間隔 (ms)", variable=self.hrv_data_type_var, value="rr").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="ECG信号", variable=self.hrv_data_type_var, value="ecg").pack(anchor="w")

        # オプション
        self.hrv_nonlinear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="非線形指標を計算", variable=self.hrv_nonlinear_var).pack(anchor="w", pady=5)

        # 実行ボタン
        ttk.Button(left_frame, text="解析実行", command=self._run_hrv_analysis).pack(pady=10)

        # 右側: 結果表示
        right_frame = ttk.LabelFrame(tab, text="解析結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.hrv_result_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.hrv_result_text.pack(fill=tk.BOTH, expand=True)

    def _create_statistics_tab(self):
        """統計検定タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="統計検定")

        # 左側: 入力設定
        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # ファイル選択
        ttk.Label(left_frame, text="データファイル (Excel/CSV):").pack(anchor="w")
        self.stat_file_var = tk.StringVar()
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.stat_file_var, width=40).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="参照", command=self._stat_browse_file).pack(side=tk.LEFT, padx=5)

        # 列設定
        col_frame = ttk.LabelFrame(left_frame, text="列設定", padding="5")
        col_frame.pack(fill=tk.X, pady=10)

        ttk.Label(col_frame, text="被験者列:").grid(row=0, column=0, sticky="w", pady=2)
        self.stat_subject_col_var = tk.StringVar(value="Subject")
        ttk.Entry(col_frame, textvariable=self.stat_subject_col_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(col_frame, text="条件列:").grid(row=1, column=0, sticky="w", pady=2)
        self.stat_condition_col_var = tk.StringVar(value="Condition")
        ttk.Entry(col_frame, textvariable=self.stat_condition_col_var, width=15).grid(row=1, column=1, pady=2)

        ttk.Label(col_frame, text="値列:").grid(row=2, column=0, sticky="w", pady=2)
        self.stat_value_col_var = tk.StringVar(value="Value")
        ttk.Entry(col_frame, textvariable=self.stat_value_col_var, width=15).grid(row=2, column=1, pady=2)

        # 検定タイプ
        ttk.Label(left_frame, text="検定タイプ:").pack(anchor="w", pady=(10, 0))
        self.stat_test_type_var = tk.StringVar(value="rm_anova")
        ttk.Radiobutton(left_frame, text="反復測定ANOVA", variable=self.stat_test_type_var, value="rm_anova").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="一元配置ANOVA", variable=self.stat_test_type_var, value="anova").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="Friedman検定", variable=self.stat_test_type_var, value="friedman").pack(anchor="w")

        # 有意水準
        ttk.Label(left_frame, text="有意水準:").pack(anchor="w", pady=(10, 0))
        self.stat_alpha_var = tk.DoubleVar(value=0.05)
        ttk.Entry(left_frame, textvariable=self.stat_alpha_var, width=10).pack(anchor="w")

        # 実行ボタン
        ttk.Button(left_frame, text="検定実行", command=self._run_stat_analysis).pack(pady=10)

        # 右側: 結果表示
        right_frame = ttk.LabelFrame(tab, text="検定結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.stat_result_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.stat_result_text.pack(fill=tk.BOTH, expand=True)

    def _create_correlation_tab(self):
        """相関分析タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="相関分析")

        # 左側: 入力設定
        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # HRVデータ
        ttk.Label(left_frame, text="HRVデータファイル:").pack(anchor="w")
        self.corr_hrv_file_var = tk.StringVar()
        hrv_frame = ttk.Frame(left_frame)
        hrv_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(hrv_frame, textvariable=self.corr_hrv_file_var, width=35).pack(side=tk.LEFT)
        ttk.Button(hrv_frame, text="参照", command=lambda: self._corr_browse_file('hrv')).pack(side=tk.LEFT, padx=5)

        # 主観評価データ
        ttk.Label(left_frame, text="主観評価データファイル:").pack(anchor="w", pady=(10, 0))
        self.corr_subj_file_var = tk.StringVar()
        subj_frame = ttk.Frame(left_frame)
        subj_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(subj_frame, textvariable=self.corr_subj_file_var, width=35).pack(side=tk.LEFT)
        ttk.Button(subj_frame, text="参照", command=lambda: self._corr_browse_file('subj')).pack(side=tk.LEFT, padx=5)

        # 列設定
        col_frame = ttk.LabelFrame(left_frame, text="列設定", padding="5")
        col_frame.pack(fill=tk.X, pady=10)

        ttk.Label(col_frame, text="結合キー:").grid(row=0, column=0, sticky="w", pady=2)
        self.corr_merge_col_var = tk.StringVar(value="Subject")
        ttk.Entry(col_frame, textvariable=self.corr_merge_col_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(col_frame, text="HRV列 (カンマ区切り):").grid(row=1, column=0, sticky="w", pady=2)
        self.corr_hrv_cols_var = tk.StringVar(value="RMSSD,SDNN,LF_HF")
        ttk.Entry(col_frame, textvariable=self.corr_hrv_cols_var, width=20).grid(row=1, column=1, pady=2)

        ttk.Label(col_frame, text="主観評価列 (カンマ区切り):").grid(row=2, column=0, sticky="w", pady=2)
        self.corr_subj_cols_var = tk.StringVar(value="Relaxation,Comfort")
        ttk.Entry(col_frame, textvariable=self.corr_subj_cols_var, width=20).grid(row=2, column=1, pady=2)

        # 相関タイプ
        ttk.Label(left_frame, text="相関係数:").pack(anchor="w", pady=(10, 0))
        self.corr_method_var = tk.StringVar(value="pearson")
        ttk.Radiobutton(left_frame, text="Pearson", variable=self.corr_method_var, value="pearson").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="Spearman", variable=self.corr_method_var, value="spearman").pack(anchor="w")

        # 実行ボタン
        ttk.Button(left_frame, text="相関分析実行", command=self._run_corr_analysis).pack(pady=10)

        # 右側: 結果表示
        right_frame = ttk.LabelFrame(tab, text="分析結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.corr_result_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.corr_result_text.pack(fill=tk.BOTH, expand=True)

    # --- 制御性能評価 ---
    def _ctrl_browse_file(self):
        filepath = filedialog.askopenfilename(
            title="HRデータファイルを選択",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")]
        )
        if filepath:
            self.ctrl_file_var.set(filepath)

    def _run_control_analysis(self):
        filepath = self.ctrl_file_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください")
            return

        try:
            # データ読み込み
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            target_hr = self.ctrl_target_hr_var.get()
            time_col = self.ctrl_time_col_var.get()
            hr_col = self.ctrl_hr_col_var.get()

            # 解析器の許容誤差を設定
            self.control_analyzer.tolerance_bpm = self.ctrl_tolerance_var.get()

            # 解析実行
            metrics = self.control_analyzer.analyze_from_dataframe(
                df, time_col=time_col, hr_col=hr_col, target_hr=target_hr
            )

            # 結果表示
            self.ctrl_result_text.delete(1.0, tk.END)
            self.ctrl_result_text.insert(tk.END, "=== 制御性能評価結果 ===\n\n")
            self.ctrl_result_text.insert(tk.END, f"目標心拍数: {target_hr} BPM\n")
            self.ctrl_result_text.insert(tk.END, f"データ点数: {len(df)}\n\n")

            for key, value in metrics.to_dict().items():
                if value is not None:
                    if isinstance(value, float):
                        self.ctrl_result_text.insert(tk.END, f"{key}: {value:.3f}\n")
                    else:
                        self.ctrl_result_text.insert(tk.END, f"{key}: {value}\n")
                else:
                    self.ctrl_result_text.insert(tk.END, f"{key}: N/A\n")

            # グラフ描画
            self._plot_control_graph(df, time_col, hr_col, target_hr)

        except Exception as e:
            messagebox.showerror("エラー", f"解析中にエラーが発生しました:\n{e}")

    def _plot_control_graph(self, df, time_col, hr_col, target_hr):
        """制御性能グラフを描画"""
        # 既存のウィジェットをクリア
        for widget in self.ctrl_fig_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df[time_col], df[hr_col], 'b-', label='実測HR', linewidth=1)
        ax.axhline(y=target_hr, color='r', linestyle='--', label=f'目標HR ({target_hr} BPM)')
        ax.axhline(y=target_hr + 5, color='g', linestyle=':', alpha=0.5, label='±5 BPM')
        ax.axhline(y=target_hr - 5, color='g', linestyle=':', alpha=0.5)
        ax.fill_between(df[time_col], target_hr - 5, target_hr + 5, alpha=0.1, color='green')

        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('HR (BPM)')
        ax.set_title('心拍数追従グラフ')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, self.ctrl_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- 高度HRV解析 ---
    def _hrv_browse_file(self):
        filepath = filedialog.askopenfilename(
            title="RR間隔/ECGデータを選択",
            filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("All", "*.*")]
        )
        if filepath:
            self.hrv_file_var.set(filepath)

    def _run_hrv_analysis(self):
        filepath = self.hrv_file_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください")
            return

        try:
            # データ読み込み
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            data_type = self.hrv_data_type_var.get()
            compute_nonlinear = self.hrv_nonlinear_var.get()

            if data_type == 'rr':
                # RR間隔データ
                if 'RR' in df.columns:
                    rr = df['RR'].values
                elif 'rr' in df.columns:
                    rr = df['rr'].values
                else:
                    rr = df.iloc[:, 0].values

                metrics = self.hrv_analyzer.calculate_metrics(rr, compute_nonlinear)
            else:
                # ECGデータ
                if 'ECG' in df.columns:
                    ecg = df['ECG'].values
                else:
                    ecg = df.iloc[:, 0].values

                metrics = self.hrv_analyzer.analyze_from_ecg(ecg, compute_nonlinear=compute_nonlinear)

            # 結果表示
            self.hrv_result_text.delete(1.0, tk.END)
            self.hrv_result_text.insert(tk.END, "=== 高度HRV解析結果 ===\n\n")

            self.hrv_result_text.insert(tk.END, "【時間領域指標】\n")
            time_keys = ['Mean RR (ms)', 'Mean HR (BPM)', 'SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)', 'pNN20 (%)']
            for key in time_keys:
                value = metrics.to_dict().get(key, 'N/A')
                if isinstance(value, float):
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value:.2f}\n")
                else:
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value}\n")

            self.hrv_result_text.insert(tk.END, "\n【周波数領域指標】\n")
            freq_keys = ['Total Power (ms²)', 'VLF Power (ms²)', 'LF Power (ms²)', 'HF Power (ms²)', 'LF/HF', 'LF nu', 'HF nu']
            for key in freq_keys:
                value = metrics.to_dict().get(key, 'N/A')
                if isinstance(value, float):
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value:.2f}\n")
                else:
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value}\n")

            self.hrv_result_text.insert(tk.END, "\n【非線形指標】\n")
            nonlinear_keys = ['SD1 (ms)', 'SD2 (ms)', 'SD2/SD1', 'Sample Entropy', 'DFA α1', 'DFA α2']
            for key in nonlinear_keys:
                value = metrics.to_dict().get(key, 'N/A')
                if value is not None and isinstance(value, float):
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value:.3f}\n")
                else:
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value}\n")

        except Exception as e:
            messagebox.showerror("エラー", f"解析中にエラーが発生しました:\n{e}")

    # --- 統計検定 ---
    def _stat_browse_file(self):
        filepath = filedialog.askopenfilename(
            title="データファイルを選択",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")]
        )
        if filepath:
            self.stat_file_var.set(filepath)

    def _run_stat_analysis(self):
        filepath = self.stat_file_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください")
            return

        try:
            # データ読み込み
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            subject_col = self.stat_subject_col_var.get()
            condition_col = self.stat_condition_col_var.get()
            value_col = self.stat_value_col_var.get()
            test_type = self.stat_test_type_var.get()

            self.stat_analyzer.alpha = self.stat_alpha_var.get()

            self.stat_result_text.delete(1.0, tk.END)
            self.stat_result_text.insert(tk.END, "=== 統計検定結果 ===\n\n")

            # 記述統計
            self.stat_result_text.insert(tk.END, "【記述統計】\n")
            for cond in df[condition_col].unique():
                cond_data = df[df[condition_col] == cond][value_col].values
                stats = self.stat_analyzer.summary_statistics(cond_data)
                self.stat_result_text.insert(tk.END, f"\n{cond}:\n")
                self.stat_result_text.insert(tk.END, f"  n={stats['n']}, M={stats['mean']:.2f}, SD={stats['std']:.2f}\n")
                self.stat_result_text.insert(tk.END, f"  Median={stats['median']:.2f}, IQR={stats['iqr']:.2f}\n")

            self.stat_result_text.insert(tk.END, "\n【検定結果】\n")

            if test_type == 'rm_anova':
                result = self.stat_analyzer.repeated_measures_anova(
                    df, subject_col, condition_col, value_col
                )
                self.stat_result_text.insert(tk.END, "\n反復測定一元配置ANOVA:\n")
            elif test_type == 'anova':
                groups = {cond: df[df[condition_col] == cond][value_col].values
                          for cond in df[condition_col].unique()}
                result = self.stat_analyzer.one_way_anova(groups)
                self.stat_result_text.insert(tk.END, "\n一元配置ANOVA:\n")
            else:
                result = self.stat_analyzer.friedman_test(
                    df, subject_col, condition_col, value_col
                )
                self.stat_result_text.insert(tk.END, "\nFriedman検定:\n")

            for key, value in result.to_dict().items():
                if isinstance(value, float):
                    self.stat_result_text.insert(tk.END, f"  {key}: {value:.4f}\n")
                else:
                    self.stat_result_text.insert(tk.END, f"  {key}: {value}\n")

            # 効果量の解釈
            if hasattr(result, 'eta_squared'):
                interpretation = self.stat_analyzer.effect_size_interpretation(
                    result.eta_squared, 'eta_squared'
                )
                self.stat_result_text.insert(tk.END, f"\n効果量の解釈: {interpretation}\n")

            # 事後検定
            if result.post_hoc:
                self.stat_result_text.insert(tk.END, "\n【事後検定（多重比較）】\n")
                for comparison, stats in result.post_hoc.items():
                    sig = "***" if stats['significant'] else ""
                    self.stat_result_text.insert(tk.END, f"\n{comparison}:\n")
                    self.stat_result_text.insert(tk.END, f"  p = {stats['p_adjusted']:.4f} {sig}\n")
                    if 'cohens_d' in stats:
                        d = stats['cohens_d']
                        interp = self.stat_analyzer.effect_size_interpretation(d, 'cohens_d')
                        self.stat_result_text.insert(tk.END, f"  Cohen's d = {d:.3f} ({interp})\n")

        except Exception as e:
            messagebox.showerror("エラー", f"検定中にエラーが発生しました:\n{e}")

    # --- 相関分析 ---
    def _corr_browse_file(self, file_type):
        filepath = filedialog.askopenfilename(
            title=f"{'HRV' if file_type == 'hrv' else '主観評価'}データを選択",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")]
        )
        if filepath:
            if file_type == 'hrv':
                self.corr_hrv_file_var.set(filepath)
            else:
                self.corr_subj_file_var.set(filepath)

    def _run_corr_analysis(self):
        hrv_path = self.corr_hrv_file_var.get()
        subj_path = self.corr_subj_file_var.get()

        if not hrv_path or not subj_path:
            messagebox.showerror("エラー", "両方のファイルを選択してください")
            return

        try:
            # データ読み込み
            if hrv_path.endswith('.csv'):
                hrv_df = pd.read_csv(hrv_path)
            else:
                hrv_df = pd.read_excel(hrv_path)

            if subj_path.endswith('.csv'):
                subj_df = pd.read_csv(subj_path)
            else:
                subj_df = pd.read_excel(subj_path)

            merge_col = self.corr_merge_col_var.get()
            hrv_cols = [c.strip() for c in self.corr_hrv_cols_var.get().split(',')]
            subj_cols = [c.strip() for c in self.corr_subj_cols_var.get().split(',')]

            # 相関分析実行
            result_df = self.corr_analyzer.analyze_hrv_subjective_relation(
                hrv_df, subj_df, hrv_cols, subj_cols, merge_col
            )

            # 結果表示
            self.corr_result_text.delete(1.0, tk.END)
            self.corr_result_text.insert(tk.END, "=== HRV-主観評価 相関分析結果 ===\n\n")

            if result_df.empty:
                self.corr_result_text.insert(tk.END, "分析できるデータがありませんでした。\n")
                return

            for _, row in result_df.iterrows():
                self.corr_result_text.insert(tk.END, f"\n{row['HRV指標']} vs {row['主観評価']}:\n")
                self.corr_result_text.insert(tk.END, f"  Pearson r = {row['Pearson r']:.3f} (p = {row['Pearson p']:.4f})\n")
                self.corr_result_text.insert(tk.END, f"  Spearman ρ = {row['Spearman ρ']:.3f} (p = {row['Spearman p']:.4f})\n")
                self.corr_result_text.insert(tk.END, f"  n = {row['n']}, 解釈: {row['解釈']}\n")

                if row['Pearson p'] < 0.05:
                    self.corr_result_text.insert(tk.END, "  *** 有意な相関 ***\n")

        except Exception as e:
            messagebox.showerror("エラー", f"分析中にエラーが発生しました:\n{e}")


def main():
    """メイン関数"""
    root = tk.Tk()
    app = AdvancedAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
