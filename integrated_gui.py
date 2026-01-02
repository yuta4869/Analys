# integrated_gui.py
"""統合解析GUI

Analysフォルダ内の全ての解析機能を統合したGUIアプリケーション:
- ECG解析（RRI算出、HRV指標計算）
- バッチ解析（複数ファイル一括処理）
- 箱ひげ図生成
- 制御性能評価
- 高度HRV解析（非線形指標）
- 統計検定
- 相関分析
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional, Dict, List
import threading
import queue

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# 基本解析モジュール
from AnalysisECG import (
    calculate_hrv_indices,
    ecg_to_rri,
    ECG_CONDITIONS,
    CONDITION_LABELS,
    CONDITION_COLORS,
)
from Analyzer import (
    run_batch_analysis,
    generate_box_plots,
    generate_summary_statistics,
    compare_subjects,
)

# 高度解析モジュール
from advanced_analysis import (
    ControlMetrics,
    ControlMetricsAnalyzer,
    AdvancedHRVMetrics,
    AdvancedHRVAnalyzer,
    StatisticalAnalyzer,
    CorrelationAnalyzer,
)


class IntegratedAnalysisGUI:
    """統合解析GUIアプリケーション"""

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("心拍変動解析ツール - 統合版")
        self.master.geometry("1400x900")

        # 解析器の初期化
        self.control_analyzer = ControlMetricsAnalyzer()
        self.hrv_analyzer = AdvancedHRVAnalyzer()
        self.stat_analyzer = StatisticalAnalyzer()
        self.corr_analyzer = CorrelationAnalyzer()

        # データ保持
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        self.batch_files: Dict[str, str] = {}

        # スタイル設定
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        """スタイルを設定"""
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Result.TLabel', font=('Courier', 10))

    def _build_ui(self):
        """UIを構築"""
        # メインフレーム
        main_frame = ttk.Frame(self.master, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タブコントロール
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 各タブを作成
        self._create_ecg_analysis_tab()
        self._create_batch_analysis_tab()
        self._create_boxplot_tab()
        self._create_control_metrics_tab()
        self._create_advanced_hrv_tab()
        self._create_statistics_tab()
        self._create_correlation_tab()

        # ステータスバー
        self.status_var = tk.StringVar(value="準備完了")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ========================================
    # ECG解析タブ
    # ========================================
    def _create_ecg_analysis_tab(self):
        """ECG解析タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="ECG解析")

        # 上部: 入力設定
        input_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # ファイル選択
        file_row = ttk.Frame(input_frame)
        file_row.pack(fill=tk.X, pady=5)
        ttk.Label(file_row, text="ECGファイル:").pack(side=tk.LEFT)
        self.ecg_file_var = tk.StringVar()
        ttk.Entry(file_row, textvariable=self.ecg_file_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_row, text="参照", command=self._ecg_browse_file).pack(side=tk.LEFT)

        # パラメータ
        param_frame = ttk.Frame(input_frame)
        param_frame.pack(fill=tk.X, pady=5)

        ttk.Label(param_frame, text="サンプリング周波数:").grid(row=0, column=0, sticky="w", padx=5)
        self.ecg_fs_var = tk.IntVar(value=130)
        ttk.Entry(param_frame, textvariable=self.ecg_fs_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="解析開始(秒):").grid(row=0, column=2, sticky="w", padx=5)
        self.ecg_start_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.ecg_start_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Label(param_frame, text="解析終了(秒):").grid(row=0, column=4, sticky="w", padx=5)
        self.ecg_end_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.ecg_end_var, width=8).grid(row=0, column=5, padx=5)

        ttk.Label(param_frame, text="ラベル:").grid(row=0, column=6, sticky="w", padx=5)
        self.ecg_label_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.ecg_label_var, width=15).grid(row=0, column=7, padx=5)

        # 出力設定
        output_row = ttk.Frame(input_frame)
        output_row.pack(fill=tk.X, pady=5)
        ttk.Label(output_row, text="出力ディレクトリ:").pack(side=tk.LEFT)
        self.ecg_output_var = tk.StringVar()
        ttk.Entry(output_row, textvariable=self.ecg_output_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_row, text="参照", command=self._ecg_browse_output).pack(side=tk.LEFT)

        # 実行ボタン
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="解析実行", command=self._run_ecg_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="結果をクリア", command=self._clear_ecg_results).pack(side=tk.LEFT, padx=5)

        # 結果表示
        result_frame = ttk.LabelFrame(tab, text="解析結果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左: テキスト結果
        left_result = ttk.Frame(result_frame)
        left_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ecg_result_text = scrolledtext.ScrolledText(left_result, height=20, width=50)
        self.ecg_result_text.pack(fill=tk.BOTH, expand=True)

        # 右: グラフ
        right_result = ttk.Frame(result_frame)
        right_result.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.ecg_fig_frame = ttk.Frame(right_result)
        self.ecg_fig_frame.pack(fill=tk.BOTH, expand=True)

    def _ecg_browse_file(self):
        filepath = filedialog.askopenfilename(
            title="ECGファイルを選択",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if filepath:
            self.ecg_file_var.set(filepath)
            # ラベルを自動設定
            if not self.ecg_label_var.get():
                self.ecg_label_var.set(Path(filepath).stem)

    def _ecg_browse_output(self):
        dirpath = filedialog.askdirectory(title="出力ディレクトリを選択")
        if dirpath:
            self.ecg_output_var.set(dirpath)

    def _run_ecg_analysis(self):
        filepath = self.ecg_file_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください")
            return

        self.status_var.set("解析中...")
        self.master.update()

        try:
            # パラメータ取得
            fs = self.ecg_fs_var.get()
            start_offset = float(self.ecg_start_var.get()) if self.ecg_start_var.get() else None
            end_offset = float(self.ecg_end_var.get()) if self.ecg_end_var.get() else None
            label = self.ecg_label_var.get() or Path(filepath).stem
            output_dir = self.ecg_output_var.get()

            # 出力パス設定
            excel_path = os.path.join(output_dir, f"{label}_hrv_result.xlsx") if output_dir else None
            csv_path = os.path.join(output_dir, f"{label}_overall.csv") if output_dir else None

            # 解析実行
            sliding_df, overall_lfhf = calculate_hrv_indices(
                filepath,
                label=label,
                fs=fs,
                output_excel_path=excel_path,
                output_csv_path=csv_path,
                analysis_start_offset=start_offset,
                analysis_end_offset=end_offset,
            )

            # 結果表示
            self.ecg_result_text.delete(1.0, tk.END)
            self.ecg_result_text.insert(tk.END, f"=== ECG解析結果: {label} ===\n\n")

            if sliding_df is not None:
                self.ecg_result_text.insert(tk.END, f"データ点数: {len(sliding_df)}\n")
                self.ecg_result_text.insert(tk.END, f"全体LF/HF: {overall_lfhf:.4f}\n\n")

                # 統計情報
                self.ecg_result_text.insert(tk.END, "【時系列統計】\n")
                for col in ['LF/HF', 'RMSSD', 'SDNN']:
                    if col in sliding_df.columns:
                        mean = sliding_df[col].mean()
                        std = sliding_df[col].std()
                        self.ecg_result_text.insert(tk.END, f"  {col}: M={mean:.2f}, SD={std:.2f}\n")

                # グラフ描画
                self._plot_ecg_results(sliding_df, label)

                if output_dir:
                    self.ecg_result_text.insert(tk.END, f"\n結果を保存しました: {output_dir}\n")
            else:
                self.ecg_result_text.insert(tk.END, "解析できませんでした。\n")

            self.status_var.set("解析完了")

        except Exception as e:
            messagebox.showerror("エラー", f"解析中にエラーが発生しました:\n{e}")
            self.status_var.set("エラー")

    def _plot_ecg_results(self, df, label):
        """ECG解析結果をプロット"""
        for widget in self.ecg_fig_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        axes[0].plot(df['Time'], df['LF/HF'], 'b-', linewidth=1)
        axes[0].set_ylabel('LF/HF')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df['Time'], df['RMSSD'], 'g-', linewidth=1)
        axes[1].set_ylabel('RMSSD (ms)')
        axes[1].grid(True, alpha=0.3)

        if 'SDNN' in df.columns:
            axes[2].plot(df['Time'], df['SDNN'], 'r-', linewidth=1)
            axes[2].set_ylabel('SDNN (ms)')
            axes[2].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (s)')
        fig.suptitle(f'HRV Time Series: {label}')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.ecg_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _clear_ecg_results(self):
        self.ecg_result_text.delete(1.0, tk.END)
        for widget in self.ecg_fig_frame.winfo_children():
            widget.destroy()

    # ========================================
    # バッチ解析タブ
    # ========================================
    def _create_batch_analysis_tab(self):
        """バッチ解析タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="バッチ解析")

        # 左側: ファイル設定
        left_frame = ttk.LabelFrame(tab, text="ファイル設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 条件ごとのファイル選択
        ttk.Label(left_frame, text="条件ごとにファイルを設定:", style='Header.TLabel').pack(anchor="w")

        self.batch_file_vars = {}
        conditions = ["Fixed", "HRF", "Sin", "HRF2_PID", "HRF2_Adaptive", "HRF2_GS"]
        for cond in conditions:
            frame = ttk.Frame(left_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{cond}:", width=15).pack(side=tk.LEFT)
            var = tk.StringVar()
            self.batch_file_vars[cond] = var
            ttk.Entry(frame, textvariable=var, width=40).pack(side=tk.LEFT, padx=2)
            ttk.Button(frame, text="...", width=3,
                       command=lambda c=cond: self._batch_browse_file(c)).pack(side=tk.LEFT)

        # フォルダから自動検出
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(left_frame, text="フォルダから自動検出",
                   command=self._batch_auto_detect).pack(pady=5)

        # パラメータ
        param_frame = ttk.LabelFrame(left_frame, text="パラメータ", padding="5")
        param_frame.pack(fill=tk.X, pady=10)

        ttk.Label(param_frame, text="被験者ID:").grid(row=0, column=0, sticky="w", pady=2)
        self.batch_subject_var = tk.StringVar(value="No1")
        ttk.Entry(param_frame, textvariable=self.batch_subject_var, width=10).grid(row=0, column=1, pady=2)

        ttk.Label(param_frame, text="解析開始(秒):").grid(row=1, column=0, sticky="w", pady=2)
        self.batch_start_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.batch_start_var, width=10).grid(row=1, column=1, pady=2)

        ttk.Label(param_frame, text="解析終了(秒):").grid(row=2, column=0, sticky="w", pady=2)
        self.batch_end_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.batch_end_var, width=10).grid(row=2, column=1, pady=2)

        # 出力設定
        output_frame = ttk.Frame(left_frame)
        output_frame.pack(fill=tk.X, pady=10)
        ttk.Label(output_frame, text="出力ディレクトリ:").pack(anchor="w")
        self.batch_output_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.batch_output_var, width=45).pack(side=tk.LEFT)
        ttk.Button(output_frame, text="...", command=self._batch_browse_output).pack(side=tk.LEFT, padx=2)

        # 実行ボタン
        ttk.Button(left_frame, text="バッチ解析実行", command=self._run_batch_analysis).pack(pady=10)

        # 右側: 結果
        right_frame = ttk.LabelFrame(tab, text="解析ログ", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.batch_log_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.batch_log_text.pack(fill=tk.BOTH, expand=True)

    def _batch_browse_file(self, condition):
        filepath = filedialog.askopenfilename(
            title=f"{condition}のECGファイルを選択",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if filepath:
            self.batch_file_vars[condition].set(filepath)

    def _batch_browse_output(self):
        dirpath = filedialog.askdirectory(title="出力ディレクトリを選択")
        if dirpath:
            self.batch_output_var.set(dirpath)

    def _batch_auto_detect(self):
        """フォルダからファイルを自動検出"""
        dirpath = filedialog.askdirectory(title="ECGファイルが含まれるフォルダを選択")
        if not dirpath:
            return

        import re
        detected = 0
        for filename in os.listdir(dirpath):
            if not filename.endswith('.csv'):
                continue
            for cond in self.batch_file_vars.keys():
                if cond.lower() in filename.lower():
                    self.batch_file_vars[cond].set(os.path.join(dirpath, filename))
                    detected += 1
                    break

        messagebox.showinfo("自動検出", f"{detected}個のファイルを検出しました")

    def _run_batch_analysis(self):
        output_dir = self.batch_output_var.get()
        if not output_dir:
            messagebox.showerror("エラー", "出力ディレクトリを指定してください")
            return

        # ファイルマップ作成
        files_map = {}
        for cond, var in self.batch_file_vars.items():
            if var.get() and os.path.exists(var.get()):
                files_map[cond] = var.get()

        if not files_map:
            messagebox.showerror("エラー", "解析するファイルがありません")
            return

        self.status_var.set("バッチ解析中...")
        self.batch_log_text.delete(1.0, tk.END)
        self.master.update()

        try:
            # パラメータ
            subject_id = self.batch_subject_var.get()
            start_offset = float(self.batch_start_var.get()) if self.batch_start_var.get() else None
            end_offset = float(self.batch_end_var.get()) if self.batch_end_var.get() else None

            # ログリダイレクト
            import io
            from contextlib import redirect_stdout

            log_buffer = io.StringIO()
            with redirect_stdout(log_buffer):
                run_batch_analysis(
                    files_map,
                    output_dir,
                    analysis_start_offset=start_offset,
                    analysis_end_offset=end_offset,
                    subject_id=subject_id,
                    save_plots=True,
                )

            self.batch_log_text.insert(tk.END, log_buffer.getvalue())
            self.batch_log_text.insert(tk.END, "\n\n解析完了!")
            self.status_var.set("バッチ解析完了")

        except Exception as e:
            self.batch_log_text.insert(tk.END, f"\nエラー: {e}\n")
            self.status_var.set("エラー")

    # ========================================
    # 箱ひげ図タブ
    # ========================================
    def _create_boxplot_tab(self):
        """箱ひげ図生成タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="箱ひげ図")

        # 入力設定
        input_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(input_frame, text="結合済みHRVデータ (Combined_HRV_Analysis.xlsx):").pack(anchor="w")
        file_row = ttk.Frame(input_frame)
        file_row.pack(fill=tk.X, pady=5)
        self.boxplot_file_var = tk.StringVar()
        ttk.Entry(file_row, textvariable=self.boxplot_file_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_row, text="参照", command=self._boxplot_browse_file).pack(side=tk.LEFT)

        output_row = ttk.Frame(input_frame)
        output_row.pack(fill=tk.X, pady=5)
        ttk.Label(output_row, text="出力ディレクトリ:").pack(side=tk.LEFT)
        self.boxplot_output_var = tk.StringVar()
        ttk.Entry(output_row, textvariable=self.boxplot_output_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_row, text="参照", command=self._boxplot_browse_output).pack(side=tk.LEFT)

        ttk.Button(input_frame, text="箱ひげ図を生成", command=self._generate_boxplots).pack(pady=10)

        # プレビュー
        preview_frame = ttk.LabelFrame(tab, text="プレビュー", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.boxplot_fig_frame = ttk.Frame(preview_frame)
        self.boxplot_fig_frame.pack(fill=tk.BOTH, expand=True)

    def _boxplot_browse_file(self):
        filepath = filedialog.askopenfilename(
            title="結合済みHRVデータを選択",
            filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")]
        )
        if filepath:
            self.boxplot_file_var.set(filepath)
            # 出力先を自動設定
            if not self.boxplot_output_var.get():
                self.boxplot_output_var.set(os.path.dirname(filepath))

    def _boxplot_browse_output(self):
        dirpath = filedialog.askdirectory(title="出力ディレクトリを選択")
        if dirpath:
            self.boxplot_output_var.set(dirpath)

    def _generate_boxplots(self):
        filepath = self.boxplot_file_var.get()
        output_dir = self.boxplot_output_var.get()

        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("エラー", "有効なファイルを選択してください")
            return
        if not output_dir:
            messagebox.showerror("エラー", "出力ディレクトリを指定してください")
            return

        self.status_var.set("箱ひげ図を生成中...")
        self.master.update()

        try:
            saved_files = generate_box_plots(filepath, output_dir)
            generate_summary_statistics(filepath, output_dir)

            messagebox.showinfo("完了", f"箱ひげ図を生成しました:\n" + "\n".join(saved_files))
            self.status_var.set("箱ひげ図生成完了")

        except Exception as e:
            messagebox.showerror("エラー", f"生成中にエラー:\n{e}")
            self.status_var.set("エラー")

    # ========================================
    # 制御性能評価タブ
    # ========================================
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
        self.ctrl_time_col_var = tk.StringVar(value="Timestamp")
        ttk.Entry(param_frame, textvariable=self.ctrl_time_col_var, width=20).grid(row=2, column=1, pady=2)

        ttk.Label(param_frame, text="HR列:").grid(row=3, column=0, sticky="w", pady=2)
        self.ctrl_hr_col_var = tk.StringVar(value="Heart Rate (BPM)")
        ttk.Entry(param_frame, textvariable=self.ctrl_hr_col_var, width=20).grid(row=3, column=1, pady=2)

        ttk.Button(left_frame, text="解析実行", command=self._run_control_analysis).pack(pady=10)

        # 右側: 結果
        right_frame = ttk.LabelFrame(tab, text="解析結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ctrl_result_text = scrolledtext.ScrolledText(right_frame, height=15, width=50)
        self.ctrl_result_text.pack(fill=tk.X)

        self.ctrl_fig_frame = ttk.Frame(right_frame)
        self.ctrl_fig_frame.pack(fill=tk.BOTH, expand=True, pady=5)

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
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            target_hr = self.ctrl_target_hr_var.get()
            time_col = self.ctrl_time_col_var.get()
            hr_col = self.ctrl_hr_col_var.get()

            # Timestampを経過秒に変換（日時形式の場合）
            if time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    df['_elapsed_seconds'] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
                    time_col = '_elapsed_seconds'
                except (ValueError, TypeError):
                    # 既に数値形式の場合はそのまま使用
                    pass

            self.control_analyzer.tolerance_bpm = self.ctrl_tolerance_var.get()

            metrics = self.control_analyzer.analyze_from_dataframe(
                df, time_col=time_col, hr_col=hr_col, target_hr=target_hr
            )

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

            self._plot_control_graph(df, time_col, hr_col, target_hr)

        except Exception as e:
            messagebox.showerror("エラー", f"解析中にエラー:\n{e}")

    def _plot_control_graph(self, df, time_col, hr_col, target_hr):
        for widget in self.ctrl_fig_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df[time_col], df[hr_col], 'b-', label='実測HR', linewidth=1)
        ax.axhline(y=target_hr, color='r', linestyle='--', label=f'目標HR ({target_hr} BPM)')
        ax.axhline(y=target_hr + 5, color='g', linestyle=':', alpha=0.5)
        ax.axhline(y=target_hr - 5, color='g', linestyle=':', alpha=0.5)
        ax.fill_between(df[time_col], target_hr - 5, target_hr + 5, alpha=0.1, color='green')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('HR (BPM)')
        ax.set_title('心拍数追従グラフ')
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, self.ctrl_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ========================================
    # 高度HRV解析タブ
    # ========================================
    def _create_advanced_hrv_tab(self):
        """高度HRV解析タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="高度HRV解析")

        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left_frame, text="RR間隔データ:").pack(anchor="w")
        self.hrv_file_var = tk.StringVar()
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.hrv_file_var, width=40).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="参照", command=self._hrv_browse_file).pack(side=tk.LEFT, padx=5)

        ttk.Label(left_frame, text="データタイプ:").pack(anchor="w", pady=(10, 0))
        self.hrv_data_type_var = tk.StringVar(value="rr")
        ttk.Radiobutton(left_frame, text="RR間隔 (ms)", variable=self.hrv_data_type_var, value="rr").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="ECG信号", variable=self.hrv_data_type_var, value="ecg").pack(anchor="w")

        self.hrv_nonlinear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="非線形指標を計算", variable=self.hrv_nonlinear_var).pack(anchor="w", pady=5)

        ttk.Button(left_frame, text="解析実行", command=self._run_hrv_analysis).pack(pady=10)

        right_frame = ttk.LabelFrame(tab, text="解析結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.hrv_result_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.hrv_result_text.pack(fill=tk.BOTH, expand=True)

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
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            data_type = self.hrv_data_type_var.get()
            compute_nonlinear = self.hrv_nonlinear_var.get()

            if data_type == 'rr':
                if 'RR' in df.columns:
                    rr = df['RR'].values
                elif 'rr' in df.columns:
                    rr = df['rr'].values
                elif 'RMSSD' in df.columns:
                    messagebox.showinfo("情報", "これはHRV結果ファイルです。RR間隔データを使用してください。")
                    return
                else:
                    rr = df.iloc[:, 0].values

                metrics = self.hrv_analyzer.calculate_metrics(rr, compute_nonlinear)
            else:
                # ECGデータ列を検索
                ecg_col = None
                for col in ['ECG Value (uV)', 'ECG', 'ecg', 'ECG_Value']:
                    if col in df.columns:
                        ecg_col = col
                        break
                if ecg_col:
                    ecg = df[ecg_col].values
                else:
                    # 最初の数値列を使用
                    ecg = df.select_dtypes(include=[np.number]).iloc[:, 0].values
                metrics = self.hrv_analyzer.analyze_from_ecg(ecg, compute_nonlinear=compute_nonlinear)

            self.hrv_result_text.delete(1.0, tk.END)
            self.hrv_result_text.insert(tk.END, "=== 高度HRV解析結果 ===\n\n")

            self.hrv_result_text.insert(tk.END, "【時間領域指標】\n")
            for key in ['Mean RR (ms)', 'Mean HR (BPM)', 'SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)', 'pNN20 (%)']:
                value = metrics.to_dict().get(key, 'N/A')
                if isinstance(value, float):
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value:.2f}\n")

            self.hrv_result_text.insert(tk.END, "\n【周波数領域指標】\n")
            for key in ['Total Power (ms²)', 'LF Power (ms²)', 'HF Power (ms²)', 'LF/HF', 'LF nu', 'HF nu']:
                value = metrics.to_dict().get(key, 'N/A')
                if isinstance(value, float):
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value:.2f}\n")

            self.hrv_result_text.insert(tk.END, "\n【非線形指標】\n")
            for key in ['SD1 (ms)', 'SD2 (ms)', 'SD2/SD1', 'Sample Entropy', 'DFA α1', 'DFA α2']:
                value = metrics.to_dict().get(key, 'N/A')
                if value is not None and isinstance(value, float):
                    self.hrv_result_text.insert(tk.END, f"  {key}: {value:.3f}\n")

        except Exception as e:
            messagebox.showerror("エラー", f"解析中にエラー:\n{e}")

    # ========================================
    # 統計検定タブ
    # ========================================
    def _create_statistics_tab(self):
        """統計検定タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="統計検定")

        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left_frame, text="データファイル:").pack(anchor="w")
        self.stat_file_var = tk.StringVar()
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(file_frame, textvariable=self.stat_file_var, width=40).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="参照", command=self._stat_browse_file).pack(side=tk.LEFT, padx=5)

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

        ttk.Label(left_frame, text="検定タイプ:").pack(anchor="w", pady=(10, 0))
        self.stat_test_type_var = tk.StringVar(value="rm_anova")
        ttk.Radiobutton(left_frame, text="反復測定ANOVA", variable=self.stat_test_type_var, value="rm_anova").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="一元配置ANOVA", variable=self.stat_test_type_var, value="anova").pack(anchor="w")
        ttk.Radiobutton(left_frame, text="Friedman検定", variable=self.stat_test_type_var, value="friedman").pack(anchor="w")

        ttk.Button(left_frame, text="検定実行", command=self._run_stat_analysis).pack(pady=10)

        right_frame = ttk.LabelFrame(tab, text="検定結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.stat_result_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.stat_result_text.pack(fill=tk.BOTH, expand=True)

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
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            subject_col = self.stat_subject_col_var.get()
            condition_col = self.stat_condition_col_var.get()
            value_col = self.stat_value_col_var.get()
            test_type = self.stat_test_type_var.get()

            self.stat_result_text.delete(1.0, tk.END)
            self.stat_result_text.insert(tk.END, "=== 統計検定結果 ===\n\n")

            # 記述統計
            self.stat_result_text.insert(tk.END, "【記述統計】\n")
            for cond in df[condition_col].unique():
                cond_data = df[df[condition_col] == cond][value_col].values
                stats = self.stat_analyzer.summary_statistics(cond_data)
                self.stat_result_text.insert(tk.END, f"\n{cond}:\n")
                self.stat_result_text.insert(tk.END, f"  n={stats['n']}, M={stats['mean']:.2f}, SD={stats['std']:.2f}\n")

            self.stat_result_text.insert(tk.END, "\n【検定結果】\n")

            if test_type == 'rm_anova':
                result = self.stat_analyzer.repeated_measures_anova(df, subject_col, condition_col, value_col)
                self.stat_result_text.insert(tk.END, "\n反復測定一元配置ANOVA:\n")
            elif test_type == 'anova':
                groups = {cond: df[df[condition_col] == cond][value_col].values for cond in df[condition_col].unique()}
                result = self.stat_analyzer.one_way_anova(groups)
                self.stat_result_text.insert(tk.END, "\n一元配置ANOVA:\n")
            else:
                result = self.stat_analyzer.friedman_test(df, subject_col, condition_col, value_col)
                self.stat_result_text.insert(tk.END, "\nFriedman検定:\n")

            for key, value in result.to_dict().items():
                if isinstance(value, float):
                    self.stat_result_text.insert(tk.END, f"  {key}: {value:.4f}\n")
                else:
                    self.stat_result_text.insert(tk.END, f"  {key}: {value}\n")

        except Exception as e:
            messagebox.showerror("エラー", f"検定中にエラー:\n{e}")

    # ========================================
    # 相関分析タブ
    # ========================================
    def _create_correlation_tab(self):
        """相関分析タブ"""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="相関分析")

        left_frame = ttk.LabelFrame(tab, text="入力設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left_frame, text="HRVデータファイル:").pack(anchor="w")
        self.corr_hrv_file_var = tk.StringVar()
        hrv_frame = ttk.Frame(left_frame)
        hrv_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(hrv_frame, textvariable=self.corr_hrv_file_var, width=35).pack(side=tk.LEFT)
        ttk.Button(hrv_frame, text="参照", command=lambda: self._corr_browse_file('hrv')).pack(side=tk.LEFT, padx=5)

        ttk.Label(left_frame, text="主観評価データファイル:").pack(anchor="w", pady=(10, 0))
        self.corr_subj_file_var = tk.StringVar()
        subj_frame = ttk.Frame(left_frame)
        subj_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(subj_frame, textvariable=self.corr_subj_file_var, width=35).pack(side=tk.LEFT)
        ttk.Button(subj_frame, text="参照", command=lambda: self._corr_browse_file('subj')).pack(side=tk.LEFT, padx=5)

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

        ttk.Button(left_frame, text="相関分析実行", command=self._run_corr_analysis).pack(pady=10)

        right_frame = ttk.LabelFrame(tab, text="分析結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.corr_result_text = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.corr_result_text.pack(fill=tk.BOTH, expand=True)

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

            result_df = self.corr_analyzer.analyze_hrv_subjective_relation(
                hrv_df, subj_df, hrv_cols, subj_cols, merge_col
            )

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
            messagebox.showerror("エラー", f"分析中にエラー:\n{e}")


def main():
    """メイン関数"""
    root = tk.Tk()
    app = IntegratedAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
