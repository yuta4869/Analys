import os
import tkinter as tk
from tkinter import filedialog, messagebox


class AnalysisGUI:
    def __init__(self, master, run_callback, combine_callback):
        self.master = master
        self.run_callback = run_callback
        self.combine_callback = combine_callback

        self.master.title("HRV Analysis File Selector")
        self.master.geometry("600x220")

        self.input_dir = None
        self.path_var = tk.StringVar(value="入力フォルダが選択されていません。")
        self.status_var = tk.StringVar(value="フォルダを選択して解析を開始してください。")

        self._build_widgets()

    def _build_widgets(self):
        label = tk.Label(self.master, text="入力フォルダ:")
        label.grid(row=0, column=0, padx=10, pady=15, sticky="w")

        path_label = tk.Label(self.master, textvariable=self.path_var, width=45, anchor="w", fg="grey")
        path_label.grid(row=0, column=1, padx=10, pady=15, sticky="w")

        browse_button = tk.Button(self.master, text="Browse...", command=self._browse_folder)
        browse_button.grid(row=0, column=2, padx=10, pady=15)

        self.run_button = tk.Button(self.master, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=1, column=1, pady=(5, 5))

        self.combine_button = tk.Button(self.master, text="被験者統合", command=self.combine_results)
        self.combine_button.grid(row=1, column=2, pady=(5, 5))

        status_label = tk.Label(
            self.master,
            textvariable=self.status_var,
            wraplength=520,
            justify="left",
            fg="blue"
        )
        status_label.grid(row=2, column=0, columnspan=3, padx=10, sticky="w")

        self.exit_button = tk.Button(self.master, text="終了", command=self.master.destroy)
        self.exit_button.grid(row=3, column=1, pady=(10, 10))

    def _browse_folder(self):
        folder_path = filedialog.askdirectory(title="解析対象のフォルダを選択してください")
        if folder_path:
            self.input_dir = folder_path
            self.path_var.set(folder_path)
            self.status_var.set("Run Analysis を押して解析を開始してください。")

    def run_analysis(self):
        if not self.input_dir:
            messagebox.showwarning("フォルダ未選択", "まず解析対象のフォルダを選択してください。")
            return

        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("解析を実行中です。完了するまでお待ちください。")
        self.master.update_idletasks()

        try:
            summary = self.run_callback(self.input_dir)
        except FileNotFoundError as exc:
            messagebox.showwarning("解析不可", str(exc))
            self.status_var.set(str(exc))
            self.run_button.config(state=tk.NORMAL)
            return
        except Exception as exc:
            messagebox.showerror("解析エラー", f"解析中にエラーが発生しました。\n{exc}")
            self.status_var.set("解析に失敗しました。ログを確認してください。")
            self.run_button.config(state=tk.NORMAL)
            return

        processed = summary.get("processed", 0)
        skipped = summary.get("skipped", {})
        result_root = summary.get("result_root")

        message_lines = [f"{processed}名の被験者を処理しました。"]
        if skipped:
            detail_lines = []
            for subject_id, reasons in skipped.items():
                detail_lines.append(f"  - {subject_id}: {', '.join(reasons)}")
            message_lines.append("以下の被験者はスキップされました:")
            message_lines.extend(detail_lines)

        if result_root:
            message_lines.append(f"出力フォルダ: {result_root}")

        messagebox.showinfo("解析完了", "\n".join(message_lines))
        self.status_var.set("解析が完了しました。")
        self.run_button.config(state=tk.NORMAL)

    def combine_results(self):
        self.combine_button.config(state=tk.DISABLED)
        self.status_var.set("全被験者の統合を実行中です...")
        self.master.update_idletasks()

        try:
            summary = self.combine_callback()
        except FileNotFoundError as exc:
            messagebox.showwarning("統合不可", str(exc))
            self.status_var.set(str(exc))
            self.combine_button.config(state=tk.NORMAL)
            return
        except Exception as exc:
            messagebox.showerror("統合エラー", f"統合処理中にエラーが発生しました。\n{exc}")
            self.status_var.set("統合が失敗しました。ログを確認してください。")
            self.combine_button.config(state=tk.NORMAL)
            return

        subjects = summary.get('subjects', [])
        output_path = summary.get('output_path')
        rows = summary.get('rows')
        boxplots = summary.get('boxplots', [])

        message_lines = ["統合処理が完了しました。"]
        if subjects:
            message_lines.append(f"対象被験者: {', '.join(subjects)}")
        if rows is not None:
            message_lines.append(f"総データ行数: {rows}")
        if output_path:
            message_lines.append(f"保存先: {output_path}")
        if boxplots:
            message_lines.append("箱ひげ図:")
            message_lines.extend([f"  - {path}" for path in boxplots])

        messagebox.showinfo("統合完了", "\n".join(message_lines))
        self.status_var.set("全被験者統合が完了しました。")
        self.combine_button.config(state=tk.NORMAL)


def launch_ui(run_callback, combine_callback):
    root = tk.Tk()
    app = AnalysisGUI(root, run_callback, combine_callback)
    root.mainloop()
