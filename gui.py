import os
import tkinter as tk
from tkinter import filedialog, messagebox


class AnalysisGUI:
    def __init__(self, master, run_callback, boxplot_callback):
        self.master = master
        self.run_callback = run_callback
        self.boxplot_callback = boxplot_callback

        master.title("HRV Analysis File Selector")
        master.geometry("560x260")

        self.file_paths = {}
        self.labels = ["Fixed", "Sin", "HRF"]
        self.path_labels = {}
        self.status_var = tk.StringVar(value="解析ファイルを選択してください。")

        self._build_widgets()

    def _build_widgets(self):
        for i, label_text in enumerate(self.labels):
            label = tk.Label(self.master, text=f"{label_text}:")
            label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            path_label = tk.Label(self.master, text="No file selected.", width=40, anchor="w", fg="grey")
            path_label.grid(row=i, column=1, padx=10, pady=10, sticky="w")
            self.path_labels[label_text] = path_label

            button = tk.Button(
                self.master,
                text="Browse...",
                command=lambda l=label_text: self._browse_file(l)
            )
            button.grid(row=i, column=2, padx=10, pady=10)

        self.run_button = tk.Button(self.master, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=len(self.labels), column=1, pady=(5, 5))

        self.boxplot_button = tk.Button(
            self.master,
            text="箱ひげ図を作成",
            state=tk.DISABLED,
            command=self.create_boxplots
        )
        self.boxplot_button.grid(row=len(self.labels) + 1, column=1, pady=(0, 5))

        status_label = tk.Label(
            self.master,
            textvariable=self.status_var,
            wraplength=480,
            justify="left",
            fg="blue"
        )
        status_label.grid(row=len(self.labels) + 2, column=0, columnspan=3, padx=10, sticky="w")

        self.exit_button = tk.Button(self.master, text="終了", command=self.master.destroy)
        self.exit_button.grid(row=len(self.labels) + 3, column=1, pady=(5, 10))

    def _browse_file(self, label):
        file_path = filedialog.askopenfilename(
            title=f"Select file for {label}",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if file_path:
            self.file_paths[label] = file_path
            filename = os.path.basename(file_path)
            self.path_labels[label].config(text=filename, fg="black")
            self.status_var.set("解析ファイルを選択しました。Run Analysis を押してください。")

    def run_analysis(self):
        if not self.file_paths:
            messagebox.showwarning("No Files", "解析対象のファイルを選択してください。")
            return

        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("解析を実行中です。完了するまでお待ちください。")
        self.master.update_idletasks()

        try:
            self.run_callback(self.file_paths)
        except Exception as exc:
            messagebox.showerror("解析エラー", f"解析中にエラーが発生しました。\n{exc}")
            self.run_button.config(state=tk.NORMAL)
            self.status_var.set("解析に失敗しました。ログを確認してください。")
            return

        self.status_var.set("解析が完了しました。箱ひげ図を作成できます。")
        self.boxplot_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)

    def create_boxplots(self):
        self.boxplot_button.config(state=tk.DISABLED)
        self.status_var.set("箱ひげ図を作成しています...")
        self.master.update_idletasks()

        try:
            saved_files = self.boxplot_callback()
        except FileNotFoundError as exc:
            messagebox.showerror("ファイルなし", str(exc))
            self.status_var.set("Combined_HRV_Analysis.xlsx が見つかりません。解析を再実行してください。")
            self.boxplot_button.config(state=tk.NORMAL)
            return
        except Exception as exc:
            messagebox.showerror("グラフエラー", f"箱ひげ図の作成に失敗しました。\n{exc}")
            self.status_var.set("箱ひげ図の作成に失敗しました。")
            self.boxplot_button.config(state=tk.NORMAL)
            return

        if saved_files:
            message = "箱ひげ図を作成しました:\n" + "\n".join(saved_files)
            messagebox.showinfo("完了", message)
            self.status_var.set("箱ひげ図の作成が完了しました。")
        else:
            messagebox.showwarning("結果なし", "保存された箱ひげ図がありませんでした。")
            self.status_var.set("箱ひげ図の保存に失敗しました。")

        self.boxplot_button.config(state=tk.NORMAL)


def launch_ui(run_callback, boxplot_callback):
    root = tk.Tk()
    app = AnalysisGUI(root, run_callback, boxplot_callback)
    root.mainloop()
