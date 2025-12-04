import os
import tkinter as tk
from tkinter import messagebox

from gui import launch_and_get_files
from Analyzer import run_batch_analysis
from GenerateBoxPlots import generate_box_plots


def launch_boxplot_gui(combined_file_path):
    """
    Combined_HRV_Analysis.xlsx が存在する場合、箱ひげ図を作成するGUIを表示する。
    """
    if not os.path.exists(combined_file_path):
        print("\nCombined_HRV_Analysis.xlsx が見つからないため、箱ひげ図GUIは表示されません。")
        return

    def on_generate():
        try:
            saved_files = generate_box_plots(combined_file_path, os.path.dirname(combined_file_path))
            message = "箱ひげ図を作成しました:\n" + "\n".join(saved_files)
            messagebox.showinfo("完了", message)
            generate_button.config(state=tk.DISABLED)
        except Exception as exc:
            messagebox.showerror("エラー", f"箱ひげ図の作成に失敗しました。\n{exc}")

    window = tk.Tk()
    window.title("箱ひげ図の作成")
    window.geometry("400x180")

    label_text = (
        "Combined_HRV_Analysis.xlsx が作成されました。\n"
        "下のボタンを押すと箱ひげ図を生成します。"
    )
    tk.Label(window, text=label_text, wraplength=360, justify="left").pack(pady=20)

    generate_button = tk.Button(window, text="箱ひげ図を作成", command=on_generate)
    generate_button.pack(pady=5)

    tk.Button(window, text="閉じる", command=window.destroy).pack(pady=5)

    print("\n箱ひげ図作成用のGUIを表示しています。")
    window.mainloop()

def main():
    """
    Main function to run the HRV analysis application.
    It launches a GUI for specific file selection and then runs 
    the analysis on the selected files.
    """
    print("Welcome to the HRV Analysis tool.")
    
    # Get the directory where this script is located to use as the base for the output folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Launch GUI to get a map of labeled files
    files_map = launch_and_get_files()
    
    # 2. Check if the file map is populated
    if not files_map:
        print("\nNo files were selected for analysis. Exiting program.")
        return
        
    print(f"\nStarting analysis for {len(files_map)} file(s)")
    for label, path in files_map.items():
        print(f"  - {label}: {path}")
        
    # 3. Run the analysis
    run_batch_analysis(files_map, base_dir)

    combined_file_path = os.path.join(base_dir, "result_batch", "Combined_HRV_Analysis.xlsx")
    launch_boxplot_gui(combined_file_path)

if __name__ == "__main__":
    main()
