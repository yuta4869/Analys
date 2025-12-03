import os
import tkinter as tk
from tkinter import filedialog, messagebox

class AnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("HRV Analysis File Selector")
        master.geometry("500x200")

        self.file_paths = {}
        self.labels = ["Fixed", "Sin", "HRF"]
        self.path_labels = {}

        # Create input rows
        for i, label_text in enumerate(self.labels):
            # Condition label (e.g., "Fixed:")
            condition_label = tk.Label(master, text=f"{label_text}:")
            condition_label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            # Label to display selected file path
            path_label = tk.Label(master, text="No file selected.", width=40, anchor="w", fg="grey")
            path_label.grid(row=i, column=1, padx=10, pady=10, sticky="w")
            self.path_labels[label_text] = path_label

            # Browse button
            browse_button = tk.Button(master, text="Browse...", command=lambda l=label_text: self.browse_file(l))
            browse_button.grid(row=i, column=2, padx=10, pady=10)

        # Create Run button
        run_button = tk.Button(master, text="Run Analysis", command=self.run_analysis)
        run_button.grid(row=len(self.labels), column=1, pady=20)

    def browse_file(self, label):
        file_path = filedialog.askopenfilename(
            title=f"Select file for {label}",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if file_path:
            self.file_paths[label] = file_path
            # Display just the filename
            filename = os.path.basename(file_path)
            self.path_labels[label].config(text=filename, fg="black")

    def run_analysis(self):
        if not self.file_paths:
            messagebox.showwarning("No Files", "No files were selected. The program will exit.")
        
        self.master.quit()

def launch_and_get_files():
    """
    Launches the GUI, waits for the user to select files and click 'Run Analysis',
    and returns the selected file paths.

    :return: A dictionary mapping labels to file paths, e.g., {'Fixed': 'path/to/file.csv'}.
             Returns an empty dictionary if no files are selected.
    """
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()
    
    selected_files = app.file_paths
    try:
        root.destroy()
    except tk.TclError:
        # This can happen if the user closes the window manually.
        pass
    return selected_files


if __name__ == '__main__':
    print("Launching GUI to select files...")
    file_map = launch_and_get_files()

    if file_map:
        print("\nAnalysis would run with the following files:")
        for label, path in file_map.items():
            print(f"  {label}: {path}")
    else:
        print("\nNo files were selected or the window was closed.")
