import os
import re
from collections import defaultdict

import pandas as pd

from Analyzer import run_batch_analysis
from GenerateBoxPlots import generate_box_plots
from gui import launch_ui


FILENAME_PATTERN = re.compile(r".*_No(?P<subject>\d+)_\d{8}_\d{6}_(?P<condition>Sin|Fixed|HRF)\.csv$", re.IGNORECASE)
CONDITION_MAP = {"sin": "Sin", "fixed": "Fixed", "hrf": "HRF"}
CONDITION_ORDER = ["Sin", "Fixed", "HRF"]


def main():
    print("Welcome to the HRV Analysis tool.")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    def collect_subject_files(input_dir):
        subject_files = defaultdict(dict)
        for entry in os.listdir(input_dir):
            full_path = os.path.join(input_dir, entry)
            if not os.path.isfile(full_path) or not entry.lower().endswith('.csv'):
                continue
            match = FILENAME_PATTERN.match(entry)
            if not match:
                continue
            subject_id = f"No{match.group('subject')}"
            condition_raw = match.group('condition').lower()
            condition = CONDITION_MAP.get(condition_raw)
            if not condition:
                continue
            subject_files[subject_id][condition] = full_path
        return subject_files

    def run_analysis_for_subject(subject_id, files_by_condition, result_root):
        ordered_files = {condition: files_by_condition[condition] for condition in CONDITION_ORDER}
        subject_dir = os.path.join(result_root, subject_id)
        print(f"\n=== {subject_id} の解析を開始します ===")
        run_batch_analysis(ordered_files, subject_dir)
        combined_file = os.path.join(subject_dir, "Combined_HRV_Analysis.xlsx")
        if os.path.exists(combined_file):
            print(f"{subject_id}: 箱ひげ図を作成します")
            generate_box_plots(combined_file, subject_dir)
        else:
            print(f"{subject_id}: Combined_HRV_Analysis.xlsx が見つかりませんでした。")

    def combine_all_subjects(result_root):
        if not os.path.isdir(result_root):
            raise FileNotFoundError("result_batch フォルダが見つかりません。先に解析を実行してください。")

        subject_files = []
        for entry in sorted(os.listdir(result_root)):
            subject_dir = os.path.join(result_root, entry)
            combined_path = os.path.join(subject_dir, "Combined_HRV_Analysis.xlsx")
            if os.path.isfile(combined_path):
                subject_files.append((entry, combined_path))

        if not subject_files:
            raise FileNotFoundError("統合対象の Combined_HRV_Analysis.xlsx が見つかりません。")

        condition_priority = {cond: idx for idx, cond in enumerate(CONDITION_ORDER)}
        frames = []
        included_subjects = set()

        for subject_id, file_path in subject_files:
            df = pd.read_excel(file_path)
            subject_included = False
            for condition in CONDITION_ORDER:
                lf_col = f"{condition}_LF/HF"
                rmssd_col = f"{condition}_RMSSD"
                if lf_col not in df.columns or rmssd_col not in df.columns:
                    continue
                subset = df[['Time', lf_col, rmssd_col]].copy()
                subset.rename(columns={
                    lf_col: 'LF/HF',
                    rmssd_col: 'RMSSD'
                }, inplace=True)
                subset['Subject'] = subject_id
                subset['Condition'] = condition
                subset['ConditionOrder'] = condition_priority[condition]
                frames.append(subset[['Subject', 'Condition', 'ConditionOrder', 'Time', 'LF/HF', 'RMSSD']])
                subject_included = True
            if subject_included:
                included_subjects.add(subject_id)

        if not frames:
            raise ValueError("統合に使用できるデータ列が見つかりませんでした。")

        combined_df = pd.concat(frames, ignore_index=True)
        combined_df.sort_values(['ConditionOrder', 'Subject', 'Time'], inplace=True)
        combined_df.drop(columns=['ConditionOrder'], inplace=True)

        output_path = os.path.join(result_root, "Combined_AllSubjects.xlsx")
        combined_df.to_excel(output_path, index=False)

        return {
            'output_path': output_path,
            'subjects': sorted(included_subjects),
            'rows': len(combined_df)
        }

    def run_callback(input_dir):
        if not input_dir or not os.path.isdir(input_dir):
            raise FileNotFoundError("入力フォルダが選択されていません。")

        print(f"\n入力フォルダ: {input_dir}")
        subject_files = collect_subject_files(input_dir)
        if not subject_files:
            raise FileNotFoundError("指定フォルダに解析可能なファイルが見つかりません。")

        os.makedirs(os.path.join(base_dir, "result_batch"), exist_ok=True)
        result_root = os.path.join(base_dir, "result_batch")

        processed = 0
        skipped = {}

        for subject_id, files_by_condition in subject_files.items():
            missing = [cond for cond in CONDITION_ORDER if cond not in files_by_condition]
            if missing:
                skipped[subject_id] = missing
                print(f"{subject_id}: {', '.join(missing)} のファイルが不足しているためスキップします。")
                continue
            try:
                run_analysis_for_subject(subject_id, files_by_condition, result_root)
                processed += 1
            except Exception as exc:
                skipped[subject_id] = [str(exc)]
                print(f"{subject_id}: 解析中にエラーが発生しました -> {exc}")

        return {
            "processed": processed,
            "skipped": skipped,
            "result_root": result_root
        }

    def combine_callback():
        result_root = os.path.join(base_dir, "result_batch")
        return combine_all_subjects(result_root)

    launch_ui(run_callback, combine_callback)


if __name__ == "__main__":
    main()
