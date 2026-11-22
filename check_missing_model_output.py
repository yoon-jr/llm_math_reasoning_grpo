from pathlib import Path
import pandas as pd

# 확인할 outputs 폴더
OUTPUT_ROOT = Path("outputs")

# 결과 저장 파일명
REPORT_PATH = Path("missing_model_output_report.csv")

def is_missing_model_output(series):
    """
    model_output이 비어 있는 경우를 판별:
    - NaN
    - 빈 문자열
    - 공백만 있는 문자열
    - 'nan', 'none', 'null' 같은 문자열
    """
    s = series.astype("string")
    return (
        s.isna()
        | s.str.strip().eq("")
        | s.str.strip().str.lower().isin(["nan", "none", "null"])
    )

def main():
    csv_files = sorted(OUTPUT_ROOT.rglob("output_gpt_multichoice_round_*.csv"))

    if not csv_files:
        print(f"CSV 파일을 찾지 못했습니다: {OUTPUT_ROOT}")
        return

    all_missing = []

    print("검사한 CSV 파일:")
    for csv_path in csv_files:
        print(f"- {csv_path}")

        df = pd.read_csv(csv_path)

        if "model_output" not in df.columns:
            print(f"  경고: model_output 컬럼이 없습니다. 건너뜁니다.")
            continue

        missing_mask = is_missing_model_output(df["model_output"])
        missing_df = df[missing_mask].copy()

        if len(missing_df) == 0:
            print(f"  model_output 누락: 0개")
            continue

        missing_df.insert(0, "source_csv", str(csv_path))

        keep_cols = [
            "source_csv",
            "round",
            "idx_in_dataset",
            "problem_index",
            "sample_index",
            "question_used",
            "predicted_final_answer",
            "reference_answer",
            "is_correct",
            "format_valid",
            "model_output",
        ]

        existing_cols = [col for col in keep_cols if col in missing_df.columns]
        missing_df = missing_df[existing_cols]

        all_missing.append(missing_df)

        print(f"  model_output 누락: {len(missing_df)}개")

    print("-" * 60)

    if not all_missing:
        print("전체 CSV에서 model_output 누락이 없습니다.")
        return

    report_df = pd.concat(all_missing, ignore_index=True)
    report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    print(f"전체 model_output 누락 개수: {len(report_df)}개")
    print(f"누락 문제 목록 저장 완료: {REPORT_PATH}")

    print("\n누락된 문제 요약:")
    summary_cols = [
        "source_csv",
        "round",
        "idx_in_dataset",
        "problem_index",
        "sample_index",
        "reference_answer",
    ]
    summary_cols = [col for col in summary_cols if col in report_df.columns]
    print(report_df[summary_cols].to_string(index=False))

if __name__ == "__main__":
    main()
