from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
DATASET_NAME = "text_only"
ALGORITHM_NAME = "trainingfreegrpo"

# 보고 싶은 prompt 목록
# 예: ["p1", "p2"] 이면
# p1_round0/1/2..., p2_round0/1/2... 각각에 대한 prompt sensitivity 계산
PROMPT_IDS = ["p1", "p2", "p3", "p4", "p5"]

# outputs 폴더
OUTPUT_ROOT = Path("outputs")

# 결과 저장 폴더
EVAL_DIR = Path("evaluation_results") / DATASET_NAME
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = EVAL_DIR / f"prompt_sensitivity_by_prompt_{ALGORITHM_NAME}.csv"
SAMPLE_CSV_PATH = EVAL_DIR / f"prompt_sensitivity_by_sample_{ALGORITHM_NAME}.csv"
CHART_PATH = EVAL_DIR / f"prompt_sensitivity_by_prompt_{ALGORITHM_NAME}.png"


# =========================
# 유틸 함수
# =========================
def parse_round_number(csv_path: Path) -> int:
    """
    파일명에서 round 번호 추출
    예: output_gpt_multichoice_round_2.csv -> 2
    """
    match = re.search(r"round_(\d+)", csv_path.name)
    if match is None:
        return -1
    return int(match.group(1))


def normalize_bool_value(value):
    """
    CSV에서 True/False가 문자열로 읽히는 경우까지 처리
    True/1/yes/y  -> 1
    False/0/no/n  -> 0
    알 수 없으면 None
    """
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return 1 if value else 0

    text = str(value).strip().lower()

    if text in ["true", "1", "yes", "y"]:
        return 1
    if text in ["false", "0", "no", "n"]:
        return 0

    return None


def is_missing_text(value):
    """
    model_output이 비어 있는지 확인
    """
    if pd.isna(value):
        return True

    text = str(value).strip()
    if text == "":
        return True

    if text.lower() in ["nan", "none", "null"]:
        return True

    return False


def get_sample_key(row):
    """
    문제를 식별하기 위한 key 생성.
    기본은 problem_index + sample_index.
    sample_index가 없으면 problem_index만 사용.
    """
    if "problem_index" not in row.index:
        raise ValueError("CSV에 problem_index 컬럼이 없습니다.")

    problem_index = str(row["problem_index"]).strip()

    if "sample_index" in row.index:
        sample_index = str(row["sample_index"]).strip()
    else:
        sample_index = ""

    return problem_index, sample_index


def get_correct_series(df: pd.DataFrame, csv_path: Path):
    """
    각 row의 정답 여부를 0/1로 반환.
    우선순위:
    1. is_correct 컬럼
    2. predicted_final_answer == reference_answer
    """
    if "is_correct" in df.columns:
        return df["is_correct"].apply(normalize_bool_value)

    required_cols = ["predicted_final_answer", "reference_answer"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_path}에 is_correct도 없고, {missing_cols} 컬럼도 없습니다."
        )

    pred = df["predicted_final_answer"].astype(str).str.strip()
    ref = df["reference_answer"].astype(str).str.strip()
    return (pred == ref).astype(int)


def get_all_round_csvs(prompt_id: str) -> list[Path]:
    """
    특정 prompt의 모든 round CSV 찾기
    """
    prompt_dir = OUTPUT_ROOT / DATASET_NAME / prompt_id / ALGORITHM_NAME

    csv_files = sorted(
        prompt_dir.glob("output_gpt_multichoice_round_*.csv"),
        key=parse_round_number,
    )

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾지 못했습니다: {prompt_dir}")

    return csv_files


def variance_binary(xs: list[int]) -> float:
    """
    compute_prompt_sensitivity.py와 같은 방식:
    xs의 population variance 계산.
    xs는 0/1 정답 여부 리스트.
    """
    n = len(xs)
    if n == 0:
        return 0.0

    mu = sum(xs) / n
    return sum((x - mu) ** 2 for x in xs) / n


# =========================
# 핵심 계산
# =========================
def compute_prompt_sensitivity_for_prompt(prompt_id: str):
    """
    하나의 prompt_id에 대해 round_0, round_1, round_2... 결과를 모아서
    sample별 prompt sensitivity를 계산하고,
    전체 평균 prompt sensitivity를 반환.
    """
    round_csvs = get_all_round_csvs(prompt_id)

    # (problem_index, sample_index) -> [(round, correct), ...]
    corrects_by_sample = defaultdict(list)

    round_numbers = []
    total_rows_by_round = {}
    missing_output_by_round = {}

    for csv_path in round_csvs:
        round_num = parse_round_number(csv_path)
        round_numbers.append(round_num)

        df = pd.read_csv(csv_path)
        total_rows_by_round[round_num] = len(df)

        if "model_output" in df.columns:
            missing_output_by_round[round_num] = int(df["model_output"].apply(is_missing_text).sum())
        else:
            missing_output_by_round[round_num] = 0

        correct_series = get_correct_series(df, csv_path)

        for idx, row in df.iterrows():
            correct_value = correct_series.iloc[idx]

            # 정답 여부를 알 수 없는 row는 제외
            if correct_value is None or pd.isna(correct_value):
                continue

            sample_key = get_sample_key(row)
            corrects_by_sample[sample_key].append((round_num, int(correct_value)))

    sample_rows = []
    ps_values = []
    n_rounds_expected = len(round_numbers)

    for (problem_index, sample_index), rc_list in sorted(corrects_by_sample.items()):
        rc_list_sorted = sorted(rc_list, key=lambda x: x[0])
        correct_list = [correct for _, correct in rc_list_sorted]
        round_list = [round_num for round_num, _ in rc_list_sorted]

        # round가 2개 이상 있어야 변화/민감도를 볼 수 있음
        if len(correct_list) < 2:
            continue

        ps = variance_binary(correct_list)
        mu = sum(correct_list) / len(correct_list)

        ps_values.append(ps)

        sample_rows.append({
            "Prompt": prompt_id.upper(),
            "Prompt ID": prompt_id,
            "problem_index": problem_index,
            "sample_index": sample_index,
            "n_rounds": len(correct_list),
            "round_list": " ".join(str(x) for x in round_list),
            "correct_list": " ".join(str(x) for x in correct_list),
            "mean_accuracy_mu": mu,
            "prompt_sensitivity_ps": ps,
        })

    avg_ps = sum(ps_values) / len(ps_values) if ps_values else 0.0
    min_ps = min(ps_values) if ps_values else 0.0
    max_ps = max(ps_values) if ps_values else 0.0

    summary_row = {
        "Prompt": prompt_id.upper(),
        "Prompt ID": prompt_id,
        "Rounds": " ".join(str(x) for x in sorted(round_numbers)),
        "N Rounds Found": len(round_numbers),
        "N Samples Used": len(ps_values),
        "Avg Prompt Sensitivity": avg_ps,
        "Min Prompt Sensitivity": min_ps,
        "Max Prompt Sensitivity": max_ps,
        "Expected Rounds Per Sample": n_rounds_expected,
        "Total Rows By Round": str(dict(sorted(total_rows_by_round.items()))),
        "Missing model_output By Round": str(dict(sorted(missing_output_by_round.items()))),
    }

    return summary_row, sample_rows


# =========================
# 그래프
# =========================
def plot_prompt_sensitivity_bar(summary_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    x_labels = summary_df["Prompt"].tolist()
    y_values = summary_df["Avg Prompt Sensitivity"].tolist()

    plt.bar(x_labels, y_values)

    for i, value in enumerate(y_values):
        plt.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    y_max = max(y_values) if y_values else 0.0
    padding = y_max * 0.2 if y_max > 0 else 0.05

    # binary correctness의 분산 최대값은 0.25
    plt.ylim(0, min(0.25, y_max + padding) if y_max > 0 else 0.25)

    plt.xlabel("Prompt")
    plt.ylabel("Average Prompt Sensitivity")
    plt.title(f"Prompt Sensitivity by Prompt ({DATASET_NAME}, {ALGORITHM_NAME})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()


# =========================
# main
# =========================
def main():
    summary_rows = []
    all_sample_rows = []

    print(f"Dataset: {DATASET_NAME}")
    print(f"Algorithm: {ALGORITHM_NAME}")
    print(f"Prompt IDs: {PROMPT_IDS}")
    print("-" * 60)

    for prompt_id in PROMPT_IDS:
        summary_row, sample_rows = compute_prompt_sensitivity_for_prompt(prompt_id)
        summary_rows.append(summary_row)
        all_sample_rows.extend(sample_rows)

        print(f"{prompt_id.upper()} prompt sensitivity 계산 완료")
        print(f"  Rounds: {summary_row['Rounds']}")
        print(f"  N Samples Used: {summary_row['N Samples Used']}")
        print(f"  Avg Prompt Sensitivity: {summary_row['Avg Prompt Sensitivity']:.6f}")
        print(f"  Min Prompt Sensitivity: {summary_row['Min Prompt Sensitivity']:.6f}")
        print(f"  Max Prompt Sensitivity: {summary_row['Max Prompt Sensitivity']:.6f}")
        print()

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.DataFrame(all_sample_rows)

    summary_df = summary_df.sort_values("Prompt ID")
    if not sample_df.empty:
        sample_df = sample_df.sort_values(["Prompt ID", "problem_index", "sample_index"])

    summary_df.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")
    sample_df.to_csv(SAMPLE_CSV_PATH, index=False, encoding="utf-8-sig")

    plot_prompt_sensitivity_bar(summary_df)

    print("-" * 60)
    print("Prompt sensitivity 평가 완료")
    print(f"요약 CSV 저장 위치: {SUMMARY_CSV_PATH}")
    print(f"샘플별 CSV 저장 위치: {SAMPLE_CSV_PATH}")
    print(f"막대 그래프 저장 위치: {CHART_PATH}")
    print()
    print(
        summary_df[
            [
                "Prompt",
                "Rounds",
                "N Samples Used",
                "Avg Prompt Sensitivity",
                "Min Prompt Sensitivity",
                "Max Prompt Sensitivity",
            ]
        ]
    )


if __name__ == "__main__":
    main()
