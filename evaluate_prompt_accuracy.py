from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
DATASET_NAME = "vision_only"
ALGORITHM_NAME = "trainingfreegrpo"

# 실험한 prompt 범위
PROMPT_IDS = ["p1"]

# outputs 폴더
OUTPUT_ROOT = Path("outputs")

# 결과 저장 폴더
EVAL_DIR = Path("evaluation_results") / DATASET_NAME
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = EVAL_DIR / "accuracy_by_prompt_all_rounds.csv"
CHART_PATH = EVAL_DIR / "accuracy_by_prompt_all_rounds.png"


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
    """
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in ["true", "1", "yes", "y"]:
        return True
    if text in ["false", "0", "no", "n"]:
        return False

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


def get_all_round_csvs(prompt_id: str) -> list[Path]:
    """
    특정 prompt의 모든 round CSV 찾기
    """
    prompt_dir = OUTPUT_ROOT / DATASET_NAME / prompt_id / ALGORITHM_NAME

    csv_files = sorted(
        prompt_dir.glob("output_gpt_multichoice_round_*.csv"),
        key=parse_round_number
    )

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾지 못했습니다: {prompt_dir}")

    return csv_files


def compute_accuracy_for_csv(csv_path: Path, prompt_id: str):
    df = pd.read_csv(csv_path)

    total_count = len(df)

    if total_count == 0:
        return {
            "Prompt": prompt_id.upper(),
            "Prompt ID": prompt_id,
            "Round": parse_round_number(csv_path),
            "Total": 0,
            "Correct": 0,
            "Wrong": 0,
            "Missing model_output": 0,
            "Accuracy": 0.0,
            "Answered Accuracy": 0.0,
            "CSV Path": str(csv_path),
        }

    # model_output 누락 개수
    if "model_output" in df.columns:
        missing_model_output_count = df["model_output"].apply(is_missing_text).sum()
    else:
        missing_model_output_count = 0

    # is_correct 컬럼이 있으면 그걸 우선 사용
    if "is_correct" in df.columns:
        is_correct_series = df["is_correct"].apply(normalize_bool_value)
        correct_count = (is_correct_series == True).sum()
        wrong_count = total_count - correct_count
    else:
        # is_correct가 없으면 predicted_final_answer와 reference_answer로 계산
        if "predicted_final_answer" not in df.columns or "reference_answer" not in df.columns:
            raise ValueError(
                f"{csv_path}에 is_correct도 없고 predicted/reference answer 컬럼도 없습니다."
            )

        pred = df["predicted_final_answer"].astype(str).str.strip()
        ref = df["reference_answer"].astype(str).str.strip()
        correct_count = (pred == ref).sum()
        wrong_count = total_count - correct_count

    # 전체 문제 기준 Accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # 답변이 생성된 문제만 기준으로 한 Accuracy
    answered_count = total_count - missing_model_output_count
    answered_accuracy = correct_count / answered_count if answered_count > 0 else 0.0

    return {
        "Prompt": prompt_id.upper(),
        "Prompt ID": prompt_id,
        "Round": parse_round_number(csv_path),
        "Total": total_count,
        "Correct": correct_count,
        "Wrong": wrong_count,
        "Missing model_output": missing_model_output_count,
        "Accuracy": accuracy,
        "Answered Accuracy": answered_accuracy,
        "CSV Path": str(csv_path),
    }


def plot_accuracy(summary_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    # Prompt별로 round에 따른 Accuracy 선 그래프
    for prompt_id in PROMPT_IDS:
        prompt_df = summary_df[summary_df["Prompt ID"] == prompt_id].sort_values("Round")

        if len(prompt_df) == 0:
            continue

        plt.plot(
            prompt_df["Round"],
            prompt_df["Accuracy"],
            marker="o",
            label=prompt_id.upper()
        )

    all_accuracies = summary_df["Accuracy"].dropna()
    y_min = all_accuracies.min()
    y_max = all_accuracies.max()

    # 위아래 여백 자동 설정
    padding = (y_max - y_min) * 0.15

    # 모든 값이 같은 경우 padding이 0이 되는 것 방지
    if padding == 0:
        padding = 0.05

    plt.ylim(
        max(0, y_min - padding),
        min(1.0, y_max + padding)
    )
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by Prompt and Round ({DATASET_NAME}, {ALGORITHM_NAME})")
    plt.xticks(sorted(summary_df["Round"].unique()))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()


def main():
    results = []

    print(f"Dataset: {DATASET_NAME}")
    print(f"Algorithm: {ALGORITHM_NAME}")
    print("-" * 60)

    for prompt_id in PROMPT_IDS:
        round_csvs = get_all_round_csvs(prompt_id)

        for csv_path in round_csvs:
            result = compute_accuracy_for_csv(csv_path, prompt_id)
            results.append(result)

            print(f"{prompt_id.upper()} Round {result['Round']} 평가 완료")
            print(f"  CSV: {csv_path}")
            print(f"  Total: {result['Total']}")
            print(f"  Correct: {result['Correct']}")
            print(f"  Missing model_output: {result['Missing model_output']}")
            print(f"  Accuracy: {result['Accuracy']:.6f}")
            print(f"  Answered Accuracy: {result['Answered Accuracy']:.6f}")
            print()

    summary_df = pd.DataFrame(results)

    # 보기 좋게 정렬
    summary_df = summary_df[
        [
            "Prompt",
            "Prompt ID",
            "Round",
            "Total",
            "Correct",
            "Wrong",
            "Missing model_output",
            "Accuracy",
            "Answered Accuracy",
            "CSV Path",
        ]
    ]

    summary_df = summary_df.sort_values(["Prompt ID", "Round"])

    summary_df.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")
    plot_accuracy(summary_df)

    print("-" * 60)
    print("성능 평가 완료")
    print(f"요약 CSV 저장 위치: {SUMMARY_CSV_PATH}")
    print(f"그래프 저장 위치: {CHART_PATH}")
    print()
    print(
        summary_df[
            [
                "Prompt",
                "Round",
                "Total",
                "Correct",
                "Missing model_output",
                "Accuracy",
                "Answered Accuracy",
            ]
        ]
    )


if __name__ == "__main__":
    main()