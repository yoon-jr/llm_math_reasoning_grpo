from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
DATASET_NAME = "vision_only"
ALGORITHM_NAME = "trainingfreegrpo"

PROMPT_IDS = ["p1"]

EVAL_DIR = Path("evaluation_results") / DATASET_NAME

SUMMARY_CSV_PATH = EVAL_DIR / "accuracy_by_prompt_all_rounds.csv"
ANSWERED_CHART_PATH = EVAL_DIR / "answered_accuracy_by_prompt_all_rounds.png"


def plot_answered_accuracy(summary_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    # Prompt별로 round에 따른 Answered Accuracy 선 그래프
    for prompt_id in PROMPT_IDS:
        prompt_df = summary_df[summary_df["Prompt ID"] == prompt_id].sort_values("Round")

        if len(prompt_df) == 0:
            continue

        plt.plot(
            prompt_df["Round"],
            prompt_df["Answered Accuracy"],
            marker="o",
            label=prompt_id.upper()
        )

    # y축 범위 자동 조정
    all_values = summary_df["Answered Accuracy"].dropna()

    y_min = all_values.min()
    y_max = all_values.max()

    padding = (y_max - y_min) * 0.15

    if padding == 0:
        padding = 0.05

    plt.ylim(
        max(0, y_min - padding),
        min(1.0, y_max + padding)
    )

    plt.xlabel("Round")
    plt.ylabel("Answered Accuracy")
    plt.title(f"Answered Accuracy by Prompt and Round ({DATASET_NAME}, {ALGORITHM_NAME})")
    plt.xticks(sorted(summary_df["Round"].unique()))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANSWERED_CHART_PATH, dpi=300)
    plt.close()


def main():
    summary_df = pd.read_csv(SUMMARY_CSV_PATH)

    plot_answered_accuracy(summary_df)

    print("Answered Accuracy 그래프 저장 완료")
    print(f"저장 위치: {ANSWERED_CHART_PATH}")


if __name__ == "__main__":
    main()