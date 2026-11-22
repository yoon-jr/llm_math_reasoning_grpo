## question을 함께 input으로 넣어 문제를 푸는 데 필요한 정보를 Visual Dominant의 Image 데이터로부터 정보 추출

# pip install pymetis datasets transformers torch torchvision bitsandbytes openai

## 학교 서버 사용시 GPU 번호 지정
import os
TARGET_GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 데이터셋 로드
import argparse
import json
from pathlib import Path
import datasets
from datasets import load_dataset

_PRE_ARG_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_ARG_PARSER.add_argument("--sampled_id_path", type=str, default="mathverse100/sampled_id.json")
_PRE_ARGS, _ = _PRE_ARG_PARSER.parse_known_args()

def load_sampled_problem_indices(sampled_id_path: str):
    sampled_id_path = Path(sampled_id_path)
    if not sampled_id_path.exists():
        raise FileNotFoundError(f"sampled_id 파일을 찾을 수 없습니다: {sampled_id_path}")

    with sampled_id_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ["problem_index", "problem_indices", "sampled_id", "sampled_ids", "ids"]:
            if key in data:
                data = data[key]
                break
        else:
            values = []
            for value in data.values():
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
            data = values

    if not isinstance(data, list):
        data = [data]

    sampled_indices = set()
    for item in data:
        if isinstance(item, dict):
            item = item.get("problem_index", item.get("id", item.get("sample_id", None)))
        if item is None:
            continue
        sampled_indices.add(int(item))

    if not sampled_indices:
        raise ValueError(f"sampled_id 파일에서 problem_index를 읽지 못했습니다: {sampled_id_path}")
    return sampled_indices

sampled_problem_indices = load_sampled_problem_indices(_PRE_ARGS.sampled_id_path)
dataset = load_dataset("AI4Math/MathVerse", "testmini")
dataset = dataset['testmini']
dataset = dataset.filter(lambda example: example["problem_version"] == "Vision Dominant")

def convert_sample_index_to_int(example):
    example["sample_index"] = int(example["sample_index"])
    example["problem_index"] = int(example["problem_index"])
    return example

dataset = dataset.map(convert_sample_index_to_int)
dataset = dataset.filter(lambda example: int(example["problem_index"]) in sampled_problem_indices)
print("dataset size:", len(dataset))
print("sampled_id_path:", _PRE_ARGS.sampled_id_path)
print("----------------------------------------------------")


## prompt 작성
def build_prompt_with_question_p1(question: str) -> str:
    return f"""You are a geometry problem solver.

You are given:
- A geometry problem question
- A diagram image related to the question

Solve step by step and provide the final answer.

Your task is to analyze the diagram and the question, then produce the output in the EXACT format below.

Output format (must follow strictly):

Information: <numbered list of extracted information from the image, starting from 1, items separated ONLY by a single space>
Solution: <step-by-step reasoning using the extracted information to solve the problem>
Final Answer: <answer>

Rules for Information:
- Extract ONLY information from the image that is necessary to solve the question.
- Use words only (no symbols).
- Use decimal numbers for angles.
- Valid examples:
  - triangle ABC
  - angle A = 80.0
  - angle BCD = 60.0
  - point D is on AB (only when D is not A or B)

Rules for Solution:
- Explain the reasoning step by step in plain English.

Rules for Answer (VERY IMPORTANT):
- Use ONLY the alphabet choices given in the problem (e.g., A B C D).
- If there is exactly one correct answer, output exactly one letter with no spaces or symbols.
  - Example: A
- If there are multiple correct answers, output the letters separated by a single space, with no other symbols.
  - Example: A B

Output constraints:
- Use plain text only.
- Use English only.
- Do not add any extra text outside the specified format.
- Do not add extra blank lines.

Question:
{question}
"""

def build_prompt_with_question_p2(question: str) -> str:
    return f"""You are a geometry problem solver.

You are given:
- A geometry problem question
- A diagram image related to the question

Carefully reason through each step, then give the final answer.

Your task is to analyze the diagram and the question, then produce the output in the EXACT format below.

Output format (must follow strictly):

Information: <numbered list of extracted information from the image, starting from 1, items separated ONLY by a single space>
Solution: <step-by-step reasoning using the extracted information to solve the problem>
Final Answer: <answer>

Rules for Information:
- Extract ONLY information from the image that is necessary to solve the question.
- Use words only (no symbols).
- Use decimal numbers for angles.
- Valid examples:
  - triangle ABC
  - angle A = 80.0
  - angle BCD = 60.0
  - point D is on AB (only when D is not A or B)

Rules for Solution:
- Explain the reasoning step by step in plain English.

Rules for Answer (VERY IMPORTANT):
- Use ONLY the alphabet choices given in the problem (e.g., A B C D).
- If there is exactly one correct answer, output exactly one letter with no spaces or symbols.
  - Example: A
- If there are multiple correct answers, output the letters separated by a single space, with no other symbols.
  - Example: A B

Output constraints:
- Use plain text only.
- Use English only.
- Do not add any extra text outside the specified format.
- Do not add extra blank lines.

Question:
{question}
"""

def build_prompt_with_question_p3(question: str) -> str:
    return f"""You are a geometry problem solver.

You are given:
- A geometry problem question
- A diagram image related to the question

Think through the solution logically before answering.

Your task is to analyze the diagram and the question, then produce the output in the EXACT format below.

Output format (must follow strictly):

Information: <numbered list of extracted information from the image, starting from 1, items separated ONLY by a single space>
Solution: <step-by-step reasoning using the extracted information to solve the problem>
Final Answer: <answer>

Rules for Information:
- Extract ONLY information from the image that is necessary to solve the question.
- Use words only (no symbols).
- Use decimal numbers for angles.
- Valid examples:
  - triangle ABC
  - angle A = 80.0
  - angle BCD = 60.0
  - point D is on AB (only when D is not A or B)

Rules for Solution:
- Explain the reasoning step by step in plain English.

Rules for Answer (VERY IMPORTANT):
- Use ONLY the alphabet choices given in the problem (e.g., A B C D).
- If there is exactly one correct answer, output exactly one letter with no spaces or symbols.
  - Example: A
- If there are multiple correct answers, output the letters separated by a single space, with no other symbols.
  - Example: A B

Output constraints:
- Use plain text only.
- Use English only.
- Do not add any extra text outside the specified format.
- Do not add extra blank lines.

Question:
{question}
"""

def build_prompt_with_question_p4(question: str) -> str:
    return f"""You are a geometry problem solver.

You are given:
- A geometry problem question
- A diagram image related to the question

Derive the answer by explicitly justifying each inference from the extracted information, then provide the final answer.

Your task is to analyze the diagram and the question, then produce the output in the EXACT format below.

Output format (must follow strictly):

Information: <numbered list of extracted information from the image, starting from 1, items separated ONLY by a single space>
Solution: <step-by-step reasoning using the extracted information to solve the problem>
Final Answer: <answer>

Rules for Information:
- Extract ONLY information from the image that is necessary to solve the question.
- Use words only (no symbols).
- Use decimal numbers for angles.
- Valid examples:
  - triangle ABC
  - angle A = 80.0
  - angle BCD = 60.0
  - point D is on AB (only when D is not A or B)

Rules for Solution:
- Explain the reasoning step by step in plain English.

Rules for Answer (VERY IMPORTANT):
- Use ONLY the alphabet choices given in the problem (e.g., A B C D).
- If there is exactly one correct answer, output exactly one letter with no spaces or symbols.
  - Example: A
- If there are multiple correct answers, output the letters separated by a single space, with no other symbols.
  - Example: A B

Output constraints:
- Use plain text only.
- Use English only.
- Do not add any extra text outside the specified format.
- Do not add extra blank lines.

Question:
{question}
"""

def build_prompt_with_question_p5(question: str) -> str:
    return f"""You are a geometry problem solver.

You are given:
- A geometry problem question
- A diagram image related to the question

Work through the solution in small, verifiable steps and keep the reasoning consistent with the extracted information before giving the final answer.

Your task is to analyze the diagram and the question, then produce the output in the EXACT format below.

Output format (must follow strictly):

Information: <numbered list of extracted information from the image, starting from 1, items separated ONLY by a single space>
Solution: <step-by-step reasoning using the extracted information to solve the problem>
Final Answer: <answer>

Rules for Information:
- Extract ONLY information from the image that is necessary to solve the question.
- Use words only (no symbols).
- Use decimal numbers for angles.
- Valid examples:
  - triangle ABC
  - angle A = 80.0
  - angle BCD = 60.0
  - point D is on AB (only when D is not A or B)

Rules for Solution:
- Explain the reasoning step by step in plain English.

Rules for Answer (VERY IMPORTANT):
- Use ONLY the alphabet choices given in the problem (e.g., A B C D).
- If there is exactly one correct answer, output exactly one letter with no spaces or symbols.
  - Example: A
- If there are multiple correct answers, output the letters separated by a single space, with no other symbols.
  - Example: A B

Output constraints:
- Use plain text only.
- Use English only.
- Do not add any extra text outside the specified format.
- Do not add extra blank lines.

Question:
{question}
"""


## GPT API 설정 및 실행 인자
from openai import OpenAI
import argparse
import base64
import json
import re
from io import BytesIO
from PIL import Image
import time
import pandas as pd
from pathlib import Path

DATA_TYPE_KEY = "visual_dominant"
DATA_DISPLAY_NAME = "Visual_Dominant"
ALGORITHM_NAME = "trainingfreegrpo"
QUESTION_CONDITIONED = True

parser = argparse.ArgumentParser()
parser.add_argument("--prompt_name", type=str, default="p1", choices=["p1", "p2", "p3", "p4", "p5"])
parser.add_argument("--model", type=str, default="gpt-5.5")
parser.add_argument("--max_output_tokens", type=int, default=1024)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--sampled_id_path", type=str, default="mathverse100/sampled_id.json")
parser.add_argument("--num_optimization_steps", type=int, default=2)
parser.add_argument("--optimizer_model", type=str, default=None)
parser.add_argument("--optimizer_max_output_tokens", type=int, default=4096)
parser.add_argument("--optimizer_sample_size", type=int, default=30)
parser.add_argument("--sleep_time", type=float, default=0.3)
args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정해주세요. 예: export OPENAI_API_KEY='your_api_key'")
client = OpenAI(api_key=api_key)

PROMPT_BUILDERS = {
    "p1": build_prompt_with_question_p1,
    "p2": build_prompt_with_question_p2,
    "p3": build_prompt_with_question_p3,
    "p4": build_prompt_with_question_p4,
    "p5": build_prompt_with_question_p5,
}

ANSWER_FIELDS = [
    "answer",
    "answer_option",
    "correct_answer",
    "correct_option",
    "label",
    "gt_answer",
    "target",
]

def compact_text(text, limit=1200):
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit] + " ..."
    return text

def normalize_choice_answer(text):
    if text is None:
        return ""
    text = str(text).strip()
    final_match = re.search(r"Final\s*Answer\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if final_match:
        text = final_match.group(1).strip()
    letters = re.findall(r"\b[A-H]\b", text.upper())
    if letters:
        seen = []
        for letter in letters:
            if letter not in seen:
                seen.append(letter)
        return " ".join(seen)
    return re.sub(r"\s+", " ", text.upper()).strip()

def extract_final_answer(model_output):
    return normalize_choice_answer(model_output)

def get_reference_answer(sample):
    for field in ANSWER_FIELDS:
        value = sample.get(field, None)
        if value is not None and str(value).strip() != "":
            normalized = normalize_choice_answer(value)
            if normalized:
                return normalized
    return ""

def check_is_correct(predicted_answer, reference_answer):
    if not predicted_answer or not reference_answer:
        return None
    return predicted_answer == reference_answer

def is_output_format_valid(model_output):
    if not model_output:
        return False
    return all(key in model_output for key in ["Information:", "Solution:", "Final Answer:"])

def clean_optimizer_response(text):
    text = (text or "").strip()
    fence_match = re.search(r"```(?:text|python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    prefixes = [
        "Improved prompt template:",
        "Revised prompt template:",
        "Optimized prompt:",
        "Prompt:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text

def ensure_prompt_constraints(new_prompt, old_prompt):
    new_prompt = clean_optimizer_response(new_prompt)
    if not new_prompt:
        return old_prompt
    if "{question}" in old_prompt and "{question}" not in new_prompt:
        if "Question:" in new_prompt:
            new_prompt = re.sub(r"Question:\s*$", "Question:\n{question}", new_prompt.rstrip(), flags=re.IGNORECASE)
        else:
            new_prompt = new_prompt.rstrip() + "\n\nQuestion:\n{question}\n"
    return new_prompt

def save_prompts(save_dir, prompt_templates):
    payload = {
        "data_type": DATA_TYPE_KEY,
        "algorithm": ALGORITHM_NAME,
        "prompt_name": args.prompt_name,
        "model": args.model,
        "num_optimization_steps": args.num_optimization_steps,
        "prompts": [
            {"round": idx, "prompt_template": prompt}
            for idx, prompt in enumerate(prompt_templates)
        ],
    }
    prompt_path = save_dir / "prompts.json"
    with prompt_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return prompt_path

def build_optimizer_examples(round_outputs, sample_size):
    rows = list(round_outputs)
    wrong_rows = [row for row in rows if row.get("is_correct") is False]
    correct_rows = [row for row in rows if row.get("is_correct") is True]
    unknown_rows = [row for row in rows if row.get("is_correct") is None]

    selected = []
    selected.extend(wrong_rows[:sample_size])
    if len(selected) < sample_size:
        selected.extend(unknown_rows[:sample_size - len(selected)])
    if len(selected) < sample_size:
        selected.extend(correct_rows[:sample_size - len(selected)])
    if not selected:
        selected = rows[:sample_size]

    example_lines = []
    for row in selected:
        example = {
            "problem_index": row.get("problem_index"),
            "sample_index": row.get("sample_index"),
            "question": compact_text(row.get("question_used", ""), 700),
            "reference_answer": row.get("reference_answer", ""),
            "predicted_final_answer": row.get("predicted_final_answer", ""),
            "is_correct": row.get("is_correct"),
            "format_valid": row.get("format_valid"),
            "model_output": compact_text(row.get("model_output", ""), 1000),
        }
        example_lines.append(json.dumps(example, ensure_ascii=False))

    summary = {
        "num_total": len(rows),
        "num_correct": len(correct_rows),
        "num_wrong": len(wrong_rows),
        "num_unknown": len(unknown_rows),
        "num_format_valid": sum(1 for row in rows if row.get("format_valid")),
    }
    return json.dumps(summary, ensure_ascii=False), "\n".join(example_lines)

def call_optimizer_llm(meta_prompt, optimizer_model, max_output_tokens):
    resp = client.responses.create(
        model=optimizer_model,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": meta_prompt}],
        }],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text.strip()

def optimize_prompt_textgrad(current_prompt_template, round_outputs, optimizer_model, max_output_tokens, sample_size):
    summary, examples = build_optimizer_examples(round_outputs, sample_size)
    meta_prompt = f"""You are applying TextGrad-style prompt optimization.
Treat the current prompt template as the trainable text variable. The generation results below are the forward pass. Use natural-language feedback as the textual gradient, then update only the prompt template.

Optimization objective:
- Improve multi-choice geometry problem-solving accuracy.
- Preserve strict output formatting: Information, Solution, Final Answer.
- Preserve modality instructions. If the prompt says text only, do not add image instructions. If the prompt says image, keep image-analysis instructions.
- Preserve the exact placeholder {{question}} if it appears in the current prompt template.
- Keep the prompt in English.
- Do not add dataset-specific memorized answers.

Current prompt template:
<<<CURRENT_PROMPT
{current_prompt_template}
CURRENT_PROMPT>>>

Aggregate result summary:
{summary}

Representative generation results:
{examples}

Return ONLY the revised complete prompt template. Do not include explanations, markdown, or code fences.
"""
    new_prompt = call_optimizer_llm(meta_prompt, optimizer_model, max_output_tokens)
    return ensure_prompt_constraints(new_prompt, current_prompt_template)

def optimize_prompt_trainingfreegrpo(current_prompt_template, round_outputs, optimizer_model, max_output_tokens, sample_size):
    summary, examples = build_optimizer_examples(round_outputs, sample_size)
    meta_prompt = f"""You are applying Training-Free GRPO prompt optimization.
Do not update model parameters. Instead, compare the group of rollouts below, identify semantic advantages of successful or better-formatted outputs over failed outputs, distill reusable experience, and integrate that experience into the prompt template.

Training-Free GRPO objective:
- Use group-relative comparison between correct, wrong, and unknown rollouts.
- Strengthen behaviors that lead to valid reasoning and correct final letter choices.
- Add concise reusable guidance that prevents common failures.
- Preserve strict output formatting: Information, Solution, Final Answer.
- Preserve modality instructions. If the prompt says text only, do not add image instructions. If the prompt says image, keep image-analysis instructions.
- Preserve the exact placeholder {{question}} if it appears in the current prompt template.
- Keep the prompt in English.
- Do not add dataset-specific memorized answers.

Current prompt template:
<<<CURRENT_PROMPT
{current_prompt_template}
CURRENT_PROMPT>>>

Aggregate rollout-group summary:
{summary}

Representative rollout group:
{examples}

Return ONLY the revised complete prompt template. Do not include explanations, markdown, or code fences.
"""
    new_prompt = call_optimizer_llm(meta_prompt, optimizer_model, max_output_tokens)
    return ensure_prompt_constraints(new_prompt, current_prompt_template)

def optimize_prompt(current_prompt_template, round_outputs):
    optimizer_model = args.optimizer_model or args.model
    if ALGORITHM_NAME == "textgrad":
        return optimize_prompt_textgrad(
            current_prompt_template,
            round_outputs,
            optimizer_model=optimizer_model,
            max_output_tokens=args.optimizer_max_output_tokens,
            sample_size=args.optimizer_sample_size,
        )
    if ALGORITHM_NAME == "trainingfreegrpo":
        return optimize_prompt_trainingfreegrpo(
            current_prompt_template,
            round_outputs,
            optimizer_model=optimizer_model,
            max_output_tokens=args.optimizer_max_output_tokens,
            sample_size=args.optimizer_sample_size,
        )
    raise ValueError(f"지원하지 않는 알고리즘입니다: {ALGORITHM_NAME}")

def _to_pil(img_field):
    if img_field is None:
        return None
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, list) and len(img_field) > 0:
        return _to_pil(img_field[0])
    if isinstance(img_field, dict) and "path" in img_field and img_field["path"] is not None:
        return Image.open(img_field["path"]).convert("RGB")
    if isinstance(img_field, str):
        return Image.open(img_field).convert("RGB")
    return None

def pil_to_data_url(pil_img, fmt="PNG"):
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def get_initial_prompt_template(prompt_name: str):
    return PROMPT_BUILDERS[prompt_name]("{question}")

def render_prompt(prompt_template: str, question_text: str):
    return prompt_template.replace("{question}", question_text)

def solve_one_sample(sample, prompt_template: str, model: str, max_output_tokens: int = 1024):
    img = _to_pil(sample.get("image", None))
    if img is None:
        return None
    data_url = pil_to_data_url(img, fmt="PNG")
    question_text = sample.get("question", "")
    prompt = render_prompt(prompt_template, question_text)

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text.strip()


print(f"Visual_Dominant Data 정보 추출 및 문제 풀이 시작")
print(f"model: {args.model}")
print(f"prompt_name: {args.prompt_name}")
print(f"algorithm: {ALGORITHM_NAME}")
print(f"num_optimization_steps: {args.num_optimization_steps}")
print("----------------------------------------------------")

SAVE_ROOT = Path(args.output_dir) if args.output_dir else Path("./outputs")
SAVE_DIR = SAVE_ROOT / DATA_TYPE_KEY / args.prompt_name / ALGORITHM_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

prompt_templates = [get_initial_prompt_template(args.prompt_name)]
save_prompts(SAVE_DIR, prompt_templates)

total_rounds = args.num_optimization_steps + 1
for round_idx in range(total_rounds):
    current_prompt_template = prompt_templates[round_idx]
    model_output = []

    for i in range(len(dataset)):
        sample = dataset[i]
        question_type = sample.get("question_type", "")
        if question_type == "multi-choice":
            try:
                output = solve_one_sample(
                    sample=sample,
                    prompt_template=current_prompt_template,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                )
            except Exception:
                time.sleep(2.0)
                output = solve_one_sample(
                    sample=sample,
                    prompt_template=current_prompt_template,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                )

            predicted_answer = extract_final_answer(output)
            reference_answer = get_reference_answer(sample)
            is_correct = check_is_correct(predicted_answer, reference_answer)
            row = {
                "idx_in_dataset": i,
                "problem_index": sample.get("problem_index", None),
                "sample_index": sample.get("sample_index", None),
                "question_used": sample.get("question", "") if QUESTION_CONDITIONED else "",
                "prompt_name": args.prompt_name,
                "algorithm": ALGORITHM_NAME,
                "round": round_idx,
                "model": args.model,
                "model_output": output,
                "predicted_final_answer": predicted_answer,
                "reference_answer": reference_answer,
                "is_correct": is_correct,
                "format_valid": is_output_format_valid(output),
            }
            model_output.append(row)

            print(f"round {round_idx} | {i+1}/{len(dataset)} 추출 완료")
            time.sleep(args.sleep_time)

    df_extract = pd.DataFrame(model_output)
    extract_csv_path = SAVE_DIR / f"output_gpt_multichoice_round_{round_idx}.csv"
    df_extract.to_csv(extract_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ round {round_idx} model_output CSV 저장 완료: {extract_csv_path}")

    if round_idx < args.num_optimization_steps:
        improved_prompt = optimize_prompt(current_prompt_template, model_output)
        prompt_templates.append(improved_prompt)
        prompt_path = save_prompts(SAVE_DIR, prompt_templates)
        print(f"✅ round {round_idx + 1} prompt 저장 완료: {prompt_path}")

print(f"✅ 전체 완료: {SAVE_DIR}")
