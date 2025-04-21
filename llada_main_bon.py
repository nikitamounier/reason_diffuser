import torch
import pandas as pd
import re
import time
import os
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
# force all HF caches into /my_vol
os.environ.update({
    "TRANSFORMERS_CACHE":    "/my_vol",
    "HF_HOME":               "/my_vol",
    "HF_DATASETS_CACHE":     "/my_vol",
    "HF_MODULES_CACHE":      "/my_vol",
    "XDG_CACHE_HOME":        "/my_vol",
})


import functools
print = functools.partial(print, flush=True)


# --- Import generation functions ---
try:
    from generate_vanilla_prm import generate as generateVanillaPRM
    from generate_gmini import generate_prm as generateBackMasking
    print("Successfully imported generation functions.")
except ImportError as e:
    print(f"Error importing generation functions: {e}")
    exit()

# --- Configuration ---
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
PRM_MODEL_NAME = "Qwen/Qwen2.5-Math-PRM-7B"
DATASET_PATH = "/app/math_test_data.csv"
RESULTS_CSV_PATH = "/app/my_vol/math_evaluation_results_detailed.csv"
MAX_QUESTIONS = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEN_PARAMS_SHARED = {
    "steps": 128,
    "gen_length": 512,
    "block_length": 32,
    "cfg_scale": 0.0,
    "remasking": "low_confidence",
}
GEN_PARAMS_RAW = {"temperature": 0.8}
GEN_PARAMS_PRM = {"temperature": 0.8}
GEN_PARAMS_BM = {
    "temperature": 0.7,
    "backmasking_frequency": 5,           # How often to apply backmasking (every N blocks)
    "backmasking_threshold": 0.8,         # Score threshold under which backmasking triggers
    "backmasking_intensity": 0.5,         # Proportion of tokens to backmask
    "global_demasking": True,             # True = demask all at once, False = block-by-block
    "max_retry_attempts": 5,              # Retry block N times until it reaches threshold
    "backmasking_alpha": 5.0,             # Controls how aggressive score→prob mapping is
}


# --- Answer extraction ---
def extract_boxed_answer(text):
    if not isinstance(text, str):
        return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return matches[-1].strip() if matches else None

def is_correct(generated_response, ground_truth_answer):
    gen_ans = extract_boxed_answer(generated_response)
    gt_ans = extract_boxed_answer(ground_truth_answer)
    if gen_ans is None or gt_ans is None:
        return False
    return gen_ans == gt_ans

# --- Evaluation ---
def run_evaluation(model, tokenizer, prm_model, prm_tokenizer):
    print(f"\nLoading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
    try:
        df = pd.read_csv(DATASET_PATH)
        if "question" not in df.columns or "answer" not in df.columns:
            print("Error: CSV must contain 'question' and 'answer' columns.")
            return
        questions = df["question"].tolist()
        answers = df["answer"].tolist()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if MAX_QUESTIONS:
        questions = questions[:MAX_QUESTIONS]
        answers = answers[:MAX_QUESTIONS]

    all_results_list = []

    for i, (question, ground_truth) in enumerate(zip(questions, answers)):
        print(f"\n{'='*20} Problem {i+1}/{len(questions)} {'='*20}")
        print(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}")

        current_q_results = {
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "raw_response": "N/A",
            "raw_time": -1.0,
            "raw_correct_auto": None,
            "prm_response": "N/A",
            "prm_time": -1.0,
            "prm_correct_auto": None,
            "backmasking_response": "N/A",
            "backmasking_time": -1.0,
            "backmasking_correct_auto": None,
        }

        try:
            prompt_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(prompt_input, return_tensors="pt").input_ids.to(DEVICE)
        except Exception as e:
            error_msg = f"ERROR (Prompt Formatting): {e}"
            print(error_msg)
            current_q_results["raw_response"] = error_msg
            current_q_results["prm_response"] = error_msg
            current_q_results["backmasking_response"] = error_msg
            all_results_list.append(current_q_results)
            continue

        # --- Vanilla PRM ---
        print("\nRunning Vanilla PRM...")
        start = time.time()
        try:
            prm_out = generateVanillaPRM(
                model=model,
                prompt=input_ids,
                prm_model=prm_model,
                tokenizer=tokenizer,
                prm_tokenizer=prm_tokenizer,
                **GEN_PARAMS_SHARED,
                **GEN_PARAMS_PRM,
            )
            prm_response = tokenizer.decode(
                prm_out[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            current_q_results["prm_response"] = prm_response
            current_q_results["prm_time"] = time.time() - start
            current_q_results["prm_correct_auto"] = is_correct(prm_response, ground_truth)
            print(f"\n--- Vanilla PRM Output ---\n{prm_response.strip()}\n")
            print(f"PRM Correct (Auto): {current_q_results['prm_correct_auto']}")
        except Exception as e:
            print(f"Error during Vanilla PRM:\n{traceback.format_exc()}")
            current_q_results["prm_response"] = f"ERROR (Generation): {e}"
            current_q_results["prm_time"] = time.time() - start
            current_q_results["prm_correct_auto"] = False

        # --- Backmasking ---
        print("\nRunning Backmasking...")
        start = time.time()
        try:
            bm_out = generateBackMasking(
                model=model,
                prompt=input_ids,
                prm_model=prm_model,
                tokenizer=tokenizer,
                prm_tokenizer=prm_tokenizer,
                **GEN_PARAMS_SHARED,
                **GEN_PARAMS_BM,
            )
            bm_response = tokenizer.decode(
                bm_out[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            current_q_results["backmasking_response"] = bm_response
            current_q_results["backmasking_time"] = time.time() - start
            current_q_results["backmasking_correct_auto"] = is_correct(bm_response, ground_truth)
            print(f"\n--- Backmasking Output ---\n{bm_response.strip()}\n")
            print(f"Backmasking Correct (Auto): {current_q_results['backmasking_correct_auto']}")
        except Exception as e:
            print(f"Error during Backmasking:\n{traceback.format_exc()}")
            current_q_results["backmasking_response"] = f"ERROR (Generation): {e}"
            current_q_results["backmasking_time"] = time.time() - start
            current_q_results["backmasking_correct_auto"] = False

        all_results_list.append(current_q_results)

    print(f"\n{'='*30} Processing Results {'='*30}")
    if not all_results_list:
        print("No results were generated.")
        return

    df = pd.DataFrame(all_results_list)
    summary = {}
    for method in ["raw", "prm", "backmasking"]:
        c_col, t_col = f"{method}_correct_auto", f"{method}_time"
        summary[method] = {
            "correct": df[c_col].sum(),
            "avg_time": df[df[t_col] >= 0][t_col].mean()
        }

    try:
        df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"Results saved to: {RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print(f"\n{'='*30} Evaluation Summary {'='*30}")
    print(f"Total Questions Evaluated: {len(questions)}")
    print("\n--- Accuracy ---")
    for method, data in summary.items():
        acc = (data["correct"] / len(questions)) * 100
        print(f"{method.capitalize():<15}: {data['correct']:>3}/{len(questions)} ({acc:.2f}%)")
    print("\n--- Avg Time ---")
    for method, data in summary.items():
        print(f"{method.capitalize():<15}: {data['avg_time']:.2f}s")

# --- Entry ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir="/my_vol",
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir="/my_vol",
        local_files_only=True,
    ).to(DEVICE).eval()

    prm_tokenizer = AutoTokenizer.from_pretrained(
        PRM_MODEL_NAME,
        trust_remote_code=True,
        cache_dir="/my_vol",
        local_files_only=True,
    )
    prm_model = AutoModel.from_pretrained(
        PRM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir="/my_vol",
        local_files_only=True,
    ).to(DEVICE).eval()

    run_evaluation(model, tokenizer, prm_model, prm_tokenizer)
    print("Script finished.")
