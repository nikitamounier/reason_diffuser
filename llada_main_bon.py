import torch
import pandas as pd
import re
import time
import os
import traceback # For more detailed error reporting
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# --- Import your generation functions ---
# Ensure these files are in the same directory or your Python path
try:
    from generate import generate as generateRawDiffusion # Assuming generate.py is Raw Diffusion
    from generate_vanilla_prm import generate as generateVanillaPRM
    from generate_backmasking import generate as generateBackMasking
    print("Successfully imported generation functions.")
except ImportError as e:
    print(f"Error importing generation functions: {e}")
    print("Please ensure generate.py, generate_vanilla_prm.py, and generate_backmasking.py are accessible.")
    exit()

# --- Configuration ---
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
PRM_MODEL_NAME = "Qwen/Qwen2.5-Math-PRM-7B"
DATASET_PATH = "math_test_data.csv"
RESULTS_CSV_PATH = "math_evaluation_results_detailed.csv" # Output file path
MAX_QUESTIONS = None # Set to an integer (e.g., 10) to limit questions, or None to run all
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generation Parameters (Adjust as needed)
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
    "backmasking_alpha": 5.0,
    "backmasking_intensity": 0.5,
    "global_demasking": True, # Set based on your backmasking function's needs
    "backmasking_frequency": 3,
    "backmasking_threshold": 0.6, # Adjust threshold as needed
    "max_retry_attempts": 5, # Added based on your backmasking function
}


# --- Helper Function for Scoring ---
def extract_boxed_answer(text):
    """Extracts the content within the last \\boxed{...} environment."""
    if not isinstance(text, str): # Handle potential errors where response is not string
        return None
    # Find all occurrences of \boxed{...}
    matches = re.findall(r"\\boxed\{(.*?)\}", text, re.DOTALL) # Added re.DOTALL to handle multi-line answers
    if matches:
        # Return the last one found, stripped of leading/trailing whitespace
        return matches[-1].strip()
    return None

def is_correct(generated_response, ground_truth_answer):
    """
    Compares the final boxed answer from the generated response
    with the ground truth answer's boxed content.

    NOTE: This is a basic implementation for demonstration.
          Robust math evaluation often requires more sophisticated
          parsing, normalization (e.g., removing commas, simplifying
          fractions), and potentially symbolic comparison.
          ***RESULTS FROM THIS FUNCTION SHOULD BE MANUALLY VERIFIED.***
    """
    gen_ans = extract_boxed_answer(generated_response)
    gt_ans = extract_boxed_answer(ground_truth_answer)

    # print(f"  [Scoring] Gen Box: '{gen_ans}', GT Box: '{gt_ans}'") # Debug print

    if gen_ans is None:
        # print("  [Scoring] Failed: Box not found in generation.") # Debug
        return False # Cannot score if generation has no box
    if gt_ans is None:
        # print("  [Scoring] Warning: Box not found in ground truth. Cannot determine correctness.") # Debug
        return False # Cannot score if ground truth has no box

    # Basic string comparison after stripping whitespace
    # TODO: Implement more robust comparison (numerical, symbolic) if needed
    is_match = gen_ans == gt_ans
    # print(f"  [Scoring] Match: {is_match}") # Debug
    return is_match

# --- Main Evaluation Function ---
def run_evaluation(model, tokenizer, prm_model, prm_tokenizer):
    """Loads data, runs inference, scores, saves results, and reports summary."""

    print(f"\nLoading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return
    try:
        df = pd.read_csv(DATASET_PATH)
        if "question" not in df.columns or "answer" not in df.columns:
            print("Error: Dataset must contain 'question' and 'answer' columns.")
            return
        questions = df["question"].tolist()
        answers = df["answer"].tolist()
        print(f"Loaded {len(questions)} questions.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Limit questions if MAX_QUESTIONS is set
    if MAX_QUESTIONS is not None and MAX_QUESTIONS < len(questions):
        print(f"Limiting evaluation to the first {MAX_QUESTIONS} questions.")
        questions = questions[:MAX_QUESTIONS]
        answers = answers[:MAX_QUESTIONS]

    total_questions = len(questions)
    if total_questions == 0:
        print("No questions to evaluate.")
        return

    # List to store results for each question before creating DataFrame
    all_results_list = []

    # Evaluation loop
    for i, (question, ground_truth) in enumerate(zip(questions, answers)):
        print(f"\n{'='*20} Problem {i+1}/{total_questions} {'='*20}")
        print(f"Question: {question[:200]}{'...' if len(question)>200 else ''}")

        # Dictionary to store results for the current question
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

        # Format prompt
        m = [{"role": "user", "content": question}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(DEVICE)
        except Exception as e:
            print(f"Error formatting prompt for question {i+1}: {e}")
            # Store error and skip generation for this question
            error_msg = f"ERROR (Prompt Formatting): {e}"
            current_q_results["raw_response"] = error_msg
            current_q_results["prm_response"] = error_msg
            current_q_results["backmasking_response"] = error_msg
            all_results_list.append(current_q_results)
            continue # Skip to next question

        # --- Run Raw Diffusion ---
        print("\nRunning Raw Diffusion...")
        start_time = time.time()
        try:
            raw_out = generateRawDiffusion(
                model=model,
                prompt=input_ids,
                tokenizer=tokenizer, # Pass tokenizer if needed by the function
                **GEN_PARAMS_SHARED,
                **GEN_PARAMS_RAW,
            )
            raw_response = tokenizer.decode(
                raw_out[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            current_q_results["raw_response"] = raw_response
            current_q_results["raw_time"] = time.time() - start_time
            current_q_results["raw_correct_auto"] = is_correct(raw_response, ground_truth)
            print(f"Raw Diffusion Time: {current_q_results['raw_time']:.2f}s")
            print(f"Raw Correct (Auto): {current_q_results['raw_correct_auto']}")
        except Exception as e:
            print(f"Error during Raw Diffusion generation for question {i+1}:\n{traceback.format_exc()}")
            current_q_results["raw_response"] = f"ERROR (Generation): {e}"
            current_q_results["raw_time"] = time.time() - start_time
            current_q_results["raw_correct_auto"] = False


        # --- Run Vanilla PRM ---
        print("\nRunning Vanilla PRM...")
        start_time = time.time()
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
            current_q_results["prm_time"] = time.time() - start_time
            current_q_results["prm_correct_auto"] = is_correct(prm_response, ground_truth)
            print(f"Vanilla PRM Time: {current_q_results['prm_time']:.2f}s")
            print(f"PRM Correct (Auto): {current_q_results['prm_correct_auto']}")
        except Exception as e:
            print(f"Error during Vanilla PRM generation for question {i+1}:\n{traceback.format_exc()}")
            current_q_results["prm_response"] = f"ERROR (Generation): {e}"
            current_q_results["prm_time"] = time.time() - start_time
            current_q_results["prm_correct_auto"] = False


        # --- Run Backmasking ---
        print("\nRunning Backmasking...")
        start_time = time.time()
        try:
            # Make sure all required args are passed
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
            current_q_results["backmasking_time"] = time.time() - start_time
            current_q_results["backmasking_correct_auto"] = is_correct(bm_response, ground_truth)
            print(f"Backmasking Time: {current_q_results['backmasking_time']:.2f}s")
            print(f"Backmasking Correct (Auto): {current_q_results['backmasking_correct_auto']}")
        except Exception as e:
            print(f"Error during Backmasking generation for question {i+1}:\n{traceback.format_exc()}")
            current_q_results["backmasking_response"] = f"ERROR (Generation): {e}"
            current_q_results["backmasking_time"] = time.time() - start_time
            current_q_results["backmasking_correct_auto"] = False

        # Add results for this question to the main list
        all_results_list.append(current_q_results)

    # --- Process and Save Results ---
    print(f"\n{'='*30} Processing Results {'='*30}")
    if not all_results_list:
        print("No results were generated.")
        return

    results_df = pd.DataFrame(all_results_list)

    # Calculate summary statistics
    summary = {}
    for method in ["raw", "prm", "backmasking"]:
        correct_col = f"{method}_correct_auto"
        time_col = f"{method}_time"
        # Count only valid boolean True values for correctness
        num_correct = results_df[correct_col].sum()
        # Calculate average time only for successful runs (time >= 0)
        valid_times = results_df[results_df[time_col] >= 0][time_col]
        avg_time = valid_times.mean() if not valid_times.empty else 0
        summary[method] = {"correct": num_correct, "avg_time": avg_time}

    # Save detailed results to CSV
    try:
        results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8')
        print(f"\nDetailed results saved to: {RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"\nError saving detailed results to CSV: {e}")

    # --- Report Final Summary ---
    print(f"\n{'='*30} Evaluation Summary {'='*30}")
    print(f"Total Questions Evaluated: {total_questions}")
    print("\n--- Accuracy (Based on Automatic Scoring) ---")
    print("*** WARNING: Automatic scoring is basic (boxed string match).")
    print("*** Please manually verify correctness using the saved CSV file. ***")
    for method, data in summary.items():
        accuracy = (data["correct"] / total_questions) * 100 if total_questions > 0 else 0
        print(f"{method.capitalize():<15}: {int(data['correct']):>3}/{total_questions} ({accuracy:.2f}%)")

    print("\n--- Average Generation Time (Successful Runs) ---")
    for method, data in summary.items():
         print(f"{method.capitalize():<15}: {data['avg_time']:.2f} seconds/question")

    print("=" * (60 + len(" Evaluation Summary ")))


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # --- Load Models and Tokenizers ---
    print("\nLoading main model and tokenizer...")
    # (Loading code remains the same as your previous version)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16 # Optional
        )
        model = model.to(DEVICE).eval()
        print("Main model loaded.")
    except Exception as e:
        print(f"Error loading main model '{MODEL_NAME}': {e}")
        exit()

    print("\nLoading PRM model and tokenizer...")
    try:
        prm_tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_NAME, trust_remote_code=True)
        prm_model = (
            AutoModel.from_pretrained(
                PRM_MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .to(DEVICE)
            .eval()
        )
        print("PRM model loaded.")
    except Exception as e:
        print(f"Error loading PRM model '{PRM_MODEL_NAME}': {e}")
        print("Cannot proceed without PRM model for PRM and Backmasking methods.")
        exit()

    # --- Run the evaluation ---
    run_evaluation(model, tokenizer, prm_model, prm_tokenizer)

    print("\nScript finished.")