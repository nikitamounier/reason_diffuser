import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from generate_vanilla_prm import generate as generateVanillaPRM
from generate import generate as generateRawDiffusion
from generate_backmasking import generate as generateBackMasking
from generate_bachmasking_bon import generate as generateBackMaskingBon
import json
from tqdm import tqdm
import numpy as np
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def setup_models():
    """Set up and return all required models and tokenizers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LLaDA model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    ).to(device)

    # Load PRM model and tokenizer
    prm_model_name = "Qwen/Qwen2.5-Math-PRM-7B"
    prm_tokenizer = AutoTokenizer.from_pretrained(
        prm_model_name, trust_remote_code=True
    )
    prm_model = (
        AutoModel.from_pretrained(
            prm_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    return model, tokenizer, prm_model, prm_tokenizer, device


def evaluate_answer(question, model_answer, ground_truth, model_name):
    """Use Groq API to evaluate if the model's answer is correct."""
    prompt = f"""
You are an expert math evaluator. Your task is to determine if a model's answer to a math problem is correct.

Question: {question}
Model ({model_name}) Answer: {model_answer}
Ground Truth Answer: {ground_truth}

Is the model's answer correct? Consider the following:
1. The final numerical answer must be correct
2. The reasoning should be mathematically sound
3. Minor formatting differences are acceptable

Respond with ONLY "Correct" or "Incorrect" followed by a brief explanation.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        evaluation = response.choices[0].message.content
        return evaluation
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return "Error: Could not evaluate"


def run_math_evaluation():
    """Run evaluation on math problems using different generation methods."""
    # Setup models
    model, tokenizer, prm_model, prm_tokenizer, device = setup_models()

    # Load test data
    df = pd.read_csv("math_test_data.csv")
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    # Limit to a reasonable number for evaluation
    max_problems = 20
    questions = questions[:max_problems]
    answers = answers[:max_problems]

    results = {
        "PRM": {"correct": 0, "total": 0, "evaluations": []},
        "Raw Diffusion": {"correct": 0, "total": 0, "evaluations": []},
        "Backmasking": {"correct": 0, "total": 0, "evaluations": []},
        "Backmasking Bon": {"correct": 0, "total": 0, "evaluations": []},
    }

    for i, (question, answer) in enumerate(
        tqdm(zip(questions, answers), total=len(questions))
    ):
        print(f"\n=== Problem {i+1}/{len(questions)} ===")
        print(f"Question: {question}")
        print(f"Ground Truth Answer: {answer}\n")

        # Format prompt
        m = [{"role": "user", "content": question}]
        formatted_prompt = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(formatted_prompt)["input_ids"]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # Generate with PRM
        prm_out = generateVanillaPRM(
            model,
            input_ids,
            prm_model=prm_model,
            tokenizer=tokenizer,
            prm_tokenizer=prm_tokenizer,
            steps=128,
            gen_length=512,
            block_length=32,
            temperature=0.8,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
        prm_response = tokenizer.batch_decode(
            prm_out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        # Generate with raw diffusion
        raw_out = generateRawDiffusion(
            model,
            input_ids,
            steps=128,
            gen_length=512,
            block_length=32,
            temperature=0.8,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
        raw_response = tokenizer.batch_decode(
            raw_out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        # Generate with backmasking
        backmasking_out = generateBackMasking(
            model,
            input_ids,
            prm_model=prm_model,
            tokenizer=tokenizer,
            prm_tokenizer=prm_tokenizer,
            steps=128,
            gen_length=512,
            block_length=32,
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            backmasking_alpha=5.0,
            backmasking_intensity=0.5,
            global_demasking=True,
            backmasking_frequency=3,
            backmasking_threshold=0.6,
        )
        backmasking_response = tokenizer.batch_decode(
            backmasking_out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        # Generate with backmasking bon
        backmasking_bon_out = generateBackMaskingBon(
            model,
            input_ids,
            prm_model=prm_model,
            tokenizer=tokenizer,
            prm_tokenizer=prm_tokenizer,
            steps=128,
            gen_length=512,
            block_length=32,
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            backmasking_alpha=5.0,
            backmasking_intensity=1,
            global_demasking=True,
            backmasking_frequency=3,
            backmasking_threshold=0.6,
            max_retry_attempts=5,
        )
        backmasking_bon_response = tokenizer.batch_decode(
            backmasking_bon_out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        print("=== Model Responses ===")
        print(f"PRM Response:\n{prm_response}\n")
        print(f"Raw Diffusion Response:\n{raw_response}\n")
        print(f"Backmasking Response:\n{backmasking_response}\n")
        print(f"Backmasking Bon Response:\n{backmasking_bon_response}\n")

        # Evaluate responses
        prm_eval = evaluate_answer(question, prm_response, answer, "PRM")
        raw_eval = evaluate_answer(question, raw_response, answer, "Raw Diffusion")
        backmasking_eval = evaluate_answer(
            question, backmasking_response, answer, "Backmasking"
        )
        backmasking_bon_eval = evaluate_answer(
            question, backmasking_bon_response, answer, "Backmasking Bon"
        )

        print("=== Evaluations ===")
        print(f"PRM: {prm_eval}")
        print(f"Raw Diffusion: {raw_eval}")
        print(f"Backmasking: {backmasking_eval}")
        print(f"Backmasking Bon: {backmasking_bon_eval}")

        # Update results
        for model_name, eval_result in [
            ("PRM", prm_eval),
            ("Raw Diffusion", raw_eval),
            ("Backmasking", backmasking_eval),
            ("Backmasking Bon", backmasking_bon_eval),
        ]:
            results[model_name]["total"] += 1
            if eval_result.startswith("Correct"):
                results[model_name]["correct"] += 1

            results[model_name]["evaluations"].append(
                {
                    "problem_id": i + 1,
                    "question": question,
                    "ground_truth": answer,
                    "model_response": locals()[
                        model_name.lower().replace(" ", "_") + "_response"
                    ],
                    "evaluation": eval_result,
                    "is_correct": eval_result.startswith("Correct"),
                }
            )

        print("=" * 80)

        # Save intermediate results
        with open("math_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Calculate final statistics
    for model_name in results:
        correct = results[model_name]["correct"]
        total = results[model_name]["total"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        results[model_name]["accuracy"] = accuracy
        print(f"{model_name} Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Save final results
    with open("math_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_math_evaluation()
