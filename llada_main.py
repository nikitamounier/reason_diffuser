# Load required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from generate_vanilla_prm import generate as generateVanillaPRM
from generate import generate as generateRawDiffusion
import pandas as pd
from generate_backmasking import generate as generateBackMasking


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, prm_model, tokenizer, prm_tokenizer):

    df = pd.read_csv("gsm8k_test_data.csv")
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 8:
            break
        print(f"\n=== Problem {i+1} ===")
        print(f"Question: {question}")
        print(f"Ground Truth Answer: {answer}\n")

        # Format prompt for both models
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
            temperature=0.6,
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
            temperature=0.6,
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
            temperature=0.3,
            cfg_scale=0.0,
            remasking="low_confidence",
            backmasking_alpha=5.0,
            backmasking_intensity=0.5,
            global_demasking=True,
            backmasking_frequency=3,
            backmasking_threshold=0.4,
        )
        backmasking_response = tokenizer.batch_decode(
            backmasking_out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        print("=== Model Responses ===")
        print(f"PRM Response:\n{prm_response}\n")
        print(f"Raw Diffusion Response:\n{raw_response}\n")
        print(f"Backmasking Response:\n{backmasking_response}\n")
        print("=" * 80)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer and model from saved files
    tokenizer = torch.load("/data/tokenizer.pt")
    model = torch.load("/data/model.pt")
    model = model.to(device)

    # Load PRM tokenizer and model from saved files
    prm_tokenizer = torch.load("/data/prm_tokenizer.pt")
    prm_model = torch.load("/data/prm_model.pt")
    run_inference(model, prm_model, tokenizer, prm_tokenizer)
