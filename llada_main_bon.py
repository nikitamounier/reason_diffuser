# Load required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from generate_backmasking_bon import generate as generateBackMaskingBon

# Load required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from generate_vanilla_prm import generate as generateVanillaPRM
from generate import generate as generateRawDiffusion
import pandas as pd
from generate_backmasking import generate as generateBackMasking
from generate_backmasking_bon import generate as generateBackMaskingBon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
)

model = model.to(device)
prm_model_name = "Qwen/Qwen2.5-Math-PRM-7B"
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
prm_model = (
    AutoModel.from_pretrained(
        prm_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    .to(device)
    .eval()
)
# Save models and tokenizers to /data
torch.save(model.state_dict(), "/data/model.pt")
torch.save(tokenizer, "/data/tokenizer.pt")
torch.save(prm_model.state_dict(), "/data/prm_model.pt")
torch.save(prm_tokenizer, "/data/prm_tokenizer.pt")


def run_inference():
    # Load test data

    df = pd.read_csv("math_test_data.csv")
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
        print("=" * 80)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    model = model.to(device)
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
    run_inference()
