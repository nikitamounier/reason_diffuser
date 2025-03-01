# Load required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate import generate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
)
model = model.to(device)
prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"


def run_inference(prompt):
    m = [
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )

    input_ids = tokenizer(formatted_prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    return tokenizer.batch_decode(
        out[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]


# Read prompts from file
with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# Run inference for each prompt
for i, prompt in enumerate(prompts, 1):
    print(f"\nPrompt {i}:")
    print(f"Q: {prompt}")
    print(f"A: {run_inference(prompt)}")
