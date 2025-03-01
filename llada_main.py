# Load required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from generate import generate


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
        prm_model=prm_model,
        tokenizer=tokenizer,
        prm_tokenizer=prm_tokenizer,
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
