# Load required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from generate_bachmasking_bon import generate as generateBackMaskingBon
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and tokenizers
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


@app.post("/generate")
async def generate(request: Request):
    # Get prompt from request body
    body = await request.json()
    prompt = body.get("prompt")

    if not prompt:
        return {"error": "No prompt provided"}

    # Format prompt
    m = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(formatted_prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

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
        temperature=0.3,
        cfg_scale=0.0,
        remasking="low_confidence",
        backmasking_alpha=5.0,
        backmasking_intensity=0.5,
        global_demasking=True,
        backmasking_frequency=3,
        backmasking_threshold=0.4,
        max_retry_attempts=5,
    )

    response = tokenizer.batch_decode(
        backmasking_bon_out[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
