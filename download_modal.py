import modal
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("build-essential")
    .pip_install(
        "transformers==4.49.0",
        "torch==2.1.2",
        "numpy==1.26.3",
        "pandas==2.1.4",
        "accelerate==0.26.1",
        "datasets==2.13.0",
        "modelscope==1.9.5",
        "transformers_stream_generator==0.0.4",
        "flask",
        "flask-cors"
    )
)

App = modal.App(name="llada_app", image=image)

vol = modal.Volume.from_name("llada")


# Mount local files (same ones copied in Dockerfile)
app_files = [
    "llada_main.py",
    "generate.py",
    "generate_vanilla_prm.py",
    "math_test_data.csv",
    "generate_backmasking.py",
    "llada_main_bon.py"
]

# Define function that runs the app
@App.function(
    mounts=[modal.Mount.from_local_file(f, remote_path=f"/app/{f}") for f in app_files],
    gpu="A100-80GB",
    timeout=36000,
    volumes={"/my_vol": vol}


)
def download_models():
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    os.environ.update({
        "TRANSFORMERS_CACHE":  "/my_vol",
        "HF_HOME":             "/my_vol",
        "HF_DATASETS_CACHE":   "/my_vol",
        "HF_MODULES_CACHE":    "/my_vol",
        "XDG_CACHE_HOME":      "/my_vol",
    })

    MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
    PRM_NAME   = "Qwen/Qwen2.5-Math-PRM-7B"

    print("Downloading LLaDA model…")
    AutoTokenizer .from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir="/my_vol")
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir="/my_vol")

    print("Downloading PRM model…")
    AutoTokenizer.from_pretrained(PRM_NAME, trust_remote_code=True, cache_dir="/my_vol")
    AutoModel     .from_pretrained(PRM_NAME, trust_remote_code=True, cache_dir="/my_vol")

    print("✅ Models downloaded and cached in /my_vol.")
