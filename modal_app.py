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
    "llada_main_bon.py", 
    "generate_gmini.py"
]

# Define function that runs the app
@App.function(
    mounts=[modal.Mount.from_local_file(f, remote_path=f"/app/{f}") for f in app_files],
    gpu="A100-80GB",
    timeout=36000,
    volumes={"/my_vol": vol}


)
def run_llada():
    import subprocess
    subprocess.run(["python3", "/app/llada_main_bon.py"], check=True)