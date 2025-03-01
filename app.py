from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import time
import threading
import importlib.util

# Import the generate_backmasking module
spec = importlib.util.spec_from_file_location(
    "generate_backmasking", "generate_backmasking.py"
)
generate_backmasking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_backmasking)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models and tokenizers
model = None
tokenizer = None
prm_model = None
prm_tokenizer = None

# Global variables for storing generation state
current_generations = {}

def load_models():
    global model, tokenizer, prm_model, prm_tokenizer
    
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the main model for text diffusion (matching llada_main.py)
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    
    # Load the PRM model for scoring (matching llada_main.py)
    prm_model_name = "Qwen/Qwen2.5-Math-PRM-7B"
    prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
    prm_model = (
        AutoModel.from_pretrained(
            prm_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    
    print("Models loaded successfully!")

# Custom wrapper for the generate function to capture steps
class GenerationMonitor:
    def __init__(self, generation_id):
        self.generation_id = generation_id
        self.steps = []
        self.completed = False
        self.text = ""
        self.error = None
    
    def add_step(self, step_data):
        self.steps.append(step_data)
    
    def mark_completed(self, final_text):
        self.completed = True
        self.text = final_text
    
    def mark_error(self, error_msg):
        self.error = error_msg
        self.completed = True

# Modified generate function that reports progress
def generate_with_monitoring(generation_id, prompt_text, **kwargs):
    try:
        # Create a monitor for this generation
        monitor = GenerationMonitor(generation_id)
        current_generations[generation_id] = monitor
        
        # Format the prompt using chat template (matching llada_main.py)
        m = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        
        # Tokenize the prompt
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Prepare for hooking into the generate function
        original_get_backmasking_tokens = generate_backmasking.get_backmasking_tokens
        original_compute_block_score = generate_backmasking.compute_block_score
        original_calculate_backmasking_probs = generate_backmasking.calculate_backmasking_probs
        
        # Override the function to capture masked tokens at each step
        def custom_get_backmasking_tokens(*args, **kwargs):
            result = original_get_backmasking_tokens(*args, **kwargs)
            # Capture the state here
            block_indices, block_probs = args[0], args[1]
            
            step_data = {
                "type": "backmasking",
                "timestamp": time.time(),
                "mask": result.cpu().numpy().tolist(),
                "block_probs": block_probs.tolist() if isinstance(block_probs, np.ndarray) else block_probs,
            }
            monitor.add_step(step_data)
            return result
        
        # Override the function to capture block scores
        def custom_compute_block_score(*args, **kwargs):
            block_text, prompt_text = args[0], args[1]
            score = original_compute_block_score(*args, **kwargs)
            
            step_data = {
                "type": "block_score",
                "timestamp": time.time(),
                "block_text": block_text,
                "score": score,
            }
            monitor.add_step(step_data)
            return score
        
        # Override the function to capture backmasking probabilities
        def custom_calculate_backmasking_probs(*args, **kwargs):
            block_scores = args[0]
            probs = original_calculate_backmasking_probs(*args, **kwargs)
            
            step_data = {
                "type": "backmasking_probs",
                "timestamp": time.time(),
                "block_scores": block_scores,
                "probs": probs.tolist() if isinstance(probs, np.ndarray) else probs,
            }
            monitor.add_step(step_data)
            return probs
        
        # Replace the functions with our instrumented versions
        generate_backmasking.get_backmasking_tokens = custom_get_backmasking_tokens
        generate_backmasking.compute_block_score = custom_compute_block_score
        generate_backmasking.calculate_backmasking_probs = custom_calculate_backmasking_probs
        
        try:
            # Call the generate function
            result = generate_backmasking.generate(
                model=model,
                prompt=input_ids,
                prm_model=prm_model,
                tokenizer=tokenizer,
                prm_tokenizer=prm_tokenizer,
                **kwargs
            )
            
            # Decode the result
            final_text = tokenizer.decode(result[0, input_ids.shape[1]:], skip_special_tokens=True)
            monitor.mark_completed(final_text)
            
        finally:
            # Restore the original functions
            generate_backmasking.get_backmasking_tokens = original_get_backmasking_tokens
            generate_backmasking.compute_block_score = original_compute_block_score
            generate_backmasking.calculate_backmasking_probs = original_calculate_backmasking_probs
            
    except Exception as e:
        monitor.mark_error(str(e))
        raise e

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt_text = data.get('prompt', '')
    parameters = data.get('parameters', {})
    
    # Generate a unique ID for this generation
    generation_id = str(int(time.time()))
    
    # Start generation in a background thread
    threading.Thread(
        target=generate_with_monitoring,
        args=(generation_id, prompt_text),
        kwargs=parameters
    ).start()
    
    return jsonify({
        "id": generation_id,
        "status": "started"
    })

@app.route('/api/status/<generation_id>', methods=['GET'])
def get_status(generation_id):
    if generation_id not in current_generations:
        return jsonify({"status": "not_found"}), 404
    
    monitor = current_generations[generation_id]
    
    if monitor.error:
        return jsonify({
            "status": "error",
            "error": monitor.error
        })
    
    if monitor.completed:
        return jsonify({
            "status": "completed",
            "text": monitor.text,
            "steps": monitor.steps
        })
    
    # If not completed, return steps so far
    return jsonify({
        "status": "in_progress",
        "steps": monitor.steps
    })

@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    # Return the default parameters for the generate function (matching generate_backmasking.py)
    defaults = {
        "steps": 128,
        "gen_length": 512,
        "block_length": 32,
        "temperature": 0.3,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "backmasking_alpha": 5.0,
        "backmasking_intensity": 0.5,
        "global_demasking": True,
        "backmasking_frequency": 3,
        "backmasking_threshold": 0.4
    }
    return jsonify(defaults)

if __name__ == '__main__':
    # Load models when the app starts
    from waitress import serve
    load_models()
    serve(app, host='0.0.0.0', port=80) 