
import os
import hashlib
import json
import time
import random
import logging
from typing import List
from flask import Flask, request, send_file, Response, render_template_string
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = "config.json"
PLACEHOLDERS_PATH = os.path.join("data", "placeholders.json")
DEFAULT_PLACEHOLDERS = [
    "prompt",
    "negative_prompt",
    "model",
    "vae",
    "sampler",
    "scheduler",
    "steps",
    "cfg",
    "denoise",
    "clip_skip",
    "width",
    "height",
    "seed",
    "img_format",
    "lora",
    "lora_1",
    "lora_2",
]

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

config = load_config()
app = Flask(__name__)
OUTPUT_DIR = config.get("output_dir", "cache")
COMFY_URL = config["comfyui_url"]
WORKFLOW_PATH = config["workflow_path"]
TIMEOUT = config.get("timeout", 30)
IMAGE_FORMAT = config.get("image_format", "png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_styles():
    """Load styles from styles.json"""
    try:
        styles_path = os.path.join("data", "styles.json")
        if os.path.exists(styles_path):
            with open(styles_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load styles: {e}")
    
    # Return default if file doesn't exist or can't be loaded
    return {"none": {"pre": "", "post": ""}}

def get_default_style():
    """Get the default style from user preferences"""
    try:
        preferences_path = os.path.join("data", "user_preferences.json")
        if os.path.exists(preferences_path):
            with open(preferences_path, 'r') as f:
                prefs = json.load(f)
                return prefs.get("default_style", "none")
    except Exception as e:
        logging.warning(f"Could not load user preferences: {e}")
    
    return "none"

def apply_style_to_prompt(prompt, style_name="none"):
    """Apply style pre and post prompts to the user prompt"""
    styles = load_styles()
    
    if style_name not in styles:
        style_name = "none"
    
    style = styles[style_name]
    pre = style.get("pre", "").strip()
    post = style.get("post", "").strip()
    
    # Build the final prompt: [pre] [user_prompt], [post]
    final_prompt = prompt.strip()
    
    if pre:
        final_prompt = f"{pre} {final_prompt}"
    
    if post:
        final_prompt = f"{final_prompt}, {post}"
    
    return final_prompt

def load_workflow_settings(workflow_path):
    """Load workflow settings from the settings file"""
    try:
        workflow_name = os.path.splitext(os.path.basename(workflow_path))[0]
        settings_path = os.path.join("workflows", "settings", f"{workflow_name}-settings.json")
        
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load workflow settings: {e}")
    
    # Return default settings if file doesn't exist or can't be loaded
    return {
        "width": 512,
        "height": 512
    }


def load_placeholders() -> List[str]:
    try:
        if os.path.exists(PLACEHOLDERS_PATH):
            with open(PLACEHOLDERS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                candidates = data.get("placeholders", [])
            else:
                candidates = data
            if isinstance(candidates, list):
                cleaned = [
                    str(item).strip()
                    for item in candidates
                    if isinstance(item, str) and item.strip()
                ]
                if cleaned:
                    return cleaned
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_PLACEHOLDERS.copy()

def get_user_id(req):
    ip = req.remote_addr or "127.0.0.1"
    ua = req.headers.get("User-Agent", "")
    return hashlib.sha256((ip + ua).encode()).hexdigest()

def prompt_to_filename(prompt, width=512, height=512):
    combined = f"{prompt}_{width}x{height}"
    return hashlib.sha256(combined.encode()).hexdigest() + f".{IMAGE_FORMAT}"

def load_workflow(prompt, seed=None, target_width=None, target_height=None, style_name="none"):
    with open(WORKFLOW_PATH, 'r') as f:
        workflow = json.load(f)

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 0xffffffffffffffff)

    # Apply style to the prompt (server-side only, user prompt unchanged)
    styled_prompt = apply_style_to_prompt(prompt, style_name)

    # Load workflow settings for default dimensions
    workflow_settings = load_workflow_settings(WORKFLOW_PATH)
    
    # Use target dimensions if provided, otherwise use workflow settings defaults
    if target_width is None:
        target_width = workflow_settings.get("width", 512)
    if target_height is None:
        target_height = workflow_settings.get("height", 512)

    # Update workflow parameters by replacing placeholders
    # Helper function to safely escape JSON strings
    def escape_json_string(value):
        if isinstance(value, str):
            return json.dumps(value)[1:-1]  # Remove the surrounding quotes
        return str(value)
    
    workflow_str = json.dumps(workflow)
    
    replacements = {
        "prompt": styled_prompt,
        "seed": seed,
        "width": target_width,
        "height": target_height,
    }

    for key, value in workflow_settings.items():
        if key == "__fields__":
            continue
        replacements.setdefault(key, value)

    placeholder_names = set(load_placeholders())
    placeholder_names.update(replacements.keys())

    for key in placeholder_names:
        value = replacements.get(key)
        placeholder = f"%{key}%"
        if placeholder not in workflow_str:
            continue
        if value is None:
            replacement = ""
        elif isinstance(value, str):
            replacement = escape_json_string(value)
        else:
            replacement = str(value)
        workflow_str = workflow_str.replace(placeholder, replacement)
    
    # Convert back to dictionary
    workflow = json.loads(workflow_str)

    return workflow

@app.route("/prompt/<path:prompt>")
def generate_image(prompt):
    logging.info(f"Received image generation request for prompt: {prompt}")
    prompt = prompt.replace("+", " ")
    user_id = get_user_id(request)
    logging.info(f"User ID: {user_id}")
    
    # Parse width and height from query parameters (None if not provided)
    target_width = int(request.args.get('width')) if request.args.get('width') else None
    target_height = int(request.args.get('height')) if request.args.get('height') else None
    
    # Parse style parameter (use default style if not provided)
    style_name = request.args.get('style')
    if style_name is None:
        style_name = get_default_style()
        logging.info(f"Using default style: {style_name}")
    else:
        logging.info(f"Using requested style: {style_name}")
    
    if style_name != 'none':
        logging.info(f"Applying style: {style_name}")
    
    # Load workflow settings to get actual dimensions for filename
    workflow_settings = load_workflow_settings(WORKFLOW_PATH)
    actual_width = target_width if target_width is not None else workflow_settings.get("width", 512)
    actual_height = target_height if target_height is not None else workflow_settings.get("height", 512)
    
    filename = prompt_to_filename(prompt, actual_width, actual_height)
    
    # Generate random seed for each request (used for generation, not filename)
    seed = random.randint(0, 0xffffffffffffffff)
    user_dir = os.path.join(OUTPUT_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, filename)

    # Note: With random seed, cache will rarely hit, but keeping for future use
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="image/png")

    workflow = load_workflow(prompt, seed, target_width, target_height, style_name)

    try:
        logging.info(f"Sending request to ComfyUI at {COMFY_URL}")
        res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
        response_data = res.json()
        logging.info(f"ComfyUI response: {response_data}")
        if "prompt_id" not in response_data:
            logging.error(f"ComfyUI response missing prompt_id: {response_data}")
            return Response(f"ComfyUI response missing prompt_id: {response_data}", status=500)
        prompt_id = response_data["prompt_id"]
        logging.info(f"Got prompt_id: {prompt_id}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to ComfyUI at {COMFY_URL}: {str(e)}")
        return Response(f"Failed to connect to ComfyUI at {COMFY_URL}: {str(e)}", status=500)
    except Exception as e:
        logging.error(f"Error sending prompt to ComfyUI: {str(e)}")
        return Response(f"Error sending prompt to ComfyUI: {str(e)}", status=500)

    elapsed = 0
    poll_interval = 1
    while elapsed < TIMEOUT:
        try:
            history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
            print(f"History response: {history}")
            
            if prompt_id in history and "outputs" in history[prompt_id]:
                outputs = history[prompt_id]["outputs"]
                print(f"Found outputs: {outputs}")
                
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for file_info in node_output["images"]:
                            print(f"File info: {file_info}")
                            
                            # Download image from ComfyUI using /view API
                            try:
                                view_url = f"{COMFY_URL}/view"
                                params = {
                                    "filename": file_info["filename"],
                                    "type": file_info.get("type", "output"),
                                    "subfolder": file_info.get("subfolder", "")
                                }
                                print(f"Downloading from: {view_url} with params: {params}")
                                
                                img_response = requests.get(view_url, params=params)
                                if img_response.status_code == 200:
                                    print(f"Successfully downloaded image, saving to: {filepath}")
                                    with open(filepath, 'wb') as f:
                                        f.write(img_response.content)
                                    return send_file(filepath, mimetype="image/png")
                                else:
                                    print(f"Failed to download from ComfyUI: {img_response.status_code}")
                            except Exception as download_error:
                                print(f"Download error: {download_error}")
        except Exception as e:
            print(f"Error checking history: {e}")
            pass
        time.sleep(poll_interval)
        elapsed += poll_interval

    return Response("Image generation timed out", status=504)

# ---- Health / Admin ----
@app.route('/health', methods=['GET'])
def health_check():
    """Lightweight endpoint for service health checks."""
    payload = {
        "status": "ok",
        "timestamp": time.time(),
    }
    return Response(json.dumps(payload), mimetype="application/json")


# ---- Admin / Utilities ----
@app.route('/admin/restart', methods=['POST', 'GET'])
def admin_restart():
    """Restart the Flask API process in-place.
    Returns 202 immediately and then re-executes the process.
    """
    try:
        logging.info("Restart requested via /admin/restart")

        def _do_restart():
            import os, sys, time
            time.sleep(0.5)  # let the response flush
            python = sys.executable
            os.execv(python, [python, 'app.py'])

        from threading import Thread
        t = Thread(target=_do_restart, daemon=True)
        t.start()

        from flask import jsonify
        return jsonify({'status': 'restarting'}), 202
    except Exception as e:
        logging.error(f"Failed to restart: {e}")
        return Response(f"Failed to restart: {e}", status=500)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)
