# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This is a Flask-based image generation server that acts as a proxy/cache layer for ComfyUI. The main application (`app.py`) receives text prompts via HTTP routes and generates images using ComfyUI workflows.

### Core Components

- **Flask Server** (`app.py`): HTTP server with single endpoint `/prompt/<prompt>` for image generation
- **ComfyUI Integration**: Communicates with ComfyUI API running on localhost:8188 by default
- **Workflow System**: Uses JSON workflow files to define image generation parameters
- **User-based Caching**: Images cached by user ID (hashed from IP + User-Agent) in `cache/` directory
- **Size Optimization**: Maps requested dimensions to predefined standard sizes, then resizes output

### Key Functions

- `load_workflow()`: Loads and configures ComfyUI workflow JSON with prompt, seed, and dimensions
- `find_nearest_standard_size()`: Maps custom dimensions to optimal standard sizes for the model
- `generate_image()`: Main route handler that orchestrates the generation process

## Configuration

Server configuration is in `config.json`:
- `comfyui_url`: ComfyUI server endpoint (default: http://127.0.0.1:8188)
- `workflow_path`: Path to workflow JSON file
- `output_dir`: Directory for cached images (default: cache)
- `timeout`: Generation timeout in seconds
- `image_format`: Output format (png)

## Development Commands

### Running the Server
```bash
python3 app.py
```
Server runs on http://0.0.0.0:4000

### Dependencies
The application requires:
- Flask
- requests

Install with:
```bash
pip3 install flask requests
```

### API Usage
Generate images by making GET requests to:
```
http://localhost:4000/prompt/<your_prompt_here>?width=640&height=384
```

## Workflow System

The server uses ComfyUI workflow JSON files located in `workflows/`. The default workflow `flux_schnell.json` defines:
- FLUX Schnell model loading
- Text encoding for prompts
- Image generation parameters
- Resize and padding operations

When modifying workflows, ensure these node types are present for proper integration:
- `CLIPTextEncode` nodes for prompt processing
- `EmptySD3LatentImage` for dimension control
- `ResizeAndPadImage` for final sizing
- `SaveImage` for output

## Cache Structure

Images are cached in user-specific directories under `cache/`:
```
cache/
├── {user_hash_1}/
│   ├── {prompt_hash_1}.png
│   └── {prompt_hash_2}.png
└── {user_hash_2}/
    └── {prompt_hash_3}.png
```

Cache keys are generated from prompt text combined with target dimensions.