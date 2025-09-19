# LinkPix

Generate AI images via simple URL prompts with this ComfyUI server. Visit `/prompt/your-description` to create images instantly. Features Flask backend, Gradio web dashboard for workflow management, prompt styles, user-based caching, and comprehensive monitoring tools for production use.

## Features

- **🔗 URL-Based Generation** - Create images by visiting `/prompt/your-description` - no complex API calls needed
- **🎨 Modern Web Interface** - Clean Gradio dashboard with intuitive controls and visual organization
- **🚀 Simple API** - RESTful endpoints with URL parameters for width, height, and styles
- **🔄 Workflow Management** - Dynamic ComfyUI workflow loading and configuration
- **🎭 Style System** - Reusable prompt styles with pre/post prompts for consistent generation
- **📱 Smart Settings** - Intelligent form controls based on ComfyUI object info and workflow placeholders
- **🌐 Network Access** - Optional network accessibility for remote usage and SillyTavern integration
- **💾 User-Based Caching** - Automatic image caching with user-specific directories
- **🔧 ComfyUI Integration** - Dynamic model, sampler, and LoRA detection with real-time updates
- **📊 System Monitoring** - Real-time service health, cache metrics, and configuration overview

## Quick Start

### Prerequisites

- Python 3.8 or higher
- ComfyUI running on localhost:8188 (or configured URL)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd linkpix
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the servers:
```bash
# Option 1: Use the launcher script (recommended)
python start_server.py

# Option 2: Start manually
python app.py         # Flask API (port 4000)
python gradio_app.py  # Web interface (port 8501 by default)
```

5. Access the web interface at `http://localhost:8501`

## API Usage

### Basic Image Generation

```http
GET /prompt/[IMAGE_PROMPT]?width=[WIDTH]&height=[HEIGHT]&style=[STYLE]
```

**Example:**
```
http://localhost:4000/prompt/beautiful%20sunset?width=768&height=1024&style=kawaii
```

### Parameters

- **`[IMAGE_PROMPT]`** (required) - Your image description (URL encoded)
- **`width`** (optional) - Image width in pixels (uses workflow default if not specified)
- **`height`** (optional) - Image height in pixels (uses workflow default if not specified)
- **`style`** (optional) - Style name from data/styles.json (uses your default style if not specified)

## Configuration

### Server Configuration

Edit `config.json` to configure server settings:

```json
{
  "comfyui_url": "http://127.0.0.1:8188",
  "workflow_path": "workflows/flux_schnell.json",
  "output_dir": "cache",
  "timeout": 120,
  "image_format": "png",
  "web_port": 8501,
  "network_access": false
}
```

### Workflow Management

1. **Workflow Files**: Place ComfyUI workflow JSON files in the `workflows/` directory
2. **Placeholders**: Use placeholders in your workflows for dynamic values:
   - `%prompt%` - User's image prompt
   - `%width%`, `%height%` - Image dimensions
   - `%model%`, `%vae%`, `%lora%` - Model selections
   - `%sampler%`, `%scheduler%` - Sampling settings
   - `%steps%`, `%cfg%`, `%denoise%` - Generation parameters
   - `%img_format%` - Output image format

3. **Workflow Settings**: Configure default values in the web interface under Dashboard → Advanced

### Style System

Create reusable prompt styles in `data/styles.json`:

```json
{
  "none": {
    "pre": "",
    "post": ""
  },
  "kawaii": {
    "pre": "kawaii style",
    "post": "cute, adorable, brightly colored, cheerful, anime influence"
  }
}
```

## SillyTavern Integration

LinkPix is designed to work seamlessly with SillyTavern for character and scene generation:

1. **Enable Network Access**: In the web interface, go to Settings → Server Configuration and enable "Allow other devices to access the UI"
2. **Configure SillyTavern**: Use the network URL provided in the Dashboard (e.g., `http://192.168.1.100:4000/prompt/`)
3. **Character Images**: Generate character portraits and scenes by describing them in prompts
4. **Consistent Styles**: Use the style system to maintain consistent art styles across your characters

**Example SillyTavern Usage:**
```
http://your-server-ip:4000/prompt/medieval knight with silver armor?width=512&height=768&style=kawaii
```

## Web Interface

The Gradio web interface provides:

- **🏠 Dashboard** - API endpoint display, workflow management, and style configuration
- **🎨 Generate** - Interactive image generation with real-time parameters
- **📊 Monitor** - System status, service health, cache metrics, and configuration overview
- **⚙️ Settings** - Server configuration, workflow management, and style editing
- **🔧 Advanced Tools** - Placeholder configuration, workflow editing, and cache management

## Architecture

### Settings Hierarchy

The system follows a clear priority order:

**User Request → Workflow Settings → Workflow File**

1. **User Request**: URL parameters override everything
2. **Workflow Settings**: Configured defaults in the web interface
3. **Workflow File**: Hardcoded values in the JSON workflow

### Directory Structure

```
linkpix/
├── app.py                 # Flask API server
├── gradio_app.py          # Gradio web interface
├── start_server.py        # Dual server launcher
├── config.json            # Server configuration
├── data/                  # Data files
│   ├── styles.json        # Prompt styles
│   ├── user_preferences.json # User preferences (auto-generated)
│   └── placeholders.json  # Available placeholder definitions
├── workflows/             # ComfyUI workflow files
│   ├── settings/          # Workflow-specific settings
│   ├── originals/         # Backup of original workflows
│   └── *.json             # Workflow files
├── cache/                 # Generated images (user-specific)
└── requirements.txt       # Python dependencies
```

## Development

### Running in Development Mode

1. Start ComfyUI
2. Run the Flask API: `python app.py`
3. Run the Gradio interface: `python gradio_app.py`
4. Access the web interface at `http://localhost:8501`

### Adding New Workflows

1. Export your ComfyUI workflow as JSON
2. Replace hardcoded values with placeholders (e.g., `%prompt%`, `%model%`)
3. Place the file in the `workflows/` directory
4. Configure default settings in the web interface

### Extending the API

The Flask API is easily extensible. Key functions:

- `load_workflow()` - Loads and processes workflow files
- `apply_style_to_prompt()` - Applies style transformations
- `load_workflow_settings()` - Loads workflow-specific settings

## Troubleshooting

### Common Issues

**ComfyUI Connection Failed**
- Ensure ComfyUI is running on the configured URL
- Check firewall settings
- Verify the `comfyui_url` in `config.json`

**Images Not Generating**
- Check ComfyUI logs for errors
- Verify workflow placeholders match your workflow file
- Ensure required models are available in ComfyUI

**Web Interface Not Loading**
- Check if port 8501 is available
- Verify `web_port` in `config.json` and that the port is free

### Logs

- Flask API logs appear in the console where `app.py` is running
- Gradio logs appear in the console where `gradio_app.py` is running
- Check ComfyUI logs for workflow execution issues

## Roadmap

### Planned Features

- [ ] **Enhanced URL Style Parameters** - Expand style system to support inline style definitions and style combinations via URL parameters
- [ ] **Dynamic Model/Workflow Selection** - Allow users to specify different ComfyUI models and workflows directly through URL parameters (e.g., `?model=flux-dev&workflow=portrait`)

### Future Enhancements

- [ ] Batch image generation support
- [ ] WebSocket integration for real-time generation status
- [ ] Advanced caching strategies with TTL and size limits
- [ ] Plugin system for custom post-processing
- [ ] REST API documentation with OpenAPI/Swagger
- [ ] Docker containerization for easy deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your chosen license here]

## Changelog

### v0.1.0 (Initial Release)

- ✅ Flask API server with image generation endpoints
- ✅ Gradio web interface for configuration
- ✅ Workflow management system with placeholder support
- ✅ Style system for reusable prompt templates
- ✅ ComfyUI integration with dynamic model detection
- ✅ User-based caching system
- ✅ Network access configuration
- ✅ Comprehensive settings management
