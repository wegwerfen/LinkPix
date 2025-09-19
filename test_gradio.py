#!/usr/bin/env python3
"""Basic smoke tests for the Gradio Image Generation Server frontend."""

import json
import os
from typing import Dict

import requests


def test_config_loading() -> bool:
    print("🔍 Testing configuration loading...")
    try:
        if not os.path.exists("config.json"):
            print("❌ config.json not found")
            return False
        with open("config.json", "r", encoding="utf-8") as fh:
            config = json.load(fh)
        required_keys = ["comfyui_url", "workflow_path", "output_dir"]
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        print("✅ Configuration loading successful")
        return True
    except Exception as exc:
        print(f"❌ Configuration loading failed: {exc}")
        return False


def test_workflow_files() -> bool:
    print("🔍 Testing workflow files...")
    try:
        workflow_dir = "workflows"
        if not os.path.isdir(workflow_dir):
            print(f"❌ Workflow directory '{workflow_dir}' not found")
            return False
        workflow_files = [f for f in os.listdir(workflow_dir) if f.endswith(".json")]
        if not workflow_files:
            print(f"❌ No workflow files found in '{workflow_dir}'")
            return False
        print(f"✅ Found {len(workflow_files)} workflow files: {', '.join(workflow_files)}")
        return True
    except Exception as exc:
        print(f"❌ Workflow files test failed: {exc}")
        return False


def test_styles_loading() -> bool:
    print("🔍 Testing styles loading...")
    try:
        styles_path = os.path.join("data", "styles.json")
        if os.path.exists(styles_path):
            with open(styles_path, "r", encoding="utf-8") as fh:
                styles = json.load(fh)
            print(f"✅ Loaded {len(styles)} styles: {', '.join(styles.keys())}")
        else:
            print("⚠️ data/styles.json not found, defaults will be used")
        return True
    except Exception as exc:
        print(f"❌ Styles loading failed: {exc}")
        return False


def test_cache_directory() -> bool:
    print("🔍 Testing cache directory...")
    try:
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"✅ Created cache directory: {cache_dir}")
        else:
            print(f"✅ Cache directory exists: {cache_dir}")
        return True
    except Exception as exc:
        print(f"❌ Cache directory test failed: {exc}")
        return False


def test_comfyui_connection() -> bool:
    print("🔍 Testing ComfyUI connection...")
    try:
        with open("config.json", "r", encoding="utf-8") as fh:
            config = json.load(fh)
        comfy_url = config.get("comfyui_url", "http://127.0.0.1:8188")
        try:
            response = requests.get(f"{comfy_url}/system_stats", timeout=5)
            if response.status_code == 200:
                print(f"✅ ComfyUI connection successful: {comfy_url}")
                return True
            print(f"⚠️ ComfyUI responded with status {response.status_code}")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"⚠️ ComfyUI not reachable at {comfy_url}: {exc}")
            print("   This is normal if ComfyUI is not running")
            return False
    except Exception as exc:
        print(f"❌ ComfyUI connection test failed: {exc}")
        return False


def test_flask_api() -> bool:
    print("🔍 Testing Flask API connection...")
    flask_url = "http://127.0.0.1:4000"
    try:
        response = requests.get(f"{flask_url}/prompt/test", timeout=2)
        print(f"✅ Flask API reachable at {flask_url}")
        return True
    except requests.exceptions.RequestException as exc:
        print(f"⚠️ Flask API not reachable at {flask_url}: {exc}")
        print("   This is normal if the Flask server is not running")
        return False
    except Exception as exc:
        print(f"❌ Flask API test failed: {exc}")
        return False


def test_gradio_imports() -> bool:
    print("🔍 Testing Gradio imports...")
    try:
        import gradio  # noqa: F401

        print("✅ Gradio imports successful")
        return True
    except ImportError as exc:
        print(f"❌ Gradio import failed: {exc}")
        print("   Run: pip install gradio")
        return False


def run_tests() -> None:
    print("🚀 Starting Gradio Image Generation Server Tests")
    print("=" * 60)

    tests = [
        ("Dependencies", test_gradio_imports),
        ("Configuration", test_config_loading),
        ("Workflow Files", test_workflow_files),
        ("Styles", test_styles_loading),
        ("Cache Directory", test_cache_directory),
        ("ComfyUI Connection", test_comfyui_connection),
        ("Flask API", test_flask_api),
    ]

    results = []
    for name, func in tests:
        print(f"\n📋 {name}:")
        results.append((name, func()))

    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    passed = 0
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:<20} {status}")
        if ok:
            passed += 1
    print(f"\n🎯 Tests Passed: {passed}/{len(results)}")
    if passed == len(results):
        print("🎉 All tests passed! Your Gradio migration looks good.")
        print("\n🚀 To start the server:")
        print("   1. Start ComfyUI (if not already running)")
        print("   2. Run: python gradio_app.py")
if __name__ == "__main__":
    run_tests()
