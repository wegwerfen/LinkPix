#!/usr/bin/env python3
"""
ComfyUI Connection Diagnostic Script

This script helps diagnose ComfyUI connection issues.
"""

import requests
import json
import time
import socket
from urllib.parse import urlparse

def test_basic_connection(url):
    """Test basic HTTP connection"""
    print(f"🔍 Testing basic connection to {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ Basic connection successful - Status: {response.status_code}")
        return True
    except requests.exceptions.ConnectException as e:
        print(f"❌ Connection refused: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Connection timeout: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_port_open(host, port):
    """Test if port is open"""
    print(f"🔍 Testing if port {port} is open on {host}")
    
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"✅ Port {port} is open")
            return True
    except socket.error as e:
        print(f"❌ Port {port} is not accessible: {e}")
        return False

def test_comfyui_endpoints(base_url):
    """Test specific ComfyUI endpoints"""
    print(f"🔍 Testing ComfyUI-specific endpoints at {base_url}")
    
    endpoints_to_test = [
        "/system_stats",
        "/queue",
        "/history",
        "/object_info",
        "/"
    ]
    
    results = {}
    
    for endpoint in endpoints_to_test:
        url = base_url + endpoint
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK ({response.status_code})")
                results[endpoint] = True
            else:
                print(f"⚠️ {endpoint}: Status {response.status_code}")
                results[endpoint] = False
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
            results[endpoint] = False
    
    return results

def test_from_config():
    """Test connection using config.json settings"""
    print("🔍 Testing connection using config.json settings")
    
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        comfyui_url = config.get("comfyui_url", "http://127.0.0.1:8188")
        print(f"📋 Config ComfyUI URL: {comfyui_url}")
        
        # Parse URL
        parsed = urlparse(comfyui_url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 8188
        
        print(f"📋 Parsed - Host: {host}, Port: {port}")
        
        # Test port
        port_open = test_port_open(host, port)
        
        # Test basic connection
        basic_connection = test_basic_connection(comfyui_url)
        
        # Test ComfyUI endpoints if basic connection works
        if basic_connection:
            endpoint_results = test_comfyui_endpoints(comfyui_url)
        else:
            endpoint_results = {}
        
        return {
            'url': comfyui_url,
            'port_open': port_open,
            'basic_connection': basic_connection,
            'endpoints': endpoint_results
        }
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def suggest_solutions(results):
    """Suggest solutions based on test results"""
    print("\n" + "="*60)
    print("💡 SUGGESTIONS:")
    
    if not results:
        print("❌ Could not run tests - check if config.json exists")
        return
    
    if not results['port_open']:
        print("🔧 Port is not accessible:")
        print("   • Make sure ComfyUI is running")
        print("   • Check if ComfyUI is using the correct port")
        print("   • Verify no firewall is blocking the port")
        print("   • Try: netstat -an | findstr :8188")
    
    if not results['basic_connection']:
        print("🔧 Basic connection failed:")
        print("   • ComfyUI may not be started")
        print("   • Check ComfyUI console for errors")
        print("   • Try accessing http://127.0.0.1:8188 in browser")
        print("   • Restart ComfyUI")
    
    if results['basic_connection'] and not any(results['endpoints'].values()):
        print("🔧 Connection works but endpoints fail:")
        print("   • ComfyUI may still be starting up")
        print("   • Check ComfyUI version compatibility")
        print("   • Look at ComfyUI logs for errors")
    
    if all(results['endpoints'].get(ep, False) for ep in ['/system_stats']):
        print("✅ ComfyUI appears to be working correctly!")
        print("   • The Gradio monitor tab should show 'Connected' status")
        print("   • If it still shows 'Disconnected', try refreshing the page")

def run_diagnostics():
    """Run all diagnostics"""
    print("🚀 ComfyUI Connection Diagnostics")
    print("="*60)
    
    # Test from config
    results = test_from_config()
    
    # Additional manual tests
    print(f"\n🔍 Additional Tests:")
    
    # Test common ComfyUI ports
    common_ports = [8188, 8080, 3000, 5000]
    for port in common_ports:
        if test_port_open('127.0.0.1', port):
            print(f"   • Found service on port {port}")
            test_basic_connection(f"http://127.0.0.1:{port}")
    
    # Suggest solutions
    suggest_solutions(results)
    
    print(f"\n📋 Summary:")
    if results:
        print(f"   • ComfyUI URL: {results['url']}")
        print(f"   • Port Open: {'✅' if results['port_open'] else '❌'}")
        print(f"   • Basic Connection: {'✅' if results['basic_connection'] else '❌'}")
        print(f"   • Working Endpoints: {sum(results['endpoints'].values())}/{len(results['endpoints'])}")

if __name__ == "__main__":
    run_diagnostics()
