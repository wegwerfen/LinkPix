#!/usr/bin/env python3
"""
Launcher script to start both Flask API and Gradio interface simultaneously
"""

import subprocess
import sys
import time
import os
import signal
import logging
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessManager:
    def __init__(self):
        self.flask_process = None
        self.gradio_process = None
        self.running = True

    def start_flask(self):
        """Start Flask API server"""
        try:
            logging.info("Starting Flask API server...")
            self.flask_process = subprocess.Popen(
                [sys.executable, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream Flask output
            for line in iter(self.flask_process.stdout.readline, ''):
                if not self.running:
                    break
                print(f"[FLASK] {line.rstrip()}")
                
        except Exception as e:
            logging.error(f"Error starting Flask: {e}")

    def start_gradio(self):
        """Start Gradio web interface"""
        try:
            # Wait a moment for Flask to start
            time.sleep(2)
            logging.info("Starting Gradio web interface...")

            self.gradio_process = subprocess.Popen(
                [sys.executable, "gradio_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream Gradio output
            for line in iter(self.gradio_process.stdout.readline, ''):
                if not self.running:
                    break
                print(f"[GRADIO] {line.rstrip()}")

        except Exception as e:
            logging.error(f"Error starting Gradio: {e}")

    def start_both(self):
        """Start both services in separate threads"""
        flask_thread = Thread(target=self.start_flask, daemon=True)
        gradio_thread = Thread(target=self.start_gradio, daemon=True)

        flask_thread.start()
        gradio_thread.start()
        
        logging.info("🚀 Both services starting...")
        logging.info("📡 Flask API will be available at: http://localhost:4000")  
        logging.info("🌐 Gradio UI will be available using the configured port (default 8501)")
        logging.info("💡 Press Ctrl+C to stop both services")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
                # Check if processes are still running
                if self.flask_process and self.flask_process.poll() is not None:
                    logging.error("Flask process died unexpectedly")
                    break
                    
                if self.gradio_process and self.gradio_process.poll() is not None:
                    logging.error("Gradio process died unexpectedly")  
                    break
                    
        except KeyboardInterrupt:
            logging.info("🛑 Shutdown requested...")
            self.stop_all()

    def stop_all(self):
        """Stop both services"""
        self.running = False
        
        if self.flask_process:
            logging.info("Stopping Flask server...")
            self.flask_process.terminate()
            try:
                self.flask_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flask_process.kill()
                
        if self.gradio_process:
            logging.info("Stopping Gradio server...")
            self.gradio_process.terminate()
            try:
                self.gradio_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gradio_process.kill()
                
        logging.info("✅ All services stopped")

def main():
    """Main entry point"""
    print("🎨 Image Generation Server Launcher")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found")
        sys.exit(1)
        
    if not os.path.exists("gradio_app.py"):
        print("❌ Error: gradio_app.py not found")
        sys.exit(1)
        
    if not os.path.exists("config.json"):
        print("❌ Error: config.json not found")
        sys.exit(1)
    
    manager = ProcessManager()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        logging.info("Signal received, shutting down...")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        manager.start_both()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        manager.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
