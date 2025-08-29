#!/usr/bin/env python3
"""
Launcher script for Task 8 Resume-Job Matching Streamlit App
"""

import subprocess
import sys
import os

def main():
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the app
    app_path = os.path.join(script_dir, "task8_resume_screening", "app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ App not found at: {app_path}")
        return
    
    print("🚀 Starting Resume-Job Matching App...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
