"""
Easy script to start the MLflow model server
"""
import os
import subprocess
import sys

# Configuration
# Updated to use the newly built deployable model that accepts raw text input
MODEL_PATH = "runs:/5144b1cf3f03442d8dd1ba8417b8a292/sentiment_model"
PORT = 5001  # Using 5001 since 5000 is typically used by MLflow UI
MLFLOW_TRACKING_URI = r"C:\Users\Atharv\Desktop\projectss\mlops\mlops_project_group1\src\mlruns"

def start_server():
    """Start the MLflow model serving server"""
    
    # Set environment variables
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = f"file:///{MLFLOW_TRACKING_URI.replace(os.sep, '/')}"
    
    print("="*80)
    print("🚀 Starting MLflow Model Server")
    print("="*80)
    print(f"\n📦 Model Path: {MODEL_PATH}")
    print(f"🌐 Port: {PORT}")
    print(f"📊 MLflow Tracking URI: {env['MLFLOW_TRACKING_URI']}")
    print("\n" + "="*80)
    print("\n✨ Server will start shortly...")
    print("\n📝 To test the server, open a NEW terminal and run:")
    print(f"   python test_model_server.py --port {PORT}")
    print("\n⏹️  To stop the server, press Ctrl+C")
    print("\n" + "="*80 + "\n")
    
    try:
        # Start the MLflow server
        cmd = [
            "mlflow", "models", "serve",
            "-m", MODEL_PATH,
            "-p", str(PORT),
            "--env-manager=local"
        ]
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
