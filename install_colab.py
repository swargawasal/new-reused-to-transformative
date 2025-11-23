# ============================================================
# GOOGLE COLAB INSTALLATION SCRIPT - FIXED
# Run this FIRST before running main.py
# ============================================================

print("🚀 Installing YouTube Automation Bot for Google Colab...")
print("=" * 60)

# Step 1: Install FFmpeg
print("\n📦 Step 1/6: Installing FFmpeg...")
import subprocess
subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
subprocess.run(['apt-get', 'install', '-y', 'ffmpeg', '-qq'], check=True, capture_output=True)
print("✅ FFmpeg installed")

# Step 2: Fix NumPy (CRITICAL for OpenCV compatibility)
print("\n📦 Step 2/6: Fixing NumPy compatibility...")
subprocess.run(['pip', 'uninstall', '-y', 'numpy'], check=True, capture_output=True)
subprocess.run(['pip', 'install', 'numpy<2.0'], check=True, capture_output=True)
print("✅ NumPy 1.x installed")

# Step 3: Install Python dependencies
print("\n📦 Step 3/6: Installing Python dependencies...")
subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'], check=True, capture_output=True)
print("✅ Dependencies installed")

# Step 4: Fix basicsr import error (CRITICAL!)
print("\n📦 Step 4/6: Patching basicsr for torchvision compatibility...")
import site
import os

# Find basicsr installation
site_packages = site.getsitepackages()[0]
degradations_file = os.path.join(site_packages, 'basicsr', 'data', 'degradations.py')

if os.path.exists(degradations_file):
    with open(degradations_file, 'r') as f:
        content = f.read()
    
    # Fix the import
    content = content.replace(
        'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
        'from torchvision.transforms.functional import rgb_to_grayscale'
    )
    
    with open(degradations_file, 'w') as f:
        f.write(content)
    
    print("✅ basicsr patched successfully")
else:
    print("⚠️ basicsr not found, skipping patch")

# Step 5: Verify installation
print("\n📦 Step 5/6: Verifying installation...")
import numpy as np
import cv2
import torch

print(f"  ✅ NumPy: {np.__version__}")
print(f"  ✅ OpenCV: {cv2.__version__}")
print(f"  ✅ PyTorch: {torch.__version__}")
print(f"  ✅ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  ✅ GPU Memory: {gpu_memory:.2f} GB")
else:
    print("  ⚠️ No GPU detected - AI enhancement will be disabled")

# Step 6: Setup environment variables
print("\n📦 Step 6/6: Setting up environment...")
print("Please configure your .env file with:")
print("  - TELEGRAM_BOT_TOKEN")
print("  - IG_USERNAME")
print("  - IG_PASSWORD")
print("\nYou can use Colab Secrets or create .env manually")

print("\n" + "=" * 60)
print("✅ Installation complete!")
print("⚠️ IMPORTANT: Ignore NumPy/scipy dependency warnings")
print("   (They won't affect the bot's functionality)")
print("\nRun: !python main.py")
print("=" * 60)
