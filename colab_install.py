# Colab-Specific Installation Script
# Run this in Google Colab BEFORE installing requirements.txt

# 1. Downgrade NumPy to 1.x (required for OpenCV compatibility)
!pip uninstall -y numpy
!pip install "numpy<2.0"

# 2. Install requirements
!pip install -r requirements.txt

# 3. Verify installation
import numpy as np
import cv2
import torch

print(f"✅ NumPy version: {np.__version__}")
print(f"✅ OpenCV version: {cv2.__version__}")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
