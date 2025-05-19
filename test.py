import torch

if torch.cuda.is_available():
    print("✅ CUDA is available. GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA is NOT available. Training will run on CPU.")
