import torch

print("=" * 50)
print("ResumeLens GPU Check")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_properties = torch.cuda.get_device_properties(0)

    total_vram = gpu_properties.total_memory / (1024**3)

    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.2f} GB")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Selected device: {device}")

else:
    device = torch.device("cpu")

    print("No CUDA GPU detected.")
    print(f"Selected device: {device}")