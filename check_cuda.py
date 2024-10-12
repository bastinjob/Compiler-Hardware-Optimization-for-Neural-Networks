import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("GPU is available and can be used.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available, GPU cannot be used.")

