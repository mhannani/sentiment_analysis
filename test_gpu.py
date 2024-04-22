import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    num_devices = torch.cuda.device_count()
    print("Number of available GPUs:", num_devices)
    for i in range(num_devices):
        print("GPU", i, ":", torch.cuda.get_device_name(i))

    # Get the current CUDA memory usage
    memory_stats = torch.cuda.memory_stats()

    # Get the available memory in bytes
    available_memory_bytes = memory_stats["allocated_bytes.all.peak"] - memory_stats["allocated_bytes.all.current"]

    # Convert bytes to gigabytes (GB)
    available_memory_gb = available_memory_bytes / (1024 ** 3)

    print("Available GPU memory:", available_memory_gb, "GB")

else:
    print("CUDA is not available. Training will be performed on CPU.")
    
    
    
import psutil

# Get the total RAM capacity in bytes
total_ram_bytes = psutil.virtual_memory().total

# Convert bytes to gigabytes (GB)
total_ram_gb = total_ram_bytes / (1024 ** 3)

print("Total RAM capacity:", total_ram_gb, "GB")