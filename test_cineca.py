import torch

for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
        #print(f"Device properties: {torch.cuda.get_device_properties()}")
        print(f"Is available: {torch.cuda.is_available()}")
