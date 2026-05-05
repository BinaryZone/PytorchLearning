import torch

print(torch.__version__)        # PyTorch 版本
print(torch.version.cuda)       # 编译该 PyTorch 时使用的 CUDA 版本
print(torch.cuda.is_available()) # 通常为 False

ngpu = 1
device = torch.device()