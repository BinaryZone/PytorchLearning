import torch

flag = torch.cuda.is_available()
print(flag)

ngpu = 1

device = torch.device("cuda:0" if flag else 'cpu')

print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

cuda_version = torch.version.cuda
print(cuda_version)

cudnn_version = torch.backends.cudnn.version()
print(cudnn_version)