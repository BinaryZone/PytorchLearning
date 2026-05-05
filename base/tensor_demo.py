import torch
import numpy as np

# 一、如何初始化
# 1、使用数据创建
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data: {x_data}")

# 2、使用numpy array创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np: {x_np}")

# 3、使用已有张量创建
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# 4、使用随机数据或者常量创建
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 二、张量的属性
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 三、张量的操作
# 1、张量转移到CPU
tensor = torch.ones(4, 4)
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
print(f"Device tensor is stored on: {tensor.device}")

# 2、索引和切片操作
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 3、连接张量
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 4、运算
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(tensor)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# 原地操作
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# 四、与Numpy的转换
# 1、转换成numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 2、numpy转换成张量
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")