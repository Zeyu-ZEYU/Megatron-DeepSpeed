import torch

a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor_split(a, [2, 5])


for i in range(1, 0):
    print(i)
