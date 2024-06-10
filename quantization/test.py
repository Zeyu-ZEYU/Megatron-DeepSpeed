# import torch

# table = torch.tensor([10, 39, 18, 23], dtype=torch.float, device="cuda:0")

# compressed = torch.tensor([[0, 1, 2, 1, 1, 1], [3, 3, 3, 3, 3, 1]], dtype=torch.uint8, device=table.device)

# a = table[compressed]

# print(a)

import torch

a = torch.ones([2, 2], dtype=torch.int8, device="cuda:0")
b = torch.ones([2, 2], dtype=torch.int8, device="cuda:0")

torch.matmul(a, b)
