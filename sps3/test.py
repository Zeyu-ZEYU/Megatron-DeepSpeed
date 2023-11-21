import math

import torch

arr = torch.tensor([[7, 2], [3, 4]], dtype=torch.float16)
idx = torch.tensor([[3, 1], [4, 1]], dtype=torch.float16)

a = arr + idx
d = torch.div(arr, idx)
print(a)
