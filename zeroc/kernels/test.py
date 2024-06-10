import torch
import zc_bmm_half

# import zc_softmax

a = torch.ones([3, 2, 6], dtype=torch.float16, device="cuda:0")
b = torch.ones([3, 6, 4], dtype=torch.float16, device="cuda:0")
print(zc_bmm_half.call(a, b))
