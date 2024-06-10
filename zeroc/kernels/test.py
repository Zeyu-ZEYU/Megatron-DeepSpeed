import torch
import zc_bmm_uint8
import zc_softmax

a = torch.ones([3, 2, 6], dtype=torch.float16, device="cuda:0")
b = torch.ones([3, 6, 4], dtype=torch.float16, device="cuda:0")
print(torch.bmm(a, b))
