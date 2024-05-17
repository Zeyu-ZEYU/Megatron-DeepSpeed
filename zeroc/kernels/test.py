import torch
import zc_softmax

input = torch.tensor([[1, 2, 3], [2, 3, 9]], dtype=torch.float, device="cuda:0")
output = zc_softmax.call(input)
print(output)
