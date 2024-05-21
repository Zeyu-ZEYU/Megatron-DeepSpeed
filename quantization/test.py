# import torch

# table = torch.tensor([10, 39, 18, 23], dtype=torch.float, device="cuda:0")

# compressed = torch.tensor([[0, 1, 2, 1, 1, 1], [3, 3, 3, 3, 3, 1]], dtype=torch.uint8, device=table.device)

# a = table[compressed]

# print(a)

import torch

# Define the mapping
mapping_tensor = torch.rand(16, dtype=torch.float16)  # Random float16 values

# Move the tensor to GPU
mapping_tensor = mapping_tensor.cuda()

# Lookup example using torch.uint8 for indices
indices = torch.tensor([0, 5, 14, 2], dtype=torch.int)  # Use uint8 for indices
lookup_values = mapping_tensor[indices.cuda()]  # Perform lookup

print("Lookup Results:", lookup_values)
