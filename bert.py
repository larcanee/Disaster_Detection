import torch
print(torch.arange(30, dtype=torch.long).expand_as(input_ids))