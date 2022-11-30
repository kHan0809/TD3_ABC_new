import torch

a = torch.tensor([1.0,2.0,3.0,4.0,5.0])
b = torch.tensor([2.0])

print(torch.where(a-b>0.0)[0])