import torch

device = torch.device("cuda")

a = torch.rand((5000, 5000), device=device)
b = torch.rand((5000, 5000), device=device)

c = torch.matmul(a, b)

print("Calculation complete.")
print("Result shape:", c.shape)
print("Device:", c.device)