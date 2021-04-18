import torch

a = torch.tensor([2., 3.], requires_grad=False)
a.requires_grad_()
print(a)
b = torch.tensor([5., 6.])
c = torch.tensor([7., 8.])
d = torch.tensor([13., 15.])
d.requires_grad_()

loss = a + b + c

print(loss)
loss.backward(torch.ones_like(loss))

print(a.grad)

