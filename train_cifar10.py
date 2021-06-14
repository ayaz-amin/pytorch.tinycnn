import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor

from stnvae import STNVAE


def logistic_ll(mean, logscale, sample, bin_size=1/256.):
    scale = torch.exp(logscale / 2)
    sample = (torch.floor(sample / bin_size) * bin_size - mean) / scale
    logp = torch.log(torch.sigmoid(sample + bin_size / scale) - torch.sigmoid(sample) + 1e-7)
    return logp.sum(dim=(1, 2, 3))

BATCH_SIZE = 16

transforms = Compose([
    ToTensor(),
    lambda x: x - 0.5
    ])

cifar_train = CIFAR10(
        "data/", train=True, download=True, transform=transforms
        )

train_loader = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True)

cifar_test = CIFAR10(
        "data/", train=False, download=True, transform=transforms
        )

test_loader = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=False)


model = STNVAE(layers=[3, 16, 160])
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
for iteration in range(10):
    sum_loss = 0
    count_loss = 0
    for (x, _) in train_loader:
        optimizer.zero_grad()
        x_hat, kl_div = model(x)
        
        log_pxz = logistic_ll(x_hat, model.logscale, x)
        elbo = -(log_pxz + kl_div).mean()
        elbo.backward()
        optimizer.step()

        sum_loss += elbo.item()
        count_loss += 1

    print("Iteration: {}, ELBO: {}".format(iteration + 1, sum_loss / count_loss))
