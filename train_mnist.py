import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image

from stnvae import STNVAE

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 16
NUM_IERS = 1

transforms = Compose([
    ToTensor(),
    lambda x: (x > 0).float()
])

mnist_train = MNIST("data/", train=True, transform=transforms, download=True)
train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)

mnist_test = MNIST("data/", train=False, transform=transforms, download=True)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

model = STNVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        recon_batch, kldiv = model(data)
        recon_batch = torch.sigmoid(recon_batch)
        log_pxz = F.binary_cross_entropy(recon_batch, data)
        loss = log_pxz + kldiv
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = data.to(memory_format=torch.channels_last)
            recon_batch, kldiv = model(data)
            recon_batch = torch.sigmoid(recon_batch)
            log_pxz = F.binary_cross_entropy(recon_batch, data)
            loss = log_pxz + kldiv
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    sample_z = torch.randn(64, 32, 7, 7).to(device)
    sample_z = sample_z.to(memory_format=torch.channels_last)
    for epoch in range(1, NUM_IERS + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.sigmoid(model.decode(sample_z).cpu())
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
    torch.save(model.state_dict(), "saved_models/mnist_vae.pt")