import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout(p=0.40)
        self.fc1 = nn.Sequential(nn.Linear(8 * 8 * 64, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = F.softmax(self.fc2(out), dim=1)
        return out


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Network()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

transformations = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

mnist_train = datasets.MNIST(
    root="./data/", train=True, transform=transforms.ToTensor(), download=True
)

mnist_test = datasets.MNIST(
    root="./data/", train=False, transform=transforms.ToTensor(), download=True
)

batch_size = 32
train_loader = DataLoader(mnist_train, batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size, shuffle=False, drop_last=True)

model.to(device)


def test():
    model.load_state_dict(torch.load("./mnist.pt"))
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        _, prediction = torch.max(preds, dim=1)
        count = sum(prediction == labels).item()
        print(count)


def train():
    n_epochs = 12
    acc_list = []
    total_step = len(train_loader)
    model.train()
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(preds.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

    if (i + 1) % 25 == 0:
        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {} / {}".format(
                epoch + 1, n_epochs, i + 1, total_step, loss.item(), correct, total
            )
        )

    torch.save(model.state_dict(), "mnist.pt")
