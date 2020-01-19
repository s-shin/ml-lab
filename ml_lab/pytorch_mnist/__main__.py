import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from optparse import OptionParser
from sklearn.metrics import accuracy_score


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        # (1, 28, 28) => (64, 24, 24) => (64, 12, 12)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # => (128, 8, 8)
        x = F.dropout(F.relu(self.conv2(x)), 0.2)
        # => (128 * 8 * 8)
        x = x.view(x.size(0), -1)
        # => (10)
        x = F.relu(self.fc1(x))
        return x


def fit(model, optimizer, criterion, epochs=20):
    training_dataset = torchvision.datasets.MNIST(
        '~/.pytorch/mnist', train=True, download=True,
        transform=torchvision.transforms.ToTensor())
    # training_dataloader = [(x, t), ...]
    # x.shape = [batch_size, 1, 28, 28]
    # t.shape = [batch]
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=4, shuffle=True, num_workers=1)

    model.train()
    for epoch in range(epochs):
        print('=== Epoch#{} ==='.format(epoch))
        for i, (input, target) in enumerate(training_dataloader):
            print(input, target)
            exit()
            if i % 10 == 0:
                end_str = '\n' if i > 0 and i % 1000 == 0 else ''
                print('{} '.format(i), end=end_str, flush=True)
            model.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print()


def test(model, criterion):
    test_dataset = torchvision.datasets.MNIST(
        '~/.pytorch/mnist', train=False, download=True,
        transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=True, num_workers=1)

    model.eval()
    labels = []
    preds = []
    losses = []
    for input, target in test_dataloader:
        labels.extend(target)
        output = model(input)
        loss = criterion(output, target)
        losses.append(loss.cpu().data)
        pred = output.argmax(1)
        preds.extend(pred)

    print('Loss: {:.3f}, Accuracy: {:.3f}'.format(
        np.mean(losses), accuracy_score(labels, preds)))


def main():
    parser = OptionParser(usage='Usage: %prog [options] <action>')
    parser.add_option('-m', '--model', default='pytorch-mnist.pt')
    parser.add_option('--epochs', type='int', default=20)
    G = globals()
    (options, args) = parser.parse_args(G['ARGS'] if 'ARGS' in G else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    action = args[0]
    if action == 'fit':
        fit(model, optimizer, criterion, epochs=options.epochs)
        torch.save(model.state_dict(), options.model)
        test(model, criterion)

    if action == 'test':
        model.load_state_dict(torch.load(options.model))
        test(model, criterion)


main()
