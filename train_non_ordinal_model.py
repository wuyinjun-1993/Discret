import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class NonOrdinalDataset(Dataset):
    def __init__(self, is_train = True, shuffle_seed = 0):
        self.data = np.load(open('non_ordinal_dataset.npy', 'rb')).astype(np.float32)
        np.random.seed(shuffle_seed)
        np.random.shuffle(self.data)
        self.features = self.data[0 if is_train else (2*len(self.data)//3): (2*len(self.data)//3) if is_train else len(self.data), :-1]
        self.labels = self.data[0 if is_train else (2*len(self.data)//3): (2*len(self.data)//3) if is_train else len(self.data),-1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.asarray(self.features[idx])).to(device=device), torch.from_numpy(np.asarray([self.labels[idx]])).to(device=device)
    
    def get_num_features(self):
        return self.features.shape[1]

class NonOrdinalEHRNetwork(nn.Module):
    def __init__(self, num_features):
        super(NonOrdinalEHRNetwork, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.linear_stack(x)


LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCHS = 1

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)*BATCH_SIZE
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_rounded = pred.detach().numpy().round()
            correct += (pred_rounded == y.detach().numpy()).sum()

    test_loss /= num_batches
    correct /= (size*BATCH_SIZE)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    train_data = NonOrdinalDataset()
    test_data = NonOrdinalDataset(is_train=False)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    loss_fn = nn.BCELoss()
    model = NonOrdinalEHRNetwork(num_features=train_data.get_num_features()).to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


