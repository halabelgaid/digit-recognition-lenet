import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from mnist import MNIST
from lenet import LeNet5
import matplotlib.pyplot as plt
from evaluation import eval_accuracy



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Pad(2, fill=0, padding_mode="constant")

    train_ds = MNIST(split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    test_ds  = MNIST(split="test", transform=transform)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = LeNet5(sigma=1.0, digit_file="DIGIT.npy").to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 20
    j = 0.1

    train_errors = []
    test_errors  = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        wrong = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).view(-1) 
            
            y = labels.item()
            o_y = outputs[y]
            all_sq = (outputs ** 2).sum()
            loss = (o_y - 1.0)**2 + j * (all_sq - o_y**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            pred = torch.argmax(outputs).item()
            if pred != y:
                wrong += 1
            total += 1

        epoch_loss = running_loss / total
        error_rate = wrong / total
        train_errors.append(error_rate)

        model.eval()
        test_acc = eval_accuracy(model, test_loader, device)
        test_err = 1 - test_acc
        test_errors.append(test_err)

        print(
            f"Epoch {epoch:2d}/{num_epochs} — "
            f"Loss: {epoch_loss:.4f} — "
            f"Error rate: {error_rate:.4f}"
        )

    epochs = range(1, num_epochs+1)
    plt.plot(epochs, train_errors, label="Train error")
    plt.plot(epochs, test_errors,  label="Test error")
    plt.xlabel("Epoch"); plt.ylabel("Error rate")
    plt.legend(); plt.show()

    torch.save(model, "LeNet1.pth")
    print("Saved trained model as LeNet1.pth")

if __name__ == "__main__":
    train()
