import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
import mnist


def test(dataloader, model):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    num_classes = 10
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    worst = {c: (-1.0, None, None) for c in range(num_classes)}
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)           # (1,10)
            probs = outputs.cpu().numpy().flatten()
            true = labels.item()
            pred = int(np.argmax(probs))

            conf_mat[true, pred] += 1
            if pred == true:
                correct += 1
            else:
                score = probs[pred]
                if score > worst[true][0]:
                    worst[true] = (score, pred, idx)
            total += 1

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}\n")

    print("Confusion Matrix (rows=true, cols=predicted):")
    print(conf_mat)
    print()

    print("Most confusing examples per true class:")
    for c in range(num_classes):
        score, pred_label, idx = worst[c]
        if pred_label is None:
            print(f"Class {c}: no misclassifications")
        else:
            print(f"Class {c}: sample idx {idx} misclassified as {pred_label} (confidence={score:.4f})")

    return accuracy


def main():
    transform = T.Pad(2, fill=0, padding_mode='constant')
    test_ds = mnist.MNIST(split="test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = torch.load("LeNet1.pth")
    test(test_loader, model)


if __name__ == "__main__":
    main()
