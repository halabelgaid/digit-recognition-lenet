from torchvision.datasets import MNIST
import os

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

os.makedirs(DATA_DIR, exist_ok=True)

train_ds = MNIST(root=DATA_DIR, train=True, download=False)
test_ds  = MNIST(root=DATA_DIR, train=False, download=False)

with open(os.path.join(DATA_DIR, "train_label.txt"), "w") as f:
    for _, label in train_ds:
        f.write(f"{label}\n")
print(f"Wrote {len(train_ds)} labels to {DATA_DIR}/train_label.txt")

with open(os.path.join(DATA_DIR, "test_label.txt"), "w") as f:
    for _, label in test_ds:
        f.write(f"{label}\n")
print(f"Wrote {len(test_ds)} labels to {DATA_DIR}/test_label.txt")
