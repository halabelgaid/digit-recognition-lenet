import os
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

print("Preparing training images...")
train_ds = MNIST(root=DATA_DIR, train=True, download=True)
for idx, (img, label) in enumerate(train_ds):
    img.save(os.path.join(TRAIN_DIR, f"{idx}.png"))
print(f"Exported {len(train_ds)} training images to {TRAIN_DIR}")

print("Preparing test images...")
test_ds = MNIST(root=DATA_DIR, train=False, download=True)
for idx, (img, label) in enumerate(test_ds):
    img.save(os.path.join(TEST_DIR, f"{idx}.png"))
print(f"Exported {len(test_ds)} test images to {TEST_DIR}")
