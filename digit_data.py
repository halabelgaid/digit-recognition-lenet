import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def make_digit_prototypes(save_path="DIGIT.npy"):
    train_ds = MNIST(root="data", train=True, download=True, transform=ToTensor())

    sums   = {d: np.zeros((28,28), dtype=np.float32) for d in range(10)}
    counts = {d: 0 for d in range(10)}

    for img_tensor, label in train_ds:
        img = img_tensor.squeeze().numpy()  
        sums[label] += img
        counts[label] += 1

    prototypes = np.zeros((10, 7, 12), dtype=np.uint8)
    for d in range(10):
        mean_img = sums[d] / counts[d]

        pil = Image.fromarray((mean_img * 255).astype(np.uint8))
        small = pil.resize((12, 7), Image.BILINEAR)

        arr = np.array(small) / 255.0 
        bitmap = (arr > 0.5).astype(np.uint8)

        prototypes[d] = bitmap

    np.save(save_path, prototypes)
    print(f"Saved {save_path} with shape {prototypes.shape}")

if __name__ == "__main__":
    make_digit_prototypes()
