from pathlib import Path

from torchvision import datasets


DATA_DIR = Path("visiontransformer_data")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    print(f"MNIST downloaded to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
