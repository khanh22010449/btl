from datasets import load_dataset
import re
import string
import contractions
import unicodedata
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_data(path):
    # Load Dataset
    dataset = load_dataset(path)

    # Clean and Normalize data
    dataset = dataset.map(
        normalize_text,
        batched=True,
        load_from_cache_file=False,
    )

    # Split train/val set
    train_val_split = dataset["train"].train_test_split(0.2, seed=42)

    test_loader = dataset["test"]

    # Chuyển thành DataLoader
    train_loader = DataLoader(train_val_split["train"], batch_size=1024)
    val_loader = DataLoader(train_val_split["test"], batch_size=1024)
    test_loader = DataLoader(test_loader, batch_size=1024)

    return train_loader, val_loader, test_loader


"""====>> Normalize Data <<===="""


def normalize_for_bpe(sentence):
    if not isinstance(sentence, str):
        raise TypeError("Expected a string input for normalization.")

    # Chuẩn hóa văn bản thành dạng NFC (Normalization Form C)
    text = unicodedata.normalize("NFC", sentence)

    # Loại bỏ thẻ HTML
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http[s]?://[^\s)]+", "", text)

    # Giãn từ viết tắt
    text = contractions.fix(text)

    # Loại bỏ tất cả các ký tự không phải ASCII, giữ lại chữ cái, số và dấu câu cơ bản
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    text = "".join(c for c in text if c in allowed_chars)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text(batch):
    batch["text"] = [normalize_for_bpe(text) for text in batch["text"]]
    return batch


if __name__ == "__main__":
    train, val, test = load_data("stanfordnlp/imdb")
    plt.figure(figsize=(10, 5))
    plt.bar(
        ["train", "val", "test"],
        [len(train.dataset), len(val.dataset), len(test.dataset)],
        color=["blue", "orange", "green"],
    )
    plt.title("Dataset Sizes")
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Samples")
    plt.show()
