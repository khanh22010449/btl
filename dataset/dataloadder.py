from datasets import load_dataset
import re
import string
import contractions
from torch.utils.data import DataLoader


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

    # text = sentence.lower()

    # Loại bỏ thẻ HTML
    text = re.sub(r"<.*?>", " ", sentence)
    text = re.sub(r"http[s]?://[^\s)]+", "", text)
    # Chuẩn hóa khoảng trắng sớm
    text = re.sub(r"\s+", " ", text).strip()
    # Giãn từ viết tắt
    text = contractions.fix(text)
    # Thêm khoảng trắng quanh dấu câu
    # text = re.sub(r"[.,?!;:()/-]", lambda match: f" {match.group(0)} ", text)
    # text = re.sub(r"([.,?!;:()\"'/\[\]{}\-])", r" \1 ", text)

    # Chuẩn hóa khoảng trắng cuối
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(batch):
    batch["text"] = [normalize_for_bpe(text) for text in batch["text"]]
    return batch


if __name__ == "__main__":
    train, val, test = load_data("stanfordnlp/imdb")
    for batch in train:
        for i in range(10):
            print(batch["text"][i])
        break
