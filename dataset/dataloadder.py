from datasets import load_dataset
import re
import string
import contractions
from torch.utils.data import DataLoader


def load_data(path):
    # Tải dataset
    dataset = load_dataset(path)
    dataset = dataset.map(normalize_text, batched=True)

    # Split train/val set
    train_val_split = dataset["train"].train_test_split(0.2, seed=42)

    # Áp dụng normalization
    # train_val_split = train_val_split.map(normalize_text, batched=True)

    test_loader = dataset["test"]

    # Chuyển thành DataLoader
    train_loader = DataLoader(train_val_split["train"], batch_size=1024, num_workers=4)
    val_loader = DataLoader(train_val_split["test"], batch_size=1024, num_workers=4)
    test_loader = DataLoader(test_loader, batch_size=1024, num_workers=4)
    # for batch in train_loader:
    #     print(batch["new_texts"][0])
    #     break

    return train_loader, val_loader, test_loader


"""====>> Normalize Data <<===="""


def normalize_for_bpe(sentence):
    if not isinstance(sentence, str):
        raise TypeError("Expected a string input for normalization.")
    # Giãn từ viết tắt
    text = contractions.fix(sentence)
    # Loại bỏ thẻ HTML
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http[s]?://[^\s)]+", "", text)
    # Chuẩn hóa khoảng trắng sớm
    text = re.sub(r"\s+", " ", text).strip()
    # Thêm khoảng trắng quanh dấu câu
    text = re.sub(r"[.,?!;:()/-]", lambda match: f" {match.group(0)} ", text)
    # Chuẩn hóa khoảng trắng cuối
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(batch):
    new_texts = []
    for text in batch["text"]:
        new_text = normalize_for_bpe(text)
        new_texts.append(new_text)
    batch["new_texts"] = new_texts
    return batch


if __name__ == "__main__":
    train, val, test = load_data("stanfordnlp/imdb")
