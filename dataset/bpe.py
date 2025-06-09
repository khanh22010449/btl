from transformers import AutoTokenizer
from collections import defaultdict
from tqdm import tqdm, trange
import heapq


def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                new_split.append(a + b)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        splits[word] = new_split
    return splits


def build_bpe_vocab(train_loader, vocab_size=20000):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    word_freqs = defaultdict(int)

    # 1. Thu thập tần suất từ toàn bộ tập huấn luyện
    for batch in tqdm(train_loader, desc="Building frequency dict"):
        for text in batch["text"]:
            words_with_offsets = (
                tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            )
            new_words = [word for word, _ in words_with_offsets]
            for word in new_words:
                word_freqs[word] += 1

    # 2. Khởi tạo bảng chữ cái ban đầu
    alphabet = sorted(set("".join(word_freqs.keys())))
    vocab = ["<|endoftext|>"] + alphabet.copy()
    splits = {word: list(word) for word in word_freqs.keys()}
    merges = {}

    # 3. Lặp để xây dựng từ vựng BPE
    for _ in trange(vocab_size - len(vocab), desc="Merging pairs"):
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=pair_freqs.get)
        merged_token = best_pair[0] + best_pair[1]

        splits = merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
        merges[best_pair] = merged_token
        vocab.append(merged_token)

    return vocab, merges, tokenizer


def tokenize_corpus(train_loader, vocab, merges, tokenizer):
    tokenized_corpus = []

    for batch in tqdm(train_loader, desc="Tokenizing corpus"):
        for text in batch["text"]:
            pre_tokenize_result = (
                tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            )
            pre_tokenized_text = [word for word, _ in pre_tokenize_result]
            splits = [[char for char in word] for word in pre_tokenized_text]

            # Áp dụng các merge theo thứ tự
            for pair, merge in merges.items():
                for idx, split in enumerate(splits):
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1
                    splits[idx] = split

            tokenized_sentence = sum(splits, [])  # Flatten
            tokenized_corpus.append(tokenized_sentence)

    return tokenized_corpus
