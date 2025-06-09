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


# # """

# #  Optimize Code for BPE Tokenization
# from transformers import AutoTokenizer
# from collections import defaultdict
# from tqdm import tqdm, trange
# import heapq


# # ---------------------
# # 1. Tính tần suất cặp ký tự với heap
# # ---------------------
# def compute_pair_freqs(splits, word_freqs):
#     """
#     Tính tần suất tất cả cặp ký tự và khởi tạo heap.
#     Trả về: pair_freqs dict và một max-heap [(-count, pair), ...]
#     """
#     pair_freqs = defaultdict(int)
#     heap = []
#     for word, freq in word_freqs.items():
#         symbols = splits[word]
#         for i in range(len(symbols) - 1):
#             pair = (symbols[i], symbols[i + 1])
#             pair_freqs[pair] += freq
#     for pair, cnt in pair_freqs.items():
#         # heap lưu âm count để pop max nhanh
#         heapq.heappush(heap, (-cnt, pair))
#     return pair_freqs, heap


# # ---------------------
# # 2. Merge một cặp và cập nhật cục bộ
# # ---------------------
# def merge_pair(a, b, splits, word_freqs, pair_freqs, heap):
#     """
#     Hợp nhất cặp (a, b) thành token mới trong splits.
#     Cập nhật pair_freqs và heap chỉ với các từ chứa cặp đó.
#     Trả về: merged_token
#     """
#     merged = a + b
#     # Lấy danh sách từ chứa cặp này
#     affected_words = [
#         w
#         for w, sym in splits.items()
#         if any(sym[i] == a and sym[i + 1] == b for i in range(len(sym) - 1))
#     ]
#     for word in affected_words:
#         freq = word_freqs[word]
#         seq = splits[word]
#         i = 0
#         while i < len(seq) - 1:
#             if seq[i] == a and seq[i + 1] == b:
#                 left = seq[i - 1] if i > 0 else None
#                 right = seq[i + 2] if i + 2 < len(seq) else None
#                 # giảm tần suất cặp cũ
#                 pair_freqs[(a, b)] -= freq
#                 # giảm cặp kề
#                 if left:
#                     pair_freqs[(left, a)] -= freq
#                 if right:
#                     pair_freqs[(b, right)] -= freq
#                 # thực hiện merge
#                 seq[i : i + 2] = [merged]
#                 # tăng cặp mới
#                 if left:
#                     pair_freqs[(left, merged)] += freq
#                 if right:
#                     pair_freqs[(merged, right)] += freq
#             else:
#                 i += 1
#         splits[word] = seq
#     # cập nhật heap với giá trị mới (lazy deletion)
#     for pair, cnt in list(pair_freqs.items()):
#         heapq.heappush(heap, (-cnt, pair))
#     return merged


# # ---------------------
# # 3. Xây dựng vocab BPE
# # ---------------------
# def build_bpe_vocab(train_loader, vocab_size=20000):
#     # load tokenizer và tần suất từ
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     word_freqs = defaultdict(int)
#     for batch in tqdm(train_loader, desc="Building frequency dict"):
#         for text in batch["text"]:
#             for word, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
#                 text
#             ):
#                 word_freqs[word] += 1

#     # khởi tạo splits và vocab cơ bản
#     splits = {w: list(w) for w in word_freqs}
#     alphabet = sorted(set("".join(word_freqs)))
#     vocab = ["<|endoftext|>"] + alphabet.copy()

#     # tính lần đầu pair_freqs + heap
#     pair_freqs, heap = compute_pair_freqs(splits, word_freqs)
#     merges = {}
#     n_merges = vocab_size - len(vocab)
#     for _ in trange(n_merges, desc="Merging pairs"):
#         # pop cặp có tần suất lớn nhất hợp lệ
#         while heap:
#             neg_cnt, pair = heapq.heappop(heap)
#             if -neg_cnt == pair_freqs.get(pair, 0) and pair_freqs[pair] > 0:
#                 break
#         else:
#             break
#         a, b = pair
#         merged = merge_pair(a, b, splits, word_freqs, pair_freqs, heap)
#         merges[pair] = merged
#         vocab.append(merged)

#     return vocab, merges, tokenizer


# # ---------------------
# # 4. Tokenize corpus nhanh
# # ---------------------
# def tokenize_corpus(train_loader, merges, tokenizer):
#     tokenized_corpus = []
#     merge_items = list(merges.items())
#     for batch in tqdm(train_loader, desc="Tokenizing corpus"):
#         for text in batch["text"]:
#             words = [
#                 w
#                 for w, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
#                     text
#                 )
#             ]
#             splits = [[c for c in w] for w in words]
#             for (a, b), merged in merge_items:
#                 for s in splits:
#                     i = 0
#                     while i < len(s) - 1:
#                         if s[i] == a and s[i + 1] == b:
#                             s[i : i + 2] = [merged]
#                         else:
#                             i += 1
#             tokenized_corpus.append(sum(splits, []))
#     return tokenized_corpus


# # """
