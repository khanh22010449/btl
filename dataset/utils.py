from dataloadder import load_data
from bpe import build_bpe_vocab, tokenize_corpus

if __name__ == "__main__":
    train, val, test = load_data("stanfordnlp/imdb")
    vocab, merges, tokenizer = build_bpe_vocab(train_loader=train, vocab_size=10000)
    print(vocab)
    print(len(vocab))
    token_corpus = tokenize_corpus(train, vocab, merges, tokenizer)
    # tokenize("I'm Love --You to , School")
