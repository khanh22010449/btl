from dataloadder import load_data
from bpe import build_bpe_vocab, tokenize_corpus
from gensim.models import Word2Vec
import numpy as np


def skip_gram(token_corpus, window_size):
    # Train the Word2Vec model
    model = Word2Vec(sentences=token_corpus, vector_size=100, window=window_size, min_count=1, sg=1)  # Skip-gram model ==> sg = 1 ; cbow model ==> sg = 0
    
    # Create a dictionary of word embeddings
    word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
    
    return word_embeddings

def sentence_embedding(sentence, word_embeddings):
    # Convert a sentence into its embedding by averaging the embeddings of its words
    words = sentence.split()
    embeddings = [word_embeddings[word] for word in words if word in word_embeddings]
    
    if not embeddings:
        return None  # Return None if no words are found in the embeddings
    
    return sum(embeddings) / len(embeddings)  # Average the embeddings

if __name__ == "__main__":
    train, val, test = load_data("stanfordnlp/imdb")
    vocab, merges, tokenizer = build_bpe_vocab(train_loader=train, vocab_size=10000)
    print(vocab)
    print(len(vocab))
    token_corpus = tokenize_corpus(train, merges, tokenizer)
    word_embeddings = skip_gram(token_corpus, window_size=5)
    X = np.array([sentence_embedding(sentence, word_embeddings) for sentence in token_corpus])
    y = np.array([batch["label"] for batch in train])
    print(X.shape, y.shape)
    print(X[:5], y[:5])
