from dataloadder import load_data
import contractions

if __name__ == "__main__":
    train, val, test = load_data("stanfordnlp/imdb")
    for batch in train:
        print(batch["new_texts"])
        break
