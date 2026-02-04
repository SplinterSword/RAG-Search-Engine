from nltk.stem import PorterStemmer

def text_preprocessing(text: str) -> list[str]:
    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()

    text = text.translate(str.maketrans("", "", "!@#$%^&*()_+[]{}|;:,.<>/?`~"))
    text = text.lower()
    tokens = text.split()

    tokens = [token for token in tokens if token not in stopwords]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens