import re
import unicodedata
 
def text_preprocessing(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if __name__ == "__main__":
    print(text_preprocessing("Hello  !!  "))