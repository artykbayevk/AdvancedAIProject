import os
import re
import gensim
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

STOPWORDS = stopwords.words("english")
STEMMER = SnowballStemmer("english")

DATA_DIR_PATH = 'data'
DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, 'training.1600000.processed.noemoticon.csv')
DATA = pd.read_csv(DATA_FILE_PATH, encoding = "ISO-8859-1", names = ["target", "ids", "data", 'flag', "user", "text"])

target_encoding = {0: "neg", 2: 'neu', 4: 'pos'}

x_raw = DATA.text
y_raw = DATA.target

x_raw.head()
y_raw.head()


def cleaning_sentence(text):
    text = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
    return text


def removing_stop_words(text):
    words = text.split()
    
    res = []
    for word in words:
        if not word in STOPWORDS:
            res.append(word)
    return ' '.join(res)


def stemming_words(text):
    words = text.split()
    res = []
    for word in words:
        res.append(STEMMER.stem(word))
    return ' '.join(res) 


def text_pre_process(text):
    cleaned = cleaning_sentence(text)
    removed = removing_stop_words(cleaned)
    stemmed = stemming_words(removed)
    return stemmed


def clean(x, y):
    x_clean = x.apply(lambda item: text_pre_process(item))
    return x_clean, y


def main():
    X, Y = clean(x_raw, y_raw)
    print(X)


if __name__ == '__main__':
    main()
