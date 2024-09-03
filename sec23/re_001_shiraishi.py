from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

"""
①任意の文章を分かち書きにし、元の文章に含まれる「名詞」だけを抜き出し、順に分かち書きして並べるPythonコードをかけ。関数化してください。def nou_waka(txt)
"""
def nou_waka(txt):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(txt, wakati=False)

    wakati_list = []
    for token in tokens:
        if token.part_of_speech.split(",")[0] == "名詞":
            wakati_list.append(token.surface)
    
    return " ".join(wakati_list)

def main():
    # txt = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。"

    with open("sample.txt", "r", encoding="utf-8") as f:
        txt = f.read()
        txt = re.sub("\n", "", txt)   
        txt = txt.split("。")
        del txt[-1] #空白要素を削除
    
    corpus = [0] * len(txt)

    for i in range(len(txt)):
        corpus[i] = nou_waka(txt[i])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(X.toarray())


if __name__ == "__main__":
    main()
