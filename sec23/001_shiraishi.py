"""
①任意の文章を分かち書きにし、元の文章に含まれる「名詞」だけを抜き出し、順に分かち書きして並べるPythonコードをかけ。関数化してください。def nou_waka(txt)
"""

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# ファイル読み込み & クリーニング
# def get_txt(file_name):
#     with open(file_name,'r', encoding='UTF-8') as f:
#         txt = f.read()
#         txt = txt.replace("\n", "")
#         txt = re.sub(r'《.+?》','',txt)#青空文庫のルビ対策
#     #txt_array = txt.split('。') 
#     return txt

def nou_waka(txt: str)-> str:
    tokenizer = Tokenizer(wakati=False)
    tokens = tokenizer.tokenize(txt)
    wakati_list = []
    for token in tokens:
        if token.part_of_speech.split(",")[0] == "名詞":
            wakati_list.append(token.surface)
    return ",".join(wakati_list)


def Q1(txt: str)-> str:
    return nou_waka(txt)

def getCorpus(txt: str)-> np.asarray:
    split_list = txt.split("。")
    corpus_list = []
    for i, split_txt in enumerate(split_list):
        i = i + 1
        if i  == len(split_list):
            break
        # print(f"split{i}:  {token_txt}")
        token_txt = nou_waka(split_txt)
        corpus_list.append(token_txt)
    return corpus_list

def Q2(txt : str)-> np.array:
    corpus = getCorpus(txt)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names_out())
    return X.toarray()


"""
②　①のコードを利用して、「。」で区切ってこの文章を6つの文にし、6つの要素をもつリストにして、そのリストをTF-IDFでベクトル化するような関数をつくれ。
与える文章は、引数としてどんな文章もその文章に応じた行数にするようにしなさい。
"""


def main():
    txt = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。"
    q1_ans = Q1(txt)
    print(f"Q1の答え: {q1_ans}")
    
    q2_ans = Q2(txt)
    print(f"Q2の答え: {q2_ans}")
if __name__ == "__main__":
    main()