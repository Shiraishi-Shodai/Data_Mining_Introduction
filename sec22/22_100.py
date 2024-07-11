from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from janome.tokenizer import Tokenizer

text_original =[
    "吾輩は猫である。",
    "名前はまだ無い。",
    "どこで生れたかとんと見当がつかぬ。",
    "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
    "吾輩はここで始めて人間というものを見た。",
    "しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。",
]

def to_wakati(stri):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(stri,wakati=True)
    w_list = [token for token in tokens]
    print(w_list)
    return " ".join(w_list)

df = pd.DataFrame({'id': ['A', 'B','C', 'D','E'],
                   'text': [
                    to_wakati(text_original[0]),
                    to_wakati(text_original[1]),
                    to_wakati(text_original[2]),
                    to_wakati(text_original[3]),
                    to_wakati(text_original[4]),
                    ],
                })

tfidf_vectorizer = TfidfVectorizer(use_idf=True,lowercase=False)# TF-IDFの計算
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])# 文章内の全単語のTfidf値を取得
terms = tfidf_vectorizer.get_feature_names_out() # index 順の単語リスト
tfidfs = tfidf_matrix.toarray()
print(terms)
print(tfidfs)