from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

raw_text = [
    '夕食にスパゲッティを食べました。',
    '昨夜の夕食はパスタでした。',
    '昨日は公園の周りをジョギングしていました。'
]
def wakati(jp_str):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(jp_str,wakati=True)
    return " ".join(list(tokens))

def get_vector(text_list):
    wakati_text = list(map(wakati,text_list))
    # print(wakati_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(wakati_text)
    #print(vectorizer.get_feature_names_out())
    return X.toarray()

v_text= get_vector(raw_text)
print(v_text)

def cos_sim(v1,v2):
    return v1@v2 /( np.sqrt(np.sum(np.abs(v1**2)))*np.sqrt(np.sum(np.abs(v2**2))))

print(cos_sim(v_text[0],v_text[1]))
print(cos_sim(v_text[0],v_text[2]))