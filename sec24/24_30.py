import re
from janome.tokenizer import Tokenizer


# wordsを配列ではなく辞書として定義。各トークンの出現頻度をカウント
def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = {}
    for line in token:
        tkn = re.split('\t|,', str(line))
        # tkn[0]が存在し、tkn[1]が名詞でtkn[2]が一般または固有名詞であるか
        if tkn[0] and tkn[1] in ['名詞'] and tkn[2] in ['一般', '固有名詞'] :
            # 既知のトークンであれば+1をし、始めてであれば1を代入する
            words[tkn[0]] = words[tkn[0]] +1 if tkn[0] in words else 1
        # 非破壊でオブジェクトを昇順にソート
        # items() は辞書の各要素を (キー, 値) のタプルとして返す。
    words = sorted(words.items(), key=lambda x:x[1])
    return words

text = open("soseki.txt", encoding="utf8").read()
text_list = wakachigaki(text)
print(text_list)