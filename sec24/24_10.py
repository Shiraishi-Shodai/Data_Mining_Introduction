import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        # line (<class 'janome.tokenizer.Token'>) : 例) 平塚    名詞,固有名詞,人名,姓,*,*,平塚,ヒラツカ,ヒラツカ
        # タブ文字または感まで分割し配列に変換
        tkn = re.split('\t|,', str(line))
        words.append(tkn[0])
    return ' ' . join(words)

text = open("soseki.txt", encoding="utf8").read()
text = wakachigaki(text)
print(text)

