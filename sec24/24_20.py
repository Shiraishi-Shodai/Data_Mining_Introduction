import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        tkn = re.split('\t|,', str(line))
        if tkn[0] and tkn[1] in ['名詞'] and tkn[2] in ['一般', '固有名詞'] :
            words.append(tkn[0])
    return ' ' . join(words)

text = open("soseki.txt", encoding="utf8").read()
text = wakachigaki(text)
print(text)