from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。松山いいところよ。",wakati=True)

# Janameの最新バージョンでは、tokenizeはジェネレータオブジェクトを返す
print(next(tokens))
print(next(tokens))
print(next(tokens))
print(next(tokens))