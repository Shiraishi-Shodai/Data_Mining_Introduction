from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。",wakati=True)

for token in tokens:
  print(token)

"""
昔
々
、
愛媛
県
松山
市
に
優しい
おじいさん
と
おばあさん
が
住ん
で
い
まし
た
。
"""