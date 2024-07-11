from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。",wakati=True)

#for token in tokens:
#  print(token)

wakati_list = []
# Janameの最新バージョンでは、tokenizeはジェネレータオブジェクトを返す
for token in tokens:
    wakati_list.append(token)
    #print(token)

print(" ".join(wakati_list))