from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。松山いいところよ。")

wakati_list = []
# Janameの最新バージョンでは、tokenizeはジェネレータオブジェクトを返す
for token in tokens:
    #print(token.part_of_speech.split(',')[0])
    if token.part_of_speech.split(',')[0] =="名詞":
        word_n = token.surface
        #print(word_n)
        wakati_list.append(word_n)

print(" , ".join(wakati_list))