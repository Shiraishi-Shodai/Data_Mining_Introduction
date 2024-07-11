from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。",wakati=False)

wakati_list = []
# Janameの最新バージョンでは、tokenizeはジェネレータオブジェクトを返す
for token in tokens:
    wakati_list.append(token.part_of_speech)
    if token.part_of_speech.split(',')[0] =='名詞':
        print(token.surface)

print(wakati_list)