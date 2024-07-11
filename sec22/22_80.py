from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("吾輩はネコである。松山でのネコの暮らしは楽である。ネコは気楽だ。もちろん、吾輩は気楽な生活を楽しんでいる。")
wakati_list = []
for token in tokens:
    if token.part_of_speech.split(',')[0] =="助詞":
        continue
    elif token.part_of_speech.split(',')[0] =="記号":
        continue
    elif token.part_of_speech.split(',')[0] =="助動詞":
        continue
    else:
        word_n = token.surface
        wakati_list.append(word_n)

print(" , ".join(wakati_list))

token_dic ={}
for word in wakati_list:
    if not word in token_dic:
        token_dic[word] = 1
    else:
        token_dic[word] +=1
print(token_dic)
vec =[]
word_index =[]
for key,val in token_dic.items():
    #print(val)
    vec.append(val)
    word_index.append(key)
print(vec)
print(word_index)