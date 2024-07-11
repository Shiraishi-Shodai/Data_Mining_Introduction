from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。松山いいところよ。")
x =[]
dic={} #単語とIDのペア
count = 0 # ID
result = []
for word in tokens:
  #word = word.strip()  #wakati=Trueのとき
  word = word.surface  
  if word == "": continue
  if not word in dic: # まだ辞書になければdicに追加
    dic[word] = count
    num = count
    count +=1
  else: #辞書にあれば、その番号を見つける。
    num=dic[word]
  print(num,word)
  result.append(num)
print(result)
