from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("昔々、愛媛県松山市に優しいおじいさんとおばあさんが住んでいました。")

# tokensは配列ではなくジェネレーター
for token in tokens:
  print(token)
  
"""
昔      名詞,副詞可能,*,*,*,*,昔,ムカシ,ムカシ
々      記号,一般,*,*,*,*,々,々,々
、      記号,読点,*,*,*,*,、,、,、
愛媛    名詞,固有名詞,地域,一般,*,*,愛媛,エヒメ,エヒメ
県      名詞,接尾,地域,*,*,*,県,ケン,ケン
松山    名詞,固有名詞,地域,一般,*,*,松山,マツヤマ,マツヤマ
市      名詞,接尾,地域,*,*,*,市,シ,シ
に      助詞,格助詞,一般,*,*,*,に,ニ,ニ
優しい  形容詞,自立,*,*,形容詞・イ段,基本形,優しい,ヤサシイ,ヤサシイ
おじいさん      名詞,一般,*,*,*,*,おじいさん,オジイサン,オジーサン
と      助詞,並立助詞,*,*,*,*,と,ト,ト
おばあさん      名詞,一般,*,*,*,*,おばあさん,オバアサン,オバーサン
が      助詞,格助詞,一般,*,*,*,が,ガ,ガ
住ん    動詞,自立,*,*,五段・マ行,連用タ接続,住む,スン,スン
で      助詞,接続助詞,*,*,*,*,で,デ,デ
い      動詞,非自立,*,*,一段,連用形,いる,イ,イ
まし    助動詞,*,*,*,特殊・マス,連用形,ます,マシ,マシ
た      助動詞,*,*,*,特殊・タ,基本形,た,タ,タ
。      記号,句点,*,*,*,*,。,。,。
"""