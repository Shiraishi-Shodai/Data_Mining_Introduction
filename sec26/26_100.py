import re

test_str ="<html><h1>こんにちは</h1><h1>ここ大見出し</h1><p>ここに文章</p></html>"
pattern = r'><h.*>'
pattern = r"<h1>(.*?)</h1>" # ()は()の中を配列に入れるという意味。groupで取り出せる。?をつけると最初に見つけたマッチングを取得
result = re.search(pattern, test_str)
if result:
    print(result)
    # print(result.group())
    print("マッチしました")
else:
    print("マッチしてません")
