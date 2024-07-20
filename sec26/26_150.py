import re

test_str ="<html><h1>ここ大見出し</h1><p>ここに文章</p><p>ここに文章2</p></html>"
pattern = r'<p>(.+?)</p>'
result = re.sub(pattern,r'\1',test_str)
if result:
    print(result)
    print("マッチしました")
else:
    print("マッチしてません")
#print(type(result))