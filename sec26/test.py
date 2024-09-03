import re

test_str = "<html><h1>ここ大見出し</h1><p>ここに文章</p></html>"
pattern = r'<h.*>'
result = re.match(pattern, test_str)

if result:
    print(result)
    print("マッチしました")
else:
    print("マッチしてません")