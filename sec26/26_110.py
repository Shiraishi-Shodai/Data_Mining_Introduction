import re
org_text =""
with open("hashire_merosu.txt",encoding="utf_8") as f:
    org_text = f.read()

#print(org_text)

# (1) 繰り返し
test_str ="<html><h1>ここ大見出し</h1><p>ここに文章</p></html>"
pattern = r'<h.*>'
pattern = '<h.*?>(.*)</p>'
result = re.search(pattern, test_str)
if result:
    print(result.groups())
    print("マッチしました")
else:
    print("マッチしてません")
#print(type(result))