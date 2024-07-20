import re

test_str ="gooooooooogle"
pattern = r'go+gle'
pattern = r'go{2,6}gle'
result = re.match(pattern,test_str)
if result:
    print(result)
    print("マッチしました")
else:
    print("マッチしてません")
