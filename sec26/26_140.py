import re

test_str ="089-123-456"
pattern = r'08\d*\-'
result = re.match(pattern,test_str)
if result:
    print(result)
    print("マッチしました")
else:
    print("マッチしてません")
