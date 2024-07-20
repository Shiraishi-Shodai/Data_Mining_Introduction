import re
test_str = "<html><body>AAAAAA</body><\html>"
pattern = r"<html>(.*)</html>"

result = re.sub(pattern, "c", test_str)

if result:
    print(result)
else:
    print("no search")