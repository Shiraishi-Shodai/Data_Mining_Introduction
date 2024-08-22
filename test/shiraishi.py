import re

def is_Gooe(test_str):
    pattern = r"^Go*e"
    result = re.match(pattern, test_str)

    if result:
        print("マッチしました", end=" ")
        return False
    else:
        print("マッチしてません", end=" ")
        return True


str_arr = ["Google", "Goooe", "Ge", "Ge is not google"]

for test_str in str_arr:
    is_Gooe(test_str)