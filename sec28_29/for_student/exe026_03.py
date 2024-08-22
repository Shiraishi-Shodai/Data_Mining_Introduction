import re

def is_Good(test_str):

    pattern = r'^Go+d$'
    result = re.match(pattern,test_str)
    if result:
        #print(result)
        print("マッチしました",end=" ")
        return True
    else:
        print("マッチしてません",end=" ")
        return False


samples =[
    "Goooooooood",
    "goooooooood",
    "god",
    "Gd",
    "God",
    "GoodMoning",
    "Goooooooogle",
]

for s in samples:
    print(f"{s} => {is_Good(s)}")
