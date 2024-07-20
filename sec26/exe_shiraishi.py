import re
import datetime
"""
練習問題（1）
7 桁の数字を 3 桁と 4 桁に分けて間を - で結んだ文字列に変換する関数conv3_4 をつくれ。入力が 7 桁の数字でない場合は Error! という文字列を返せ。
例
1670031	→	167-0031
7640003　→   764-0003
123ABCD →   Error!

"""

# パターン1
def conv3_4(phone_number: str)->str:
    pattern = r"\d{7}"
    result = re.match(pattern, phone_number)
    if result:
        res = f"{phone_number[:3]}-{phone_number[3:]}"
        return res
    else:
        return "Error"

# 先生のコード
# def conv3_4(str7: str)->str:
#     if not re.match(r"\d{7,7}", str7):
#         return "Error"
    
#     pattern= r"(\d{3})(\d{4})"
#     res = re.sub(pattern, r'\1-\2',str7)
#     return res

    
"""
練習問題（2）
次のような数字 24 時間表記で書かれた時刻のデータがある。このデータのうち、19:00 から 23:59 までの時刻の場合、時間の部分の数字を文字列として取り出す関数get_19_24を書け。それ以外の時間帯なら空文字を返せ。
15:21　なら　""
19:11　なら　19
23:34　なら　23
"""

def get_19_24(time_str: str)->str:

    time_pattern = r"([1][9]|[2][0-3]):([0-5][0-9])"
    res = re.search(time_pattern, time_str)
    if res:
        return res.group()

    return ""

"""
練習問題（3）
GoodやGodのようにGで始まり、oが1個以上続いて、d終わる文字列かどうかを判定する関数 is_god を書け。

例
God 	→	True
Good　→   True
Goad　→   False
Gooooooooogle 	→	Flase
"""

# パターン1
def is_god(test_str: str)->bool:
    pattern = r"Go+d"
    return True if re.match(pattern, test_str) else False

# 先生のコード
# def is_god(test_str: str)->bool:
#     pattern = r"Go+d"
#     return True if re.match(pattern, test_str) else False


def main():
    # 練習問題1
    print(conv3_4("1234567"))
    print(conv3_4("7"))
    
    # 練習問題2
    # print(get_19_24("現時刻は19:00です。"))
    # print(get_19_24("24:60"))
    # print(get_19_24("0:00"))
    # print(get_19_24("18:59"))
    # print(get_19_24("19:00"))
    # print(get_19_24("23:59"))
    # print(get_19_24("24:00"))
    
    # 練習問題3
    # print(is_god("1Good9"))
    # print(is_god("GooAd"))
    # print(is_god("GooOd"))
    # print(is_god("Good"))
    # print(is_god("God"))

if __name__ == "__main__":
    main()