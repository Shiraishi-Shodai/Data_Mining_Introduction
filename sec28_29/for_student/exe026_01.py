import re

def conv3_4_with_regex(number):
    # 入力が7桁の数字でない場合はエラーメッセージを返す#北海道には、064-0941がある。
    if not isinstance(number, int) or not (number <= 9999999):
    #if not isinstance(number, int) or not (1000000 <= number <= 9999999):
        return "Error!"

    # 数字を3桁と4桁に分け、-で結んで文字列に変換
    number_str = str(number)
    match = re.match(r'^(\d{3})(\d{4})$', number_str)

    # 正規表現に一致する場合は変換して返し、一致しない場合はエラーメッセージを返す
    if match:
        result = '-'.join(match.groups())
        return result
    else:
        return "Error!"

test_numbers =[1670031,7640003,2235648,12345678,"abc123"]
for n in test_numbers:
    print(f"{n} => {conv3_4_with_regex(n)}")
