
import re
def get_19_24(t_time:str)->str:
        res = re.match(r'(\d{2}):\d{2}',t_time)
        if res:
            #print(res.groups(1))
            hh = (res.groups(1))[0]
            hh = int(hh)
            if hh >= 19:
                 return str(hh)
            else:
                 return ""
        else:
            print("正しい形式ではありません")
            return False
def get_19_24(time_str: str) -> str:
    # 正規表現パターンの定義
    pattern = r'^(1[9-9]|2[0-3]):[0-5][0-9]$'
    # 正規表現でマッチをチェック
    match = re.match(pattern, time_str)

    if match:
        # マッチした場合、時間の部分を取り出して返す
        return match.group(1)
    else:
        # マッチしない場合は空文字を返す
        return ""
samples =[
    "00:05",
    "05:34",
    "12:71",
    "19:11",
    "23:34",
]
for s in samples:
    print(f"{s} => {get_19_24(s)}")
