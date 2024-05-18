"""
(1)教師あり学習を使わずに、何かしらの閾値だけで判断できないだろうか。教師なし学習だけを使って digits の中から「0」とそれ以外の数字を判別する関数 is_zero を完成させなさい。(以下のコードを完成させなさい)
ヒント：次元削減
"""
# digitsデータを閾値th でdataから0かそれ以外の数字化を判定する関数
def is_zero(th,data):
    if th <= data:
        return "zero"
    else:
        return "other"