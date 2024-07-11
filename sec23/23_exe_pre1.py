"""
演習第23回
(1)元の文章に含まれる「名詞」だけを抜き出し、順に分かち書きして並べる
(2) TF-IDFで数値にする。
"""
from janome.tokenizer import Tokenizer
import re
tokenizer = Tokenizer()

#特定のファイルからテキストを取り出す。クリーニングもこの関数でやる予定
def get_txt(file_name):
    with open(file_name,'r', encoding='UTF-8') as f:
        txt = f.read()
        txt = txt.replace("\n", "")
        txt = re.sub(r'《.+?》','',txt)#青空文庫のルビ対策
    #txt_array = txt.split('。') 
    return txt

# 引数を自然言語解析して分かち書きされたテキストかリストで返す
def nou_waka(txt_t,mode="txt",only_noum=False):
    tokens = tokenizer.tokenize(txt_t,wakati=False)
    wakati_list = []
    # Janameの最新バージョンでは、tokenizeはジェネレータオブジェクトを返す
    for token in tokens:
        if only_noum:
            if token.part_of_speech.split(',')[0] =='名詞':
                wakati_list.append(token.surface)
        else:
            wakati_list.append(token.surface)        
    
    if mode=="txt":
        return " ".join(wakati_list)
    else:
        return wakati_list

#txt ="吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。"

# 文章を。で区切って、名詞だけを取り出して分かち書きにして、配列にしたものを返す
def txt_to_list(txt):
    return_list =[]
    for t in txt.split('。'):
        print(t)
        print(nou_waka(t,mode="txt"))
        return_list.append(nou_waka(t,mode="txt"))
        #print("")
    return return_list

txt = get_txt("target_text.txt")
print(txt_to_list(txt))