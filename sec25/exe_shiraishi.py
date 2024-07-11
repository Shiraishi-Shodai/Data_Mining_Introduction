import re
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud

def wakachigaki(text):
    tokens = Tokenizer().tokenize(text)
    words = []
    for line in tokens:
        tkn = re.split('\t|,', str(line))
        # tkn[0]が存在し、tkn[1]が名詞でtkn[2]が一般または固有名詞であるか
        if tkn[0] and tkn[1] in ["名詞"] and tkn[2] in ["一般", "固有名詞"]:
            if tkn[0] != "水原一平" and tkn[0] != "水原" and tkn[0] != "大谷":
                words.append(tkn[0])
    return " ".join(words)

def main():
    with open("mizuhara.txt", encoding="utf-8") as f:
        text = f.read()
    wakachi_text = wakachigaki(text)    
    
    wordcloud = WordCloud(max_font_size=400,width=900,height=600, font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").generate(wakachi_text)
    wordcloud.to_file("mizuhara.png")

if __name__ == "__main__":
    main()