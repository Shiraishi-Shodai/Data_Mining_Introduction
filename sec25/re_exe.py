from wordcloud import WordCloud
import re

from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        tkn = re.split("\t|,", str(line))
        if tkn[0] and tkn[1] in ["名詞"] and tkn[2] in ["一般", "固有名詞"]:
            if tkn[0] != "漱石":
                words.append(tkn[0])
    return ' '.join(words)

def main():
    with open("mizuhara.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    wakachi_text = wakachigaki(text)
    wordcloud = WordCloud(max_font_size=400, width=900, height=600, font_path="C:/Windows/Fonts/HGRSGU.TTC").generate(wakachi_text)
    wordcloud.to_file("re.png")


if __name__ == "__main__":
    main()


