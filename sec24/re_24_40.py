from wordcloud import WordCloud
import cv2
import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        print(line)
        tkn = re.split("\t|,", str(line))
        if tkn[0] and tkn[1] in ["名詞"] and tkn[2] in ["一般", "固有名詞"]:
            if tkn[0] == "漱石":
                continue
            words.append(tkn[0])
    return " ".join(words)

text = open("soseki.txt", encoding="utf-8").read()
text = wakachigaki(text)

wordcloud = WordCloud(max_font_size=400,width=900,height=600, font_path="C:/Windows/Fonts/HGRSGU.TTC").generate(text)
wordcloud.to_file("result.png")

img = cv2.imread("result.png")
cv2.imshow("Image", img)
cv2.waitKey()