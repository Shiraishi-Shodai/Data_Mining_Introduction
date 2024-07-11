from wordcloud import WordCloud
import cv2
import re
from janome.tokenizer import Tokenizer

def wakachigaki(text):
    token = Tokenizer().tokenize(text)
    words = []
    for line in token:
        tkn = re.split('\t|,', str(line))
        # tkn[0]が存在し、tkn[1]が名詞でtkn[2]が一般または固有名詞であるか
        if tkn[0] and tkn[1] in ['名詞'] and tkn[2] in ['一般', '固有名詞'] :
             if tkn[0] != '漱石':
                words.append(tkn[0])
    return ' ' . join(words)

text = open("/media/shota/share2/education/Data_Mining_Introduction/sec24/soseki.txt", encoding="utf8").read()
text = wakachigaki(text)
wordcloud = WordCloud(max_font_size=400,width=900,height=600,font_path='C:/Windows/Fonts/HGRSGU.TTC').generate(text)
wordcloud.to_file("result.png")

img = cv2.imread("result.png")
cv2.imshow("Image", img)
cv2.waitKey()

