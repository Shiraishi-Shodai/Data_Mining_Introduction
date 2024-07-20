import re
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud,ImageColorGenerator
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def wakachigaki(text):
    tokens = Tokenizer().tokenize(text)
    words = []
    for line in tokens:
        tkn = re.split('\t|,', str(line))
        # tkn[0]が存在し、tkn[1]が名詞でtkn[2]が一般または固有名詞であるか
        if tkn[0] and tkn[1] in ["名詞"] and tkn[2] in ["一般", "固有名詞"]:
            words.append(tkn[0])
    return " ".join(words)

def getword_cloud(wakachi_text:str, mask_img_path: str)-> None:
    mask_img = np.array(Image.open( mask_img_path ))
    img_color = ImageColorGenerator(mask_img)
    wordcloud = WordCloud(max_font_size=400,
        width=900,
        height=600,
        font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        mask=mask_img,
        # background_color="white",
        stopwords=["水原一平","水原","大谷", "選手", "月"],
        color_func=img_color).generate(wakachi_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud.to_file("korekore.png")

def main():
    with open("korekore.txt", encoding="utf-8") as f:
        text = f.read()
    wakachi_text = wakachigaki(text)

    mask_img_path = "mizuhara_mask.png"
    getword_cloud(wakachi_text, mask_img_path)


if __name__ == "__main__":
    main()
