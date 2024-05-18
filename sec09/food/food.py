import glob
import re
from PIL import Image
import numpy as np

# ファイルのパスを引数に、その画像の16次元のベクトルデータとラベルを返す
def image2vector(img_path):
    image_c = Image.open(img_path).convert("L")
    image_c = image_c.resize((8,8),Image.ANTIALIAS)
    img = np.asarray(image_c,dtype=float)
    img = 16 - np.floor(17*img/256) #0-255 の値を白黒逆で16階調にする
    img = img.flatten()

    stype = 0 # 1＝〇 ,2=三角 , 3=四角,4= バツ ,
    if re.match('.+_sushi', img_path):
        stype = 1
    elif re.match('.*fries.+', img_path):
        stype = 2
    else:
        stype = -1
        print("Error!"+str(img_path))
    
    return [img,stype]

# pngファイルのあるパスを指定し、中のpngファイルの16次元のベクトルとラベルベクトルを返す
def dir2tensor(dir_path='./jpg'):
    pngs = glob.glob(dir_path + "/*.jpg")

    png_data =np.array(
    [ 0. ,1. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,7. ,4. ,0. ,0. ,5. ,3. ,0. ,0. ,1. ,
    9. ,1. ,7. ,8. ,0. ,0. ,0. ,0. ,5. ,13. ,5. ,0. ,0. ,0. ,0. ,5. ,7. ,7. ,
    5. ,0. ,0. ,0. ,2. ,6. ,0. ,0. ,9. ,3. ,0. ,0. ,0. ,0. ,0. ,0. ,2. ,6. ,
    0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.]
    )
    #print(len(png_data))
    shape_clf =np.array([1]) #個々のpng画像のラベルを確報するリスト
    for png in pngs:
        png_vec ,stype = image2vector(png)
        png_data = np.concatenate([png_data, png_vec],0)
        shape_clf= np.concatenate([shape_clf, [stype]])
    
    #ここの処理雑
    png_data =  png_data.reshape(-1,64)
    png_data =  np.delete(png_data, 0, 0)
    #shape_clf = shape_clf.reshape(1,-1)
    #print(shape_clf)
    shape_clf = np.delete(shape_clf, 0, 0)
    return [png_data,shape_clf]
