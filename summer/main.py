import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib
import cv2
import glob
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import itertools
import seaborn as sns
import sys
import re


"""
レポートのリンク
https://docs.google.com/document/d/1dhwxJUNEezEtucHkjEleQRjPS92ktL3cK54MWFshenE/edit?usp=sharing
"""


def change_number(text):
    """受け取ったテキストが数字なら整数に変換し、そうでなければそのまま返す
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """テキストに数字があれば数字とその前後でテキストを分割し、change_numberに分割した値を渡す
    """
    return [change_number(i) for i in re.split(r'(\d+)', text)]
    
def get_rice_size(img_path):
    """グレースケール化し、白色の部分の大きさを返す。
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rice_size = cv2.countNonZero(img)
    return rice_size

def predict(rice_size):
    """rice_sizeが620より小さければ0を返し、そうでなければ1を返す(0が割れ米、1が青龍米)
    """
    if rice_size < 620:
        return 0
    else:
        return 1
 
def main():
    
    test_dir_path = sys.argv[1]
    test_image_list = sorted(glob.glob(test_dir_path + "/*"), key=natural_keys)

    ans_dict = {
        0: "ware-mai",
        1: "seiryu-mai"
    }
    
    for test_image_path in test_image_list:

        test_image_name = os.path.basename(test_image_path)
        rice_size = get_rice_size(test_image_path)
        y_pred = predict(rice_size)
        
        print(f"{test_image_name}: {ans_dict[y_pred]}")
    
if __name__ == "__main__":
    main()