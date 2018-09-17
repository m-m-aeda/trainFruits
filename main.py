import PIL
import os, glob
import numpy as np
import random, math

# 画像が保存されているルートディレクトリのパス
root_dir = "/Users/mitsukim/PythonProjects/ayataka/images"
# 商品名
categories = ["orange", "lemon", "apple"]

# 画像がデータ用配列
X = []
# ラベルデータ用配列
Y = []

# 画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    pass

# 渡された画像データを読み込んでXに格納し、
# 画像データに対応するcategoryのidxをYに格納する関数
def add_sample(category_index, file_name):
    pass

# 全データ格納用配列
allfiles = []

# カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for index, category in enumerate(categories):
    image_dir = root_dir + "/" + category
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((index, f))

# シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
number_of_train = math.floor(len(allfiles) * 0.8)
train = allfiles[0:number_of_train]
test = allfiles[number_of_train:]
X_train, Y_train = make_sample(train)
X_test, Y_test = make_sample(test)
xy = (X_train, Y_train, X_test, Y_test)
# データを保存する（データの名前を「fruit_data.npy」としている
np.save(root_dir + "/fruit_data.npy")


