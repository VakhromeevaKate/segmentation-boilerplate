import pandas as pd
import os
import re

img_path = 'data/food/img'
file_img_path = 'food/img'
mask_path = 'data/food/masks_machine'
file_mask_path = 'food/masks_machine'

fds = sorted(os.listdir(img_path))

imageList = []
maskList = []

for i, img in enumerate(fds, start=0):
    mask_filename = os.path.splitext(os.path.basename(os.path.join(img_path, img)))[0]
    mask = mask_filename.join(["", ".png"])

    imageList.append(os.path.join(file_img_path, img))
    maskList.append(os.path.join(file_mask_path, mask))

d = {"image": imageList, "mask": maskList}
df = pd.DataFrame(d)

df1 = df.iloc [:200]
df2 = df.iloc [200:]

df1.to_csv('data/train.csv', index= False)
df2.to_csv('data/test.csv', index= False)
