##import
import numpy as np ## numpy 설정
from PIL import Image


# 같은 경로에 있는 dataset안에 mnist.py 파일을 불러온다.
from dataset.mnist import load_mnist

#이미지 띄워주는 함수
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


## flatten - 이미지 데이터를 1차원 벡터로 표현할 것인지 설정
## normalize - Pixel 0 ~ 255 사이의 데이터를 0.0 ~ 1.0 으로 정규화 할것인지 설정
## one_hot_label - 정답을 뜻하는 원소만 1 이고 나머지는 0인 배열 
(X_Train , T_Train), (X_Test , T_Test) = load_mnist(flatten = True, normalize = False)

## 학습 데이터 형상 표시
# print(X_Train.shape)
# print(T_Train.shape)
# print(X_Test.shape)
# print(T_Test.shape)

img = X_Train[0]
label = T_Train[0]

img = img.reshape(28,28)
img_show(img)


