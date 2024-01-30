
from PAR import PAR
from PIL import Image

par = PAR()

image = Image.open("train_image/17.jpg")

att = par.attribute_recognition(image)

print(att)
