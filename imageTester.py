from PIL import Image
import matplotlib.pyplot as plt


#grab image
#testImage = Image.open('images/kitty.jpg')
new_image = plt.imread('images/kitty.jpg')
plt.imshow(new_image)
plt.show()
#testImage.show()

