from PIL import Image
import os

img_dir = './results/BIWI_masked/03'
save_gif = './results/BIWI_masked/03.gif'
images = []
img_list = os.listdir(img_dir)
img_list.sort()

for img_name in img_list[100:]:
    im = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    im = im.resize((im.width // 2, im.height // 2))
    images.append(im)

images[0].save(save_gif, save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)