from PIL import Image
import os
import random

def randomPosition():

    pass

backgrounds_path = 'dataset/background for augmentation'
bills_path = 'dataset\png augmentation'
backgrounds = os.listdir(backgrounds_path)


for bill in os.listdir(bills_path):

    for imgen in backgrounds:

        bill_img = Image.open(os.path.join(bills_path, bill))
        img = Image.open(os.path.join(backgrounds_path,imgen))
        img_w, img_h = img.size
        bill_img = bill_img.resize((int(img.size[0]/4), int(img.size[1]/4)))
        mask = Image.new('L', bill_img.size, 255)
        new = Image.new('L',(img_w *2, img_h *2), 255)
        rot = 45
        bill_img = bill_img.rotate(rot, expand=True)
        mask = mask.rotate(rot, expand=True)
        img.paste(bill_img, (img_w -100, img_h -100), mask=mask)
        img.show()
        