from PIL import Image
import os
import random

backgrounds_path = 'dataset/background for augmentation'
bills_path = 'dataset\png augmentation'
dataset_path = 'dataset/bill_dataset'
backgrounds = os.listdir(backgrounds_path)

reuse_same_background = 6

def randomPosition(background_size: tuple) -> tuple:
    width = random.randint(0, background_size[0])
    height = random.randint(0, background_size[1])
    return (width, height)


for bill_folder in os.listdir(bills_path):
    pos = 0
    for bill in os.listdir(os.path.join(bills_path, bill_folder)):
        for background in backgrounds:
            for i in range(reuse_same_background):

                bill_img = Image.open(os.path.join(bills_path, bill_folder, bill))
                background_img = Image.open(os.path.join(backgrounds_path, background))

                img_w, img_h = background_img.size
                bill_w, bill_h = bill_img.size

                # W/H => H = W/aspect
                aspect_ratio = bill_w/bill_h

                bill_img = bill_img.resize((int(img_w/3), int((img_w/3)/aspect_ratio)))

                bill_w, bill_h = bill_img.size
                mask_bill = Image.new('L', bill_img.size, 255)
                mask_background = Image.new('L', background_img.size, 255)
                new = Image.new('RGB',(img_w + bill_w, img_h + bill_h), 255)

                new.paste(background_img, (int(bill_w/2), int(bill_h/2)), mask_background)

                rot = random.randint(0, 360)
                bill_img = bill_img.rotate(rot, expand=True)
                mask_bill = mask_bill.rotate(rot, expand=True)
                new.paste(bill_img, randomPosition(background_img.size), mask=mask_bill)
                new = new.crop((bill_w/2, bill_h/2 ,img_w + bill_w/2, img_h + bill_h/2))

                new.save(os.path.join(dataset_path, bill_folder, f'augmented-{pos}.jpeg'))
                pos += 1
            

            
        
        
        