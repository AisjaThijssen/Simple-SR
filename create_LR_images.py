import os
import sys
from multiprocessing import Pool

import cv2
from torchvision import transforms

from utils.common import ProgressBar
from utils.resizer import imresize
from PIL import Image

def main():
    input_path='datasets/BSDS100/BSDS100_HR'
    output_path='datasets/BSDS100/BSDS100_LR'
    n_thread = 20

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs("datasets/BSDS100/BSDS100_HR_mod4")
        print('mkdir [{:s}] ...'.format(output_path))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(output_path))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_path)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    for path in img_list:
        resizer(path, output_path)

    print('All subprocesses done.')

def resizer(path, output_folder):
    img_name = os.path.basename(path)
    img = Image.open(path)

    img = img.resize((img.size[0] - (img.size[0] % 4), img.size[1] - (img.size[1] % 4)))

    img.save(os.path.join("datasets/BSDS100/BSDS100_HR_mod4", img_name))

    tenor_convertor = transforms.ToTensor()
    tensor_image = tenor_convertor(img)

    resized_image = imresize(tensor_image, scale=1/4)

    pil_convertor = transforms.ToPILImage()
    pil_image = pil_convertor(resized_image)
    print(pil_image)
    pil_image.save(os.path.join(output_folder, img_name.replace('.png', '')+'_LRx4.png'))

if __name__ == '__main__':
    main()



