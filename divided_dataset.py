import os
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def compute(path, save_dir):
    file_names = os.listdir(path)
    file_name_green = []
    file_name_none = []
    for file_name in tqdm(file_names):
        img = cv2.imread(os.path.join(path, file_name))
        per_image_Bmean = np.mean(img[:, :, 0])
        per_image_Gmean = np.mean(img[:, :, 1])
        per_image_Rmean = np.mean(img[:, :, 2])
        if per_image_Bmean > 65 and per_image_Gmean > 65 and per_image_Rmean > 65:
            file_name_green.append(file_name)
        else:
            file_name_none.append(file_name)
    file = open(os.path.join(save_dir, path.split('/')[-1]+'_green.txt'), 'w')

    for filename in sorted(file_name_green):
        file.write(str(filename+'\n'))
    file.close()
    file = open(os.path.join(save_dir, path.split('/')[-1]+'_normal.txt'), 'w')
    for filename in sorted(file_name_none):
        file.write(str(filename+'\n'))
    file.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='vis txt result Tool')
    parser.add_argument('--data_dir_query', help='dir to the query datasets')
    parser.add_argument('--data_dir_gallery', help='dir to the gallery datasets')
    parser.add_argument('--save_dir', help='dir to save the generated txt file')
    args = parser.parse_args()
    compute(args.data_dir_query, args.save_dir)
    compute(args.data_dir_gallery, args.save_dir)


