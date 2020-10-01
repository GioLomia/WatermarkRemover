import cv2
import os
import random as rnd
import numpy as np
import time
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import string
from PIL import Image

import gen_wm

def plot_rgb_img(img):
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.axis('off')
  plt.show()

def apply_watermark (im, watermark, transperency):
    """
    Apply watermart on a given image

    Args:
        im (cv2.image): clean image
        watermark (cv2.image): watermart
        0 < transperency < 1 (float): the level of transperancy of the watermark

    Returns:
       cv2.image: Output image
    """
    background = im
    shape =  background.shape
    shape = ((shape[1],shape[0]))
    overlay = cv2.resize(watermark,shape)
    wm_image = cv2.addWeighted(background,1,overlay,transperency,0)

    print(wm_image.shape)
    return wm_image

def load_data (dir):
    im_ls = os.listdir(dir)
    print(im_ls)
    dataset = []
    for i in range(len(im_ls)):

        im = cv2.imread(f"{dir}{im_ls[i]}")
        print(im)
        dataset.append(im)

    return np.array(dataset)

def randomize_watermark (seed, watermark, max_rotation = 0.7, max_sheer = 0.4):
    """
    Adds image augmentation to the watermark to improve generalization and reduce overfitting    

    Args:
        seed (int): seed to allow for scientific method design
        watermark (cv2.image): the watermark image
        max_rotation (float, optional): the range of rotation. Defaults to 0.7.
        max_sheer (float, optional): the range of max shearing. Defaults to 0.4.

    Returns:
        cv2.image: resulting output image
    """
    # rnd.seed(seed)
    rotation_angle = rnd.uniform(0,max_rotation)
    

    ##################Image Rotation Block###########################
    image_center = tuple(np.array(watermark.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle * 100, 1.0)
    result = cv2.warpAffine(watermark, rot_mat, watermark.shape[1::-1], flags=cv2.INTER_LINEAR)
    #################################################################

    #########################Flip Block##############################
    h_flip = rnd.randint(0,1)
    v_flip = rnd.randint(0,1)
    
    print(h_flip,v_flip)
    if h_flip == 1:
        result = cv2.flip(result,0)

    if v_flip == 1:
        result = cv2.flip(result,1)
    #################################################################

    return result

def generate_wm_directory(orig_dir, output_dir, wm_dir):

    all_im_ls = os.listdir(orig_dir)
    wm_ls = os.listdir(wm_dir)
    for i in range(len(all_im_ls)):
        
        wm_path = rnd.choices(wm_ls)
        wm = randomize_watermark(1,cv2.imread(f"{wm_dir}{wm_path[0]}"))
        im = cv2.imread(f"{orig_dir}{all_im_ls[i]}")
        print(im.shape)
        transperency = rnd.uniform(0.5, 0.65)

        wm_type = rnd.choice(["simple_apply", "random_apply", "grid_apply"])

        print(wm_type, all_im_ls[i])
        if wm_type == "simple_apply":
            wm_im = apply_watermark(im, wm, transperency)
        elif wm_type == "random_apply":
            
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(im)

            wm_im = gen_wm.apply_rnd_watermark(im_pil)
            wm_im = np.array(wm_im)
            wm_im = wm_im[:, :, ::-1].copy() 
        else:

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(im)

            wm = cv2.cvtColor(wm, cv2.COLOR_BGR2RGB)
            wm_pil = Image.fromarray(wm)

            wm_im = gen_wm.apply_grid_watermark(im_pil,wm_pil)

            wm_im = np.array(wm_im)
            wm_im = wm_im[:, :, ::-1].copy()

        print(f"{output_dir}{all_im_ls[i]}")
        wrt = cv2.imwrite( f"{output_dir}{all_im_ls[i]}", wm_im)
        print(wrt)

        if i+1 > len(os.listdir(output_dir)):
            print("ERROR!!!!!!!!!!!!!!!!!!!")
            exit()
        print(i,f"{(100*i/len(all_im_ls))}% done")
        print()

def toss_ims2dirs(source_dir, out_dir, dataset_size, data_split = (0.8, 0.15, 0.05), same = True):
    im_data_source = os.listdir(f"{source_dir}Original_Ims/")
    wm_data_source = os.listdir(f"{source_dir}WM_Images/")
    rnd.shuffle(im_data_source)
    rnd.shuffle(wm_data_source)
    os.makedirs(f"{out_dir}Train/", exist_ok = True)
    os.makedirs(f"{out_dir}Val/", exist_ok = True)
    os.makedirs(f"{out_dir}Test/", exist_ok = True)
    for i in range(dataset_size):
        ratio = i/dataset_size
        
        if ratio <= data_split[2]:    
            sub_dir = "Test"
            print(i, ratio,sub_dir)
        elif ratio - data_split[2]<= data_split[1]:
            sub_dir = "Val"
            print(i, ratio,sub_dir)
        else:
            sub_dir = "Train"
            print(i, ratio,sub_dir)

        os.makedirs(f"{out_dir}{sub_dir}/Original_Ims/", exist_ok = True)
        os.makedirs(f"{out_dir}{sub_dir}/WM_Images/", exist_ok = True)
        os.makedirs(f"{out_dir}{sub_dir}/Original_Ims/original/", exist_ok = True)
        os.makedirs(f"{out_dir}{sub_dir}/WM_Images/watermark/", exist_ok = True)


        shutil.copy(f"{source_dir}Original_Ims/{im_data_source[i]}",f"{out_dir}{sub_dir}/Original_Ims/original/{im_data_source[i]}")
        if same:
            shutil.copy(f"{source_dir}WM_Images/{im_data_source[i]}",f"{out_dir}{sub_dir}/WM_Images/watermark/{im_data_source[i]}")
        else:
            shutil.copy(f"{source_dir}WM_Images/{wm_data_source[i]}",f"{out_dir}{sub_dir}/WM_Images/watermark/{im_data_source[i]}")

    print(len(os.listdir(f"{out_dir}/Train/Original_Ims/")))
    print(len(os.listdir(f"{out_dir}/Val/Original_Ims/")))
    print(len(os.listdir(f"{out_dir}/Test/Original_Ims/")))

    print(len(os.listdir(f"{out_dir}/Train/WM_Images/")))
    print(len(os.listdir(f"{out_dir}/Val/WM_Images/")))
    print(len(os.listdir(f"{out_dir}/Test/WM_Images/")))


def main ():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    _SEED_ = 1
    im_path = "Data/Big_Dataset/Original_Ims/0a0c901d1cdc555d.jpg"
    wm_path = "Data/Watermarks/"
    wm_list = os.listdir(wm_path)

    # print(get_random_string(3))
    #Run this function first after it is done comment it out.
    # generate_wm_directory("Data/Better_Marked_DS/Original_Ims/", "Data/Better_Marked_DS/WM_Images/", "Data/Watermarks/")
    
    #Then run this function
    # toss_ims2dirs("Data/Better_Marked_DS/", "Data/BetterDataset1000/", 30000)

    toss_ims2dirs("Data/Big_Dataset/", "Data/Dataset1000/", 1500)
if __name__ == "__main__":
    main()






