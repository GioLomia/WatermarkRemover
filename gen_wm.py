import os, shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw, ImageOps
import string
import numpy as np
import cv2
import random


def create_rnd_txt_wm ():

    prints = list(string.printable)[0:84]
    #open all of the images from the VOC2008 dataset as jpegs

    font_size = np.random.randint(low = 50, high = 350)
    
    #create the watermark font for the image
    font = ImageFont.truetype("arial.ttf", font_size) 
    
    #generate image to hold the watermark text object
    img_temp = Image.new('L', (350,350))
    
    #create the watermark text, of random length, using random printable characters
    text_str = np.random.choice(prints, np.random.randint(low=5, high = 14))
    text_str = "".join(text_str)
    
    #draw on temporary image with text
    draw_temp = ImageDraw.Draw(img_temp) 
    
    #generate a random integer for the opacity argument (fill)
    opac = np.random.randint(low= 70, high=150)
    
    #insert text onto the temporary image
    draw_temp.text((0, 0), text_str,  font=font, fill=opac)
    
    #generate a random integer for rotation:
    rot_int = np.random.randint(low = 0, high = 180)
    
    #rotate the text on the temporary image
    rotated_text = img_temp.rotate(rot_int,  expand=1)
    
    #generate a random location for the watermark on the image
    #merge the temporary image with text with the image passed in 
    #third tuple also needs to be random: controls the location of the img
    return rotated_text

def apply_rnd_watermark(im):

    img = im
    
    for i in range(np.random.randint(low=1, high=3)):

        #default color of watermark set to white; change if desired
        col_1 = (255,255,255)
        col_2 = (255,255,255)
        width, height = img.size

        # #generate a random location for the watermark on the image
        rand_loc = (np.random.randint(low=10,high=width-width*0.3), np.random.randint(low=10,high=height-height*0.3))

        rotated_text = create_rnd_txt_wm()
        img.paste(ImageOps.colorize(rotated_text, col_1, col_2), rand_loc,  rotated_text)
    
    #this yeilds a new image with a watermark
    #save this jpeg with a watermark to the WATS directory

    return img


def apply_grid_watermark(im, watermark_1):

    main = im
    mark = random.choice([watermark_1, create_rnd_txt_wm().convert("RGBA")])

    mask = mark.convert('L').point(lambda x: min(x, np.random.randint(low = 100, high= 170)))
    mark.putalpha(mask)

############################################################
    #generate a random integer for rotation:
    rot_int = np.random.randint(low = 0, high = 180)

    mark_width, mark_height = mark.size
    main_width, main_height = main.size
    aspect_ratio = mark_width / mark_height
    new_mark_width = main_width * random.uniform(0.125, 0.35)
    
    mark = mark.rotate(rot_int,  expand=1)

    mark.thumbnail((new_mark_width, new_mark_width / aspect_ratio), Image.ANTIALIAS)
############################################################

    tmp_img = Image.new('RGB', main.size)

    for i in range(0, tmp_img.size[0], mark.size[0]):
        for j in range(0, tmp_img.size[1], mark.size[1]):
            main.paste(mark, (i, j), mark)
            main.thumbnail((8000, 8000), Image.ANTIALIAS)
            # main.save(f"{out_str}{im_name}", quality=100)
    return main




