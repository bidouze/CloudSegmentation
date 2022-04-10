import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()


path = cwd+'/data/'
test_imgs_folder = path+'/test_images'
train_imgs_folder = path+'/train_images'
train_annot_folder = path+'/train_annotations_1/'
checkpoint_filepath = cwd+'/Model 1_LLA'


path_train = os.path.join('r', path, 'train.csv')

df_train = pd.read_csv(path_train)

df_train['id_image'] = df_train['Image_Label'].apply(lambda row: row.split('_')[0])
df_train['cloud_type'] = df_train['Image_Label'].apply(lambda row: row.split('_')[1])
df_train = df_train.set_index('id_image')
df_train['ImageId'] = df_train['Image_Label'].apply(lambda row: row.split('_')[0])


# Annotation v2 : seulement 4 classes

rep = train_annot_folder

def create_annotation_v2(df_train):
  nb_rows = len(df_train)
  id_row = 1
  precentage_prec = 0
  for row in df_train.iterrows():
    row = row[1]
    mask = rle_decode(row[1])
    if mask is None:
      mask = np.zeros((1400,2100), np.float32)
    # A ce niveau là pour chaque ligne et donc chaque mask on a une matrice de 0 et de 1
    # Création de l'image
    mask_name = row[0].split('.')[0] + "_" + row[2] + ".png"
    path = os.path.join(rep, mask_name)
    plt.imsave(path, mask, vmin = 0, vmax = 255, cmap ='gray')

    perc = round(id_row/nb_rows*100, 0)
    if perc!=precentage_prec:
      print("{} % of images saved ".format(perc))
      precentage_prec = perc
    id_row +=1
  

def encode_mask(mask, c_type):
  
  dico = {'Fish' : 9, 'Flower' : 11, 'Gravel' : 7, 'Sugar' : 3}
  return mask*dico[c_type]

def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if not isinstance(mask_rle, np.float):
      s = mask_rle.split()
      starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
      starts -= 1
      ends = starts + lengths
      img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
      for lo, hi in zip(starts, ends):
          img[lo:hi] = 1
      return img.reshape(shape, order='F')