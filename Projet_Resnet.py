import streamlit as st
import os
st.title('Loopy Clouds')

st.markdown("Ce projet est à l’origine une compétition Kaggle à l’initiative du Max Planck Institute of Meteorology. Le but de cette compétition est de pouvoir identifier quatre types de formations nuageuses (« gravel », « fish », « flower » et « sugar ») sur des images satellite. \n Ces formations nuageuses jouent un rôle déterminant sur le climat et sont difficiles à comprendre et à implémenter dans des modèles climatiques. En classant ces formations nuageuses, les chercheurs espèrent mieux les comprendre et améliorer les modèles existants.")

from PIL import Image
path_base = os.path.dirname(__file__)
img = Image.open(path_base+"/img1.png")
st.image(img)

st.header("Importation des librairies")

with st.echo():
  # Tensorflow
  import tensorflow as tf
  import tensorflow.keras.backend as K
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, ZeroPadding2D, UpSampling2D, Concatenate, Input
  from tensorflow.keras.models import Model, load_model, Sequential
  from tensorflow.keras.utils import Sequence
  from tensorflow.keras.activations import relu, sigmoid
  from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
  import tensorflow.keras as keras


  # Scikit-learn pour la création du jeu de validation
  from sklearn.model_selection import train_test_split

  # Pour sauvegarde des poids
  import h5py

  # Augmentation
  import albumentations as A
  from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion,CenterCrop

  # Data processing
  import pandas as pd 
  import numpy as np 

  # Data'viz
  import matplotlib.pyplot as plt
  import matplotlib.patches as ptch
  import seaborn as sns

  # Others
  import os
  import random
  import cv2
  import glob
  import multiprocessing
  from copy import deepcopy
  import statistics

with st.echo():
  SM_FRAMEWORK=tf.keras
  from segmentation_models import Unet
  from segmentation_models import get_preprocessing

st.header("Preprocessing")

st.subheader("Définition des répertoires de travail")

with st.echo():
  path = path_base+'/data/'
  test_imgs_folder = path+'/test_images/'
  train_imgs_folder = path+'/train_images/'
  train_annot_folder = path+'/train_annotations_1/'
  checkpoint_folder = path_base+'/Model/'
  optimizer_wieghts = path+'/optimizer_state/'

st.header("Préparation du csv")

with st.echo():
  path_train = os.path.join('r', path, 'train.csv')

  df_train = pd.read_csv(path_train)

  df_train['id_image'] = df_train['Image_Label'].apply(lambda row: row.split('_')[0])
  df_train['cloud_type'] = df_train['Image_Label'].apply(lambda row: row.split('_')[1])
  df_train = df_train.set_index('id_image')
  df_train['ImageId'] = df_train['Image_Label'].apply(lambda row: row.split('_')[0])

st.dataframe(df_train.head())

st.subheader("Mise en place des train et test sets")

with st.echo():
  train_imgs, test_imgs = train_test_split(df_train['ImageId'].unique(),
                                         test_size = 0.2)

with st.echo():
  preprocess_input = get_preprocessing('resnet34')

with st.echo():
  train_imgs = preprocess_input(train_imgs)
  test_imgs = preprocess_input(test_imgs)

st.write("Nombre d'images train :", len(train_imgs))
st.write("Nombre d'images test :", len(test_imgs))

st.header("Data Augmentation")

with st.echo():
  albumentations_train = A.Compose([
      A.VerticalFlip(p=.5), #Effet miroir vertical
      A.Rotate(limit=20, interpolation = 0, p=.5), #Rotation 
      A.HorizontalFlip(p=.5),
      A.RandomSizedCrop(min_max_height=(200, 200), height=256, width=416, interpolation = 0), #Crop sur l'image
  ], p=.5, 
      additional_targets={'img' : 'image', 'mask_gravel': 'mask', 'mask_fish': 'mask','mask_flower': 'mask', 'mask_sugar': 'mask' }) 

model = Unet('resnet34', encoder_weights='imagenet' ,classes = 4, input_shape = (256, 416, 3), activation = 'sigmoid', encoder_freeze =True)

st.header("Data generator")

with st.echo():
  output_height = model.output.get_shape()[1]
  output_width = model.output.get_shape()[2]

  input_height = model.input.get_shape()[1]
  input_width = model.input.get_shape()[2]

with st.echo():
  BATCH_SIZE = 8
  class DataGenenerator(Sequence):
      def __init__(self, images_list=None, folder_imgs=train_imgs_folder, folder_annot=train_annot_folder,
                   batch_size=BATCH_SIZE, shuffle=True, augmentation=None, is_test = False, output_height = output_height,
                   output_width = output_width, resized_height=input_height, resized_width=input_width, num_channels=3, num_classes = 4):
          # Taille du batch
          self.batch_size = batch_size
          # Variable pour le mélange aléatoire des images
          self.shuffle = shuffle
          # Gestion de la data augment
          self.augmentation = augmentation
          #Retourne la liste des images 
          if images_list is None:
              self.images_list = os.listdir(folder_imgs)
          else:
              self.images_list = deepcopy(images_list)
          # Nom du répertoire des images
          self.folder_imgs = folder_imgs
          # Nom du répertoire des annotations
          self.folder_annot = folder_annot
          # Nb d'itérations pour chaque epoch
          self.len = len(self.images_list) // self.batch_size
          # Hauteur de l'image redimensionnée
          self.resized_height = resized_height
          # Largeur de l'image redimensionnée
          self.resized_width = resized_width
          # Profondeur de la dernière dimension (couleurs). Par défaut = 3 (RGB)
          self.num_channels = num_channels
          # Nb de classes 
          self.num_classes = num_classes
          self.is_test = not 'train' in folder_imgs
          #self.encoding_list = dico_encod
          self.output_height = output_height
          self.output_width = output_width

      # Retourne le nombre d'itérations par epoch
      def __len__(self):
          return self.len
    
      # Tri aléatoire des images
      def on_epoch_start(self):
          if self.shuffle:
              random.shuffle(self.images_list)

      def __getitem__(self, idx):
          current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]
          X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
          y = np.empty((self.batch_size, self.output_height, self.output_width, self.num_classes))

          for i, image_name in enumerate(current_batch):
              path_img = os.path.join(self.folder_imgs, image_name)
              path_annot1 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Gravel.png')
              path_annot2 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Fish.png')
              path_annot3 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Flower.png')
              path_annot4 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Sugar.png')
             
              img = cv2.imread(path_img).astype(np.float32)
              mask_gravel = np.expand_dims(cv2.resize(cv2.imread(path_annot1, cv2.IMREAD_GRAYSCALE).astype(np.int32),(self.output_width, self.output_height), interpolation =0).astype(np.int32) ,2)
              mask_fish = np.expand_dims(cv2.resize(cv2.imread(path_annot2, cv2.IMREAD_GRAYSCALE).astype(np.int32),(self.output_width, self.output_height), interpolation =0).astype(np.int32) ,2)
              mask_flower = np.expand_dims(cv2.resize(cv2.imread(path_annot3, cv2.IMREAD_GRAYSCALE).astype(np.int32),(self.output_width, self.output_height), interpolation =0).astype(np.int32) ,2)
              mask_sugar = np.expand_dims(cv2.resize(cv2.imread(path_annot4, cv2.IMREAD_GRAYSCALE).astype(np.int32),(self.output_width, self.output_height), interpolation =0).astype(np.int32) ,2)  
            
              if not self.augmentation is None:
                  random.seed(np.random.randint(10000))
                  # Application de la data augmentation sur l'image et le masque
                  augmented = self.augmentation(image=img,  mask_gravel=mask_gravel, mask_fish=mask_fish, mask_flower=mask_flower, mask_sugar=mask_sugar)
            
              X[i, :, :, :] = cv2.resize(img, (self.resized_width, self.resized_height), interpolation = 0)/255.0

              # One Hot encdoding
              if not self.is_test:
                seg_labels = np.concatenate((mask_gravel, mask_fish, mask_flower, mask_sugar), axis = 2)
                y[i, :, :, :] = seg_labels
          return X, y

st.header("Loss and metrics functions")

with st.echo():
  def masked_bce(ytrue, ypred):
      y_sum = tf.reduce_sum(ytrue, axis = [1,2])
      y_where = tf.where(y_sum >= 1, 1, 0)
      y_where = tf.reshape(y_where,[BATCH_SIZE,1,1,4])
      y_prod = ypred*tf.cast(y_where, tf.float32)
      return tf.reduce_mean(K.binary_crossentropy(tf.cast(ytrue, tf.float32),tf.cast(y_prod, tf.float32)))

  def dice_coef(y_true, y_pred, smooth=1):
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      intersection = K.sum(y_true_f * y_pred_f)
      return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def dice_loss(y_true, y_pred):
      smooth = 1.
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      intersection = y_true_f * y_pred_f
      score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
      return 1. - score

  def bce_dice_loss(y_true, y_pred):
      return tf.reduce_mean(binary_crossentropy(y_true, y_pred)) + dice_loss(y_true, y_pred)

  def masked_bce_dice_loss(y_true, y_pred):
      return masked_bce(y_true, y_pred) + dice_loss(y_true, y_pred)

st.header("Callbacks")

with st.echo():
  # Callbacks
  K.clear_session()
  
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_folder,"weights_3.{epoch:02d}-{val_loss:.2f}.h5"),
      #save_weights_only=True,
      monitor='val_dice_coef',
      save_best_only=True)

  model_checkpoint_callback_2 = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_folder,"weights_3.{epoch:02d}-{val_loss:.2f}.h5"),
      save_weights_only=True,
      monitor='val_dice_coef',
     save_best_only=True)

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 2, min_delta =0.01)
  optimizer = tf.keras.optimizers.Adam(1e-6)

st.header("Setting up training")

with st.echo():
  data_generator_train = DataGenenerator(train_imgs, augmentation=albumentations_train)
  data_generator_val = DataGenenerator(test_imgs, shuffle=False)

st.header("Chargement du modèle")

st.code('''
model = Unet('resnet34', encoder_weights='imagenet' ,classes = 4, input_shape = (256, 416, 3), activation = 'sigmoid', encoder_freeze =True)
model.summary()
''', language = "python")

img = Image.open(path_base+"/model.png")
st.image(img)

st.code('''
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_folder,"weights_3.{epoch:02d}-{val_loss:.2f}.h5"),
    save_weights_only=True,
    monitor='val_dice_coef',
    save_best_only=True)
''', language = "python")

with st.echo():
  model.compile(optimizer = optimizer, loss = masked_bce_dice_loss, metrics = [dice_coef, bce_dice_loss, 'acc']) 

model.load_weights(os.path.join(checkpoint_folder,'last_wights_aug.h5'))

st.code('''
history = model.fit(data_generator_train, steps_per_epoch=data_generator_train.__len__(),
                    callbacks= [model_checkpoint_callback ,reduce_lr],
                    epochs = 25,
                    workers = -1,
                    validation_data = data_generator_val,
                    validation_steps = data_generator_val.__len__())
''', language = "python")
plot1_img = Image.open(path_base+"/fit.png")
st.image(plot1_img)



st.code('''
model.save_weights(os.path.join(checkpoint_folder, "last_wights_aug.h5"))
''', language = "python")

st.code('''
plt.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
plt.plot(history.history['dice_coef'], label = 'dice_coef')
plt.legend();
''', language = "python")

plot1_img = Image.open(path_base+"/plot1.png")
st.image(plot1_img)

st.code('''
plt.plot( history.history['val_loss'], label = 'val_loss')
plt.plot(history.history['loss'], label = 'loss')
plt.plot( history.history['bce_dice_loss'],":" ,label = 'bce_dice_loss')
plt.plot(history.history['val_bce_dice_loss'],":", label = 'val_bce_dice_loss')
plt.legend();
''', language = "python")

plot2_img = Image.open(path_base+"/plot2.png")
st.image(plot2_img)


from skimage.color import label2rgb
from PIL import Image
import matplotlib.patches as mpatches

  # Fonction pour afficher les images avec leurs masques prédits
@st.cache
def visu_images_4classes_test(folder, model, height_graph = 15, width_graph = 10):

  plt.figure(figsize = (12, 12))

  pop_1 = mpatches.Patch(color='blue', label='Gravel', alpha = 0.2)
  pop_2 = mpatches.Patch(color='red', label='Fish', alpha = 0.2)
  pop_3 = mpatches.Patch(color='yellow', label='Flower', alpha = 0.2)
  pop_4 = mpatches.Patch(color='green', label='Sugar', alpha = 0.2)

  #On sélectionne nb_images dans le folder
  im_rep = random.sample(os.listdir(folder), 1)
  #im_rep = ['1b55e5b.jpg']
  path_img = folder + im_rep[0]
  path_img_gravel = train_annot_folder +'/' + im_rep[0].split('.')[0] + '_Gravel.png'
  path_img_fish = train_annot_folder + '/' + im_rep[0].split('.')[0] + '_Fish.png'
  path_img_flower = train_annot_folder + '/' + im_rep[0].split('.')[0] + '_Flower.png'
  path_img_sugar = train_annot_folder + '/' +im_rep[0].split('.')[0] + '_Sugar.png'
  #On met l'image aux bonnes dimensions pour la prédiction (1, 480, 320, 3)
  img = cv2.imread(path_img).reshape(1400, 2100, 3)
  img = cv2.resize(img, (input_width, input_height))
  print(im_rep)
  print(img.shape)
  # Redimensionnement de l'image pour prédiction
  img1 = img.reshape(-1, input_height, input_width, 3)
  # Redimensionnement de l'image pour comparaison avec le futur masque
  img2 = cv2.resize(img,(input_width, input_height))

  #On obtient un masque en shape (38400, 4)
  mask = model.predict(img1/255.0)

  masked_gravel_hm=mask[0,:,:,0].reshape(input_height,input_width)
  masked_fish_hm=mask[0,:,:,1].reshape(input_height,input_width)
  masked_flower_hm=mask[0,:,:,2].reshape(input_height,input_width)
  masked_sugar_hm=mask[0,:,:,3].reshape(input_height,input_width)

  #On crée les masques de chaque formation
  ret, masked_gravel = cv2.threshold(mask[0,:,:,0].reshape(input_height,input_width), .7,1,cv2.THRESH_BINARY)
  ret, masked_fish =  cv2.threshold(mask[0,:,:,1].reshape(input_height,input_width), .7,1,cv2.THRESH_BINARY)
  ret, masked_flower =  cv2.threshold(mask[0,:,:,2].reshape(input_height,input_width), .7,1,cv2.THRESH_BINARY)
  ret, masked_sugar =  cv2.threshold(mask[0,:,:,3].reshape(input_height,input_width), .7,1,cv2.THRESH_BINARY)

  plt.subplot(4,4,1)
  plt.imshow(masked_gravel, label = 'gravel')
  plt.text(5, 5, 'Gravel', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])
  plt.subplot(4,4,2)
  plt.imshow(masked_fish, label='fish')
  plt.text(5, 5, 'Fish', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])
  plt.subplot(4,4,3)
  plt.imshow(masked_flower, label='flower')
  plt.text(5, 5, 'Flower', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])
  plt.subplot(4,4,4)
  plt.imshow(masked_sugar, label ='sugar')
  plt.text(5, 5, 'Sugar', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])

  plt.subplot(4,4,5)
  sns.heatmap(masked_gravel_hm, vmin=0, vmax=1)
  plt.text(5, 5, 'Gravel', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])
  plt.subplot(4,4,6)
  sns.heatmap(masked_fish_hm, vmin=0, vmax=1)
  plt.text(5, 5, 'Fish', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])
  plt.subplot(4,4,7)
  sns.heatmap(masked_flower_hm, vmin=0, vmax=1)
  plt.text(5, 5, 'Flower', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])
  plt.subplot(4,4,8)
  sns.heatmap(masked_sugar_hm, vmin=0, vmax=1)
  plt.text(5, 5, 'Sugar', bbox={'facecolor': 'white', 'pad': 10})
  plt.xticks([])
  plt.yticks([])

  #On crée une image de couleurs correspondant à chaque formation

  im_mask_gravel = label2rgb(image = img2,label = masked_gravel, colors = ['blue'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_mask_fish = label2rgb(image = img2, label = masked_fish, colors = ['red'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_mask_flower = label2rgb(image = img2, label = masked_flower, colors = ['yellow'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_mask_sugar = label2rgb(image = img2, label = masked_sugar, colors = ['green'], alpha = 0.2, kind = 'overlay', bg_label = 0)

  plt.subplot(4,4,9)
  plt.imshow(im_mask_gravel)
  plt.subplot(4,4,10)
  plt.imshow(im_mask_fish)
  plt.subplot(4,4,11)
  plt.imshow(im_mask_flower)
  plt.subplot(4,4,12)
  plt.imshow(im_mask_sugar)

  plt.subplot(4, 4, 13)
  annot_gravel = cv2.imread(path_img_gravel, cv2.IMREAD_GRAYSCALE)
  plt.imshow(annot_gravel)
  plt.subplot(4, 4, 14)
  annot_fish = cv2.imread(path_img_fish, cv2.IMREAD_GRAYSCALE)
  plt.imshow(annot_fish)
  plt.subplot(4, 4, 15)
  annot_flower = cv2.imread(path_img_flower, cv2.IMREAD_GRAYSCALE)
  plt.imshow(annot_flower)
  plt.subplot(4, 4, 16)
  annot_sugar = cv2.imread(path_img_sugar, cv2.IMREAD_GRAYSCALE)
  plt.imshow(annot_sugar)

  plt.show();

st.code('''
folder = train_imgs_folder + '/'
visu_images_4classes_test(folder, model, height_graph = 15, width_graph = 10)
''', language = "python")

folder = train_imgs_folder + '/'

if st.button("Visualiser une image"):
  st.pyplot(visu_images_4classes_test(folder, model, height_graph = 15, width_graph = 10))
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.echo():
  def make_prediction(img, model, threshold = 0.7, lower_bound = 3000,  prescaled = False):
    if not prescaled:
        img = img/255.0
    mask = model.predict(img)
    #print("Prediction done")
    output_height = model.output.get_shape()[1]
    output_width = model.output.get_shape()[2]
    mask_gravel = post_process(mask[0,:,:,0], output_height, output_width, threshold, lower_bound)
    mask_fish = post_process(mask[0,:,:,1], output_height, output_width, threshold, lower_bound)
    mask_flower = post_process(mask[0,:,:,2], output_height, output_width, threshold, lower_bound)
    mask_sugar = post_process(mask[0,:,:,3], output_height, output_width, threshold, lower_bound)
    # Unprocessed, post-processed
    masks_gravel = [mask[0,:,:,0], mask_gravel]
    masks_fish = [mask[0,:,:,1], mask_fish]
    masks_flower = [mask[0,:,:,2], mask_flower]
    masks_sugar = [mask[0,:,:,3], mask_sugar]
    return masks_gravel, masks_fish, masks_flower, masks_sugar

with st.echo():
  def post_process(mask, height, width, threshold, lower_bound):
    mask = mask.reshape(height, width)
    _, thresh = cv2.threshold(mask, threshold,1,cv2.THRESH_BINARY)
    area = cv2.moments(thresh, True)['m00']
    print("Total Area : {}".format(area), '\n')
    # Contours
    thresh_8bit = np.uint8(thresh * 255)
    contours, _ = cv2.findContours(thresh_8bit, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_areas = []
    for contour in contours:
        area_c = cv2.contourArea(contour)
        contours_areas.append(area_c)
        print("Contour area : {}".format(area_c), '\n')
    # Aire maximum
    if contours_areas: # Si la liste des contours n'est pas vide
        max_area = np.max(np.array(contours_areas))
        if max_area < lower_bound: # Si la liste des contours a au moins un contour suffisamment grand
            mask = np.zeros_like(mask)
        else:
            mask = thresh
    else:
        mask = np.zeros_like(mask)
    return mask


@st.cache(suppress_st_warning=True)
def demo(folder, model,height_graph = 15, width_graph = 10):
  #plt.figure(figsize = (12, 12))
  pop_1 = mpatches.Patch(color='blue', label='Gravel', alpha = 0.2)
  pop_2 = mpatches.Patch(color='red', label='Fish', alpha = 0.2)
  pop_3 = mpatches.Patch(color='yellow', label='Flower', alpha = 0.2)
  pop_4 = mpatches.Patch(color='green', label='Sugar', alpha = 0.2)
  im_rep = random.sample(os.listdir(folder), 1)
  #im_rep = ['4a7b6e3.jpg']
  #path_img = folder + im1
  path_img = folder + '/' + im_rep[0]
  path_img_gravel = train_annot_folder +'/' + im_rep[0].split('.')[0] + '_Gravel.png'
  path_img_fish = train_annot_folder + '/' + im_rep[0].split('.')[0] + '_Fish.png'
  path_img_flower = train_annot_folder + '/' + im_rep[0].split('.')[0] + '_Flower.png'
  path_img_sugar = train_annot_folder + '/' +im_rep[0].split('.')[0] + '_Sugar.png'
  #On met l'image aux bonnes dimensions pour la prédiction (1, 480, 320, 3)
  img = cv2.imread(path_img).reshape(1400, 2100, 3)
  img = cv2.resize(img, (input_width, input_height))
  print("nom du fichier:",im_rep[0])
  # Redimensionnement de l'image pour prédiction
  img1 = img.reshape(-1, input_height, input_width, 3)
  # Redimensionnement de l'image pour comparaison avec le futur masque
  img2 = cv2.resize(img,(input_width, input_height))
  #On obtient un masque en shape (38400, 4)
  masks_gravel, masks_fish, masks_flower, masks_sugar = make_prediction(img1, model, lower_bound=3000)
  im_mask_gravel = label2rgb(image = img2,label = masks_gravel[1], colors = ['blue'], alpha = 0.3, kind = 'overlay', bg_label = 0)
  im_mask_fish = label2rgb(image = img2, label = masks_fish[1], colors = ['red'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_mask_flower = label2rgb(image = img2, label = masks_flower[1], colors = ['yellow'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_mask_sugar = label2rgb(image = img2, label = masks_sugar[1], colors = ['green'], alpha = 0.4, kind = 'overlay', bg_label = 0)
  annot_gravel = np.expand_dims(cv2.resize(cv2.imread(path_img_gravel, cv2.IMREAD_GRAYSCALE).astype(np.int32),(output_width, output_height), interpolation =0).astype(np.int32) ,2)
  annot_fish = np.expand_dims(cv2.resize(cv2.imread(path_img_fish, cv2.IMREAD_GRAYSCALE).astype(np.int32),(output_width, output_height), interpolation =0).astype(np.int32) ,2)
  annot_flower = np.expand_dims(cv2.resize(cv2.imread(path_img_flower, cv2.IMREAD_GRAYSCALE).astype(np.int32),(output_width, output_height), interpolation =0).astype(np.int32) ,2)
  annot_sugar = np.expand_dims(cv2.resize(cv2.imread(path_img_sugar, cv2.IMREAD_GRAYSCALE).astype(np.int32),(output_width, output_height), interpolation =0).astype(np.int32) ,2)
  annot_labels = np.concatenate((annot_gravel, annot_fish, annot_flower, annot_sugar), axis = 2)
  masked_gravel2=annot_labels[:,:,0].reshape(input_height,input_width)
  masked_fish2=annot_labels[:,:,1].reshape(input_height,input_width)
  masked_flower2=annot_labels[:,:,2].reshape(input_height,input_width)
  masked_sugar2=annot_labels[:,:,3].reshape(input_height,input_width)
  im_annot_gravel = label2rgb(image = img2,label = masked_gravel2, colors = ['blue'], alpha = 0.3, kind = 'overlay', bg_label = 0)
  im_annot_fish = label2rgb(image = img2,label = masked_fish2, colors = ['red'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_annot_flower = label2rgb(image = img2,label = masked_flower2, colors = ['yellow'], alpha = 0.2, kind = 'overlay', bg_label = 0)
  im_annot_sugar = label2rgb(image = img2,label = masked_sugar2, colors = ['green'], alpha = 0.4, kind = 'overlay', bg_label = 0)
  fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,figsize=(20 ,10))
  grid = plt.GridSpec(3, 4, wspace=0.2, hspace=0.5)
  ax1 = plt.subplot(grid[0, 0])
  plt.imshow(im_mask_gravel)
  ax2 = plt.subplot(grid[0, 1])
  plt.imshow(im_mask_fish)
  ax3 = plt.subplot(grid[0, 2])
  plt.imshow(im_mask_flower)
  ax4 = plt.subplot(grid[0, 3])
  plt.imshow(im_mask_sugar)
  ax5 = plt.subplot(grid[1, 0])
  plt.imshow(im_annot_gravel)
  ax6 = plt.subplot(grid[1, 1])
  plt.imshow(im_annot_fish)
  ax7 = plt.subplot(grid[1, 2])
  plt.imshow(im_annot_flower)
  ax8 = plt.subplot(grid[1, 3])
  plt.imshow(im_annot_sugar)
  dice_coef_gravel = dice_coef(tf.cast(masks_gravel[1], tf.float32), tf.cast(annot_gravel, tf.float32)).numpy()
  dice_coef_fish = dice_coef(tf.cast(masks_fish[1], tf.float32), tf.cast(annot_fish, tf.float32)).numpy()
  dice_coef_flower = dice_coef(tf.cast(masks_flower[1], tf.float32), tf.cast(annot_flower, tf.float32)).numpy()
  dice_coef_sugar = dice_coef(tf.cast(masks_sugar[1], tf.float32), tf.cast(annot_sugar, tf.float32)).numpy()
  ax1.title.set_text('Prediction Gravel - Dice coef:{0:.2f}'.format(dice_coef_gravel))
  ax2.title.set_text('Prediction Fish - Dice coef:{0:.2f}'.format(dice_coef_fish))
  ax3.title.set_text('Prediction Flower - Dice coef:{0:.2f}'.format(dice_coef_flower))
  ax4.title.set_text('Prediction Sugar - Dice coef:{0:.2f}'.format(dice_coef_sugar))
  ax5.title.set_text('Gravel')
  ax6.title.set_text('Fish')
  ax7.title.set_text('Flower')
  ax8.title.set_text('Sugar')
  for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
      ax.axis("off")

  #return masks_gravel, masks_fish, masks_flower, masks_sugar
  print("Mean Dice: {0:.2f}".format(statistics.mean([dice_coef_fish, dice_coef_flower, dice_coef_gravel, dice_coef_sugar])))
  #st.write("Global dice : {0:.2f}".format(dice_coef((tf.cast(model.predict(img1), tf.float32)), (tf.cast(annot_labels, tf.float32)))))
  plt.show()

if st.button("Visualisation train"):
  st.pyplot(demo(train_imgs_folder, model))
  st.balloons()
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(suppress_st_warning=True)
def visu_demo_test(folder, model,height_graph = 15, width_graph = 10):
    #plt.figure(figsize = (12, 12))
    pop_1 = mpatches.Patch(color='blue', label='Gravel', alpha = 0.2)
    pop_2 = mpatches.Patch(color='red', label='Fish', alpha = 0.2)
    pop_3 = mpatches.Patch(color='yellow', label='Flower', alpha = 0.2)
    pop_4 = mpatches.Patch(color='green', label='Sugar', alpha = 0.2)
    
    im_rep = random.sample(os.listdir(folder), 1)

    path_img = folder + '/' + im_rep[0]

    #On met l'image aux bonnes dimensions pour la prédiction (1, 480, 320, 3)
    img = cv2.imread(path_img).reshape(1400, 2100, 3)
    img = cv2.resize(img, (input_width, input_height))

    # Redimensionnement de l'image pour prédiction
    img1 = img.reshape(-1, input_height, input_width, 3)
    # Redimensionnement de l'image pour comparaison avec le futur masque
    img2 = cv2.resize(img,(input_width, input_height))
    #On obtient un masque en shape (38400, 4)
    masks_gravel, masks_fish, masks_flower, masks_sugar = make_prediction(img1, model, lower_bound=3000)
    

    
    im_mask_gravel = label2rgb(image = img2,label = masks_gravel[1], colors = ['blue'], alpha = 0.3, kind = 'overlay', bg_label = 0)
    im_mask_fish = label2rgb(image = img2, label = masks_fish[1], colors = ['red'], alpha = 0.2, kind = 'overlay', bg_label = 0)
    im_mask_flower = label2rgb(image = img2, label = masks_flower[1], colors = ['yellow'], alpha = 0.2, kind = 'overlay', bg_label = 0)
    im_mask_sugar = label2rgb(image = img2, label = masks_sugar[1], colors = ['green'], alpha = 0.4, kind = 'overlay', bg_label = 0)

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(nrows=1, ncols=4,figsize=(width_graph, height_graph/2))
    
    grid = plt.GridSpec(2, 4, wspace=0.1, hspace=0.1)
  
    ax1 = plt.subplot(grid[0, 0])
    plt.imshow(im_mask_gravel)
    ax2 = plt.subplot(grid[0, 1])
    plt.imshow(im_mask_fish)
    ax3 = plt.subplot(grid[0, 2])
    plt.imshow(im_mask_flower)
    ax4 = plt.subplot(grid[0, 3])
    plt.imshow(im_mask_sugar)

    ax1.title.set_text('Prediction Gravel')
    ax2.title.set_text('Prediction Fish')
    ax3.title.set_text('Prediction Flower')
    ax4.title.set_text('Prediction Sugar')

    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis("off")

    plt.show()

if st.button("Visualisation test"):
  st.pyplot(visu_demo_test(test_imgs_folder, model))
  st.balloons()
st.set_option('deprecation.showPyplotGlobalUse', False)
