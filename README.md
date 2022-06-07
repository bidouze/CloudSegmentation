# CloudSegmentation

The full research paper can be found at this link [paper](https://github.com/bidouze/CloudSegmentation/blob/main/Research%20_paper.pdf)
## Project Deep Learning 2022 Ismail Benaija - Ayoub Bakkoury - Othmane Baddou ##


This project is inspired by a Kaggle competition initiated
by the Max Planck Institute of Meteorology. The goal of
the competition is to identify cloud formations on satellite
images. It is proposed to identify 4 different classes of cloud
formations :

~~~
  • Sugar
  • Flower
  • Fish
  • Gravel
 ~~~
These cloud formations play a determining role on the climate and are difficult to understand and to implement in
climate models. By classifying these cloud formations, researchers hope to better understand them and improve existing models.

## Content ##

You will find in this repository a streamlit presentation of the latest version of our code in the project. 

### Requirements ###

In order to install the required python packages run the following:

~~~
 pip install -r requirements.txt
 ~~~

### Add the data ###

You have to download the data from the kaggle competition https://www.kaggle.com/competitions/understanding_cloud_organization/data
Once you have downloaded it, add all the files on the data folder on this repository and you are set to go. Normally, you should run utils to create the annotated images but it's already given in this repository. Run the following cell on your terminal in order to launch the streamlit file.

~~~
streamlit run /path/Projet_Resnet.py
~~~


### What you can do in the streamlit ###

You can use the buttons in the streamlit Visualize an image, Visualize train, Visualize test to test the model on randomly drawn images from the dataset.

#### Notes ####
Please note that the model is not trained in the streamlit file for obvious reasons but if you go on the python file you can have access to commented lines highlighting the cells for training in case you want to try and train it differently.

