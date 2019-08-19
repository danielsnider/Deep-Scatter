#!/usr/bin/env ython
import os
import os.path
import csv
import numpy as np
from IPython import embed
#import pylab
import mahotas as mh
import joblib #used to save parameters of model
import umap #different algorithm than t-SNE
from openTSNE import TSNE as opentsne#different algorithm than t-SNE
#import watershed # label image by calling watershed.py
import utils # crop cell by calling utils.py
from tsne_lib import plot_helper
from PIL import Image
import skimage
import skimage.io
import scipy
import pandas as pd
import click
#import matplotlib.patches as mpatches
if os.name != 'nt':
  from tsne import bh_sne
from time import time
#from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tsne_lib import get_features_resnet152
from tsne_lib import get_features_vgg16
#from get_features_resnet152 import resnet
#from get_features_vgg16 import vgg16
#from get_features_inceptionv3 import inceptionv3
from skimage import data, color
from skimage.transform import rescale, resize

UMAP_filename = 'UMAP_parameters.sav' #IMPORTANT! This is the file where the parameters of the UMAP model are being saved
TSNE_filename = 'TSNE_parameters.sav' #IMPORTANT! This is the file where the parameters of the t-SNE model are being saved
openTSNE_filename = 'openTSNE_parameters.sav' #IMPORTANT! This is the file where the parameters of the OPEN t-SNE model are being saved
 
def tsne_images(session_id,colors_dict, res, perplexity, early_exaggeration, learning_rate, dpi, canvasSize,colour, model_name, algorithm_name):

  l = [c['images'] for c in colors_dict]  # pull out image names into list of lists
  filenames = [item for sublist in l for item in sublist] # flatten list

  if not filenames:
    return  # do nothing if there are no files to operate on
  total_res = res**2
  x_value = np.zeros((len(filenames),total_res)) # Dimension of the image: 70*70=4900; x_value will store images in 2d array
  count = 0
  images = []
  #--------------------------------------------------------------------------------------------------------------------
  for imageName in filenames: 
  #image = scipy.misc.imresize(skimage.io.imread(imageName), (res,res)) #reshape size to (70,70) for every image; 70 being the res
    image =  resize(skimage.io.imread(imageName), (res, res), anti_aliasing=True)
    if len(image.shape) != 2:
      image3d=image[:,:,:3]
      image2d=mh.colors.rgb2grey(image3d)
    else:
      image2d = image
      image3d = np.stack((image2d, image2d, image2d), axis=2)
    image1d = image2d.flatten() #image1d stores a 1d array for each image
    images.append(image3d)
    x_value[count,:] = image1d # add a row of values
    count += 1
  if model_name != 'None':
    if model_name == 'ResNet V2 152':
      x_value = get_features_resnet152.resnet(filenames= filenames, session_id=session_id, res=res,perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, dpi=dpi)
    if model_name == 'VGG 16':
      x_value = get_features_vgg16.vgg16(filenames=filenames, session_id=session_id, res=res,perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=0, dpi=dpi)
    if model_name == 'Inception V3':
      x_value = inceptionv3(filenames=filenames, session_id=session_id, res=res,perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, dpi=dpi)

  if algorithm_name == 'NEW UMAP (create new parameters)': #To use UMAP to analyze images and to save the UMAP parameters
    print("new UMAP")
    reducer = umap.UMAP()
    vis_data = reducer.fit_transform(x_value)
    joblib.dump(reducer, UMAP_filename)

  elif algorithm_name == 'APPLY UMAP (use set parameters)': #To use UMAP to analyze images and to use already saved parameters
    print("old UMAP")
    loaded_reducer = joblib.load(UMAP_filename)
    vis_data = loaded_reducer.transform(x_value)

  elif algorithm_name == 'NEW t-SNE (create new parameters)': #To use t-SNE to analyze images and to save the t-SNE parameters
    print("NEW TSNE")
    tsne = manifold.TSNE(init='pca', random_state=0, early_exaggeration=early_exaggeration, learning_rate=0,perplexity=perplexity)
    vis_data = tsne.fit_transform(x_value)
    joblib.dump(tsne, TSNE_filename)

  elif algorithm_name == 'APPLY t-SNE (use set parameters)': #To use t-SNE to analyze images and to use already saved parameters
    print("OLD TSNE")
    loaded_file = joblib.load(TSNE_filename)
    vis_data = loaded_file.fit_transform(x_value)

  elif algorithm_name == 'NEW open-t-SNE (create new parameters)': #To use t-SNE to analyze images and to save the t-SNE parameters
    print("NEW OPEN TSNE")
    open_tsne = opentsne()
    vis_data = open_tsne.fit(x_value)
    joblib.dump(open_tsne, openTSNE_filename)

  elif algorithm_name == 'APPLY open-t-SNE (use set parameters)': #To use t-SNE to analyze images and to use already saved parameters
    print("OLD OPEN TSNE")
    loaded_file = joblib.load(openTSNE_filename)
    vis_data = loaded_file.transform(x_value)
 

  canvas = plot_helper.image_scatter(vis_data[:, 0], vis_data[:, 1], images, colour,res, min_canvas_size=canvasSize )

  plt.imshow(canvas,origin='lower')
  #plt.axis('off')
  # save_location = 'static/output/%s/output.png' % session_id
  save_location = 'static/output/%s/ResultPlot_ModelName:%s__Resolution:%d_Perplexity:%d_EarlyExaggeration:%d_LearningRate:%d_DPI:%d.png' % (session_id,model_name, res,perplexity, early_exaggeration, learning_rate, dpi)

  xmin, xmax, ymin, ymax = plt.axis()
  s = '1xmin = ' + str(round(xmin, 2)) + ', ' + \
    'xmax = ' + str(xmax) + '\n' + \
    'ymin = ' + str(ymin) + ', ' + \
    'ymax = ' + str(ymax) + ' '
  print(s)

  plt.xlim([-0.5, 20000])
  plt.ylim([-0.5, 20000])

  xmin, xmax, ymin, ymax = plt.axis()
  s = '2xmin = ' + str(round(xmin, 2)) + ', ' + \
    'xmax = ' + str(xmax) + '\n' + \
    'ymin = ' + str(ymin) + ', ' + \
    'ymax = ' + str(ymax) + ' '
  print(s)
  print(vis_data)
  plt.savefig(save_location,dpi=dpi,pad_inches=1,bbox_inches='tight')
  print('Saved image scatter to %s' % save_location)

  #csv stuff
  colours_csv = [None] * len(filenames) #create empty list 
  for x in range (0, len(filenames)): #create a list of colours in the order of the files
    colours_csv[x] =filenames[x][52:58]

  filenames_csv = [None] * len(filenames) #create empty list 
  for x in range (0, len(filenames)): #create a list of filenames in the order of the files
    filenames_csv[x] =filenames[x][59:]

  csv= 'static/output/%s/ReducedFeatures.csv' % session_id # name the csv
  if os.path.isfile(csv):
    df = utils.read_csv(csv)
    df['File Name']=pd.Series (filenames_csv)
  else:
    df = pd.DataFrame (filenames_csv, columns=["File Name"]) 
  # df['FileName']=pd.Series (filenames)
  df['Colour (in Hex)']=pd.Series (colours_csv)
  df['Tsne_1']=pd.Series (vis_data[:, 0])
  df['Tsne_2']=pd.Series (vis_data[:, 1])
  df.to_csv(csv, index=False)

  return vis_data
