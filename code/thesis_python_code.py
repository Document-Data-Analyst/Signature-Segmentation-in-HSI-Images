import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pysptools
from PIL import Image

import rarfile
import os
import shutil
import zipfile
import glob

from spectral import *
import spectral.io.envi as envi

from pysptools import *
import pysptools.util as util
import pysptools.noise as noise
import pysptools.eea as eea
import pysptools.abundance_maps as amp

from skimage import data, measure, img_as_float
import skimage.exposure
from skimage.morphology import skeletonize, medial_axis, thin, skeletonize_3d
from skimage.util import invert
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from image_similarity_measures.quality_metrics import sam


#-------------- lOADING DATA --------------
#HSI data cube can be loaded through one of the following three methods, depending upon the type of HSI data file 

#Load HSI data form .hdr & .bil files 
def load_data_hdr_bil(hdr_filepath, bil_filepath):
  # hsi_signature = envi.open('/content/drive/MyDrive/HSI FILES/3.bil.hdr', '/content/drive/MyDrive/HSI FILES/3.bil')
  hsi_signature = envi.open(hdr_filepath, bil_filepath)
  hsi_signature_nparr = np.array(hsi_signature.load())
  print(hsi_signature_nparr.shape)
  # img = imshow(file)
  # img_arr = []
  hsi_arr = hsi_signature_nparr.shape
  imshow(hsi_signature_nparr)

  for i in range(hsi_arr[2]):
    if i == 100:
      hsi_01 = hsi_signature_nparr[:,:,i]
      plt.imsave(f'hsi_image_{i}.jpg', hsi_01, cmap='gray')
      plt.close()

  return hsi_signature_nparr, hsi_arr
hsi_signature_nparr, hsi_arr = load_data_hdr_bil('/content/drive/MyDrive/data/HSI FILES/3.bil.hdr', '/content/drive/MyDrive/data/HSI FILES/3.bil')

#Loading data numpy arrays 
def load_data_numpy(npy_filepath):
  np_file = np.load(npy_filepath)
  np_file = np_file.transpose(1,2,0)
  # img_arr = []
  np_arr = np_file.shape

  for i in range(np_arr[2]):
      if i == 20:
          np_01 = np_file[:, :, i]
          # Save the image as a JPG file
          plt.imsave(f'np_image_{i}.jpg', np_01, cmap='gray')
          # Close the figure to release memory (optional)
          plt.close()

  npy_filepath_arr = npy_filepath.split('/')
  name = npy_filepath_arr[len(npy_filepath_arr)-1]
  name = name.split('.') 
  name = name[0]
  print(name)

  return np_file, np_arr, name

np_file, np_arr, name = load_data_numpy('/content/drive/MyDrive/data/25Bg3.npy')

# #Load data from .zip file
def load_data_zip(zip_filepath, extract_folderpath):
  local_zip = zip_filepath
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall(extract_folderpath)
  zip_ref.close()

  img1_path = "/content/25Bg3/content/25Bg3/channel_30.jpg"
  img2_path = "/content/25Bg3/content/25Bg3/channel_180.jpg"
  return img1_path, img2_path
img1_path, img2_path = load_data_zip('/content/drive/MyDrive/data/25Bg3.zip', '/content/25Bg3')

#Save individual channel images 
def ind_channel_imgs(npy_file, sample_name):
  # ch_zip_file = np.load('/content/drive/MyDrive/data/25Bg9.npy')
  # ch_zip_file = ch_zip_file.transpose(1,2,0)
  ch_zip_file = np.array(npy_file)
  print(ch_zip_file.shape)

  # os.mkdir("individualChannelsTest")
  if not os.path.exists('/content/' + sample_name):
    # If it doesn't exist, create the directory
    os.makedirs(sample_name)
    print(f"Directory /content/'{sample_name}' created successfully.")
  else:
    print(f"Directory /content/'{sample_name}' already exists. Continuing...")
  # os.mkdir(sample_name)

  for i in range(ch_zip_file.shape[2]):
      print(i)
      # figName = "/content/individualChannelsTest/channel_" + str(i) + ".jpg"
      figName = "/content/" + str(sample_name) + "/channel_" + str(i) + ".jpg"
      plt.imsave(figName, ch_zip_file[:,:,i], cmap = "gray")
  command = 'zip -r ' + str(sample_name) + '.zip /content/' + str(sample_name)
  os.system(command)
  return print('Individual channels saved successfully!')
ind_channel_imgs(np_file, name)

def load_gt(gt_filepath, gt_extract_path, gt_filename):
  zip_ref = zipfile.ZipFile(gt_filepath, 'r')
  # zip_ref.extractall('/content/truths')
  zip_ref.extractall(gt_extract_path)
  zip_ref.close()
  gt_path = str(gt_extract_path) + '/truths/' + str(gt_filename) + '.png'
  # truth = cv2.imread('/content/drive/MyDrive/data/truths/truths/25B_g_9.png')
  truth = cv2.imread(gt_path)
  # plt.subplots(figsize=(7, 7))
  plt.imsave(f'GT.jpg', truth, cmap='gray')
  # plt.imshow(truth)
  return truth, gt_path
truth, gt_path = load_gt('/content/drive/MyDrive/data/truths.zip', '/content/truths', '25B_g_3')

#-------------- PRE-PROCESSING --------------
def get_images(img1_path, img2_path):
  img1 = cv2.imread(img1_path)
  img2 = cv2.imread(img2_path)

  plt.imsave(f'PP_01.jpg', img1, cmap='gray')
  plt.imsave(f'PP_02.jpg', img2, cmap='gray')

  return img1, img2
img1, img2 = get_images('/content/'+name+'/channel_30.jpg','/content/'+name+'/channel_180.jpg')

def denoise(img1, img2):
  dn_img1 = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 5, 15)
  dn_img2 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 5, 7)

  plt.imsave(f'DN_01.jpg', dn_img1, cmap='gray')
  plt.imsave(f'DN_02.jpg', dn_img2, cmap='gray')

  return dn_img1, dn_img2
dn_img1, dn_img2 = denoise(img1, img2)

def im_bw(dn_img1, dn_img2):
  thresh1 = 100         #100
  im_bw1 = cv2.threshold(dn_img1, thresh1, 255, cv2.THRESH_BINARY)[1]
  thresh2 = 140         #140
  im_bw2 = cv2.threshold(dn_img2, thresh2, 255, cv2.THRESH_BINARY)[1]

  plt.imsave(f'BN_01.jpg', im_bw1, cmap='gray')
  plt.imsave(f'BN_02.jpg', im_bw2, cmap='gray')
  return im_bw1, im_bw2
im_bw1, im_bw2 = im_bw(dn_img1, dn_img2)

def overlapping_region(im1, im2):
  img4 = cv2.bitwise_and(cv2.bitwise_not(im1) , cv2.bitwise_not(im2))

  plt.figure(figsize = (5, 5))
  plt.imshow(img4)

  return img4

def printed_text_fun(im_bw2):
  totalPixels = im_bw2[:,:,0].shape[0] * im_bw2[:,:,0].shape[1]
  colorPixels = np.sum(im_bw2[:, :, 0]) / 255

  perPixels = colorPixels / totalPixels
  # print("Percentage of white pixels:", perPixels*100)
  print("Percentage of printed text:", (1-perPixels)*100)

def signature_percentage(im1, im2):
  img3 = im1 - im2
  invt_img3 = cv2.bitwise_not(img3)
  dn_img3 = cv2.fastNlMeansDenoisingColored(invt_img3, None, 55, 55, 5, 55)

  plt.imsave(f'img3.jpg', img3, cmap='gray')
  plt.imsave(f'invt_img3.jpg', invt_img3, cmap='gray')
  plt.imsave(f'dn_img3.jpg', dn_img3, cmap='gray')

  totalPixels_img3 = dn_img3[:,:,0].shape[0] * dn_img3[:,:,0].shape[1]
  colorPixels_img3 = np.sum(dn_img3[:, :, 0]) / 255
  perPixels_img3 = colorPixels_img3 / totalPixels_img3
  # print("Percentage of white pixels:", perPixels_img3*100)
  print("Percentage of signature:", (1-perPixels_img3)*100)
  return dn_img3

#-------------- PRINTED TEXT --------------
def printedtext(im_bw2, im_bw1):
  printed_text = cv2.cvtColor(im_bw2, cv2.COLOR_RGB2GRAY)
  plt.imsave(f'PT_color.jpg', printed_text, cmap='gray')

  Printed_text_copy = im_bw2.copy()
  gray = cv2.cvtColor(im_bw2, cv2.COLOR_BGR2GRAY)

  # Set threshold level
  threshold_level = 50

  # Find coordinates of all pixels below threshold
  printed_pixels_coords = np.column_stack(np.where(gray < threshold_level))
  printed_pixels_coords_list = printed_pixels_coords.tolist()
  print(len(printed_pixels_coords_list))

  # Create mask of all pixels lower than threshold level
  mask = gray < threshold_level

  # Color the pixels in the mask
  Printed_text_copy = im_bw2.copy()
  Printed_text_copy[mask] = (204, 119, 0)
  plt.imsave(f'PT_mask.jpg', Printed_text_copy, cmap='gray')

  printed_text_fun(im_bw2)

  dn_img3 = signature_percentage(im_bw2 , im_bw1)
  return printed_text, Printed_text_copy, dn_img3
_, _, dn_img3 = printedtext(im_bw2, im_bw1)

#-------------- HANDWRITTEN TEXT --------------
def handwritten_text(dn_img3):
  imagedn3 = cv2.cvtColor(dn_img3, cv2.COLOR_RGB2GRAY)
  plt.imsave(f'imagedn3.jpg', imagedn3, cmap='gray')

  imagedn3_copy = dn_img3.copy()
  gray = cv2.cvtColor(dn_img3, cv2.COLOR_BGR2GRAY)

  # Set threshold level
  threshold_level = 242

  # Find coordinates of all pixels below threshold
  handwritten_pixels_coords = np.column_stack(np.where(gray < threshold_level))
  handwritten_pixels_coords_list = handwritten_pixels_coords.tolist()
  print(len(handwritten_pixels_coords_list))

  # Create mask of all pixels lower than threshold level
  mask = gray < threshold_level

  # Color the pixels in the mask
  imagedn3_copy = dn_img3.copy()
  imagedn3_copy[mask] = (204, 119, 0)
  plt.imsave(f'HP_mask.jpg', imagedn3_copy, cmap='gray')

  # # load image
  original_img = imagedn3_copy
  # convert to gray
  gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
  # threshold
  thresh = cv2.threshold(gray, 200, 250, cv2.THRESH_BINARY)[1]
  # blur threshold image
  blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=2, sigmaY=1, borderType = cv2.BORDER_DEFAULT)
  # stretch so that 255 -> 255 and 127.5 -> 0
  stretch = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)
  # threshold again
  thresh2 = cv2.threshold(stretch, 128, 255, cv2.THRESH_BINARY)[1]

  kernel = np.ones((3,3),np.uint8)
  erosion = cv2.erode(thresh2,kernel,iterations = 3)

  img_masked = cv2.bitwise_and(original_img, original_img, mask=erosion)

  plt.imsave(f'HP_PP_01.jpg', img_masked, cmap='gray')

  kernel = np.ones((3,3),np.uint8)
  dilate = cv2.dilate(img_masked,kernel,iterations = 1)
  # plt.imshow(dilate, 'gray')
  plt.imsave(f'HP_PP_02.jpg', dilate, cmap='gray')
  return dilate, img_masked
dilate, img_masked = handwritten_text(dn_img3)

def contours_colored(dilate):
  #colored contours
  # # Invert the image
  image = dilate.copy()
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  contours = measure.find_contours(image, 0.8, fully_connected='high')
  contour = sorted(contours, key=lambda x: len(x))[-1]

  fig, ax = plt.subplots()
  ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
  for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

  ax.xaxis.label.set_color('black')        #setting up X-axis label color to yellow
  ax.yaxis.label.set_color('black')          #setting up Y-axis label color to blue

  ax.tick_params(axis='x', colors='black')    #setting up X-axis tick color to red
  ax.tick_params(axis='y', colors='black')

  ax.axis('image')
  ax.set_xticks([])
  ax.set_yticks([])
  plt.show()
  fig.savefig('contours_colored.png', bbox_inches='tight')
  return fig
figure_colored = contours_colored(dilate)

def contours_black(dilate):
  #black contours
  # # Invert the image
  image = dilate.copy()
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  contours = measure.find_contours(image, 0.8, fully_connected='high')
  contour = sorted(contours, key=lambda x: len(x))[-1]

  fig, ax = plt.subplots()
  ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
  for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], 'k-' ,linewidth=2)

  ax.xaxis.label.set_color('black')        #setting up X-axis label color to yellow
  ax.yaxis.label.set_color('black')          #setting up Y-axis label color to blue

  ax.tick_params(axis='x', colors='black')    #setting up X-axis tick color to red
  ax.tick_params(axis='y', colors='black')

  ax.axis('image')
  ax.set_xticks([])
  ax.set_yticks([])
  plt.show()
  fig.savefig('contours_black.png', bbox_inches='tight')
  return fig
figure_black = contours_black(dilate)

#--------------------------------
def signature(img_masked, dn_img1):
  kernel = np.ones((5,5),np.uint8)
  dilate222 = cv2.dilate(img_masked,kernel,iterations = 1)
  # plt.imshow(dilate222, 'gray')

  signature222 = overlapping_region(dn_img1, dilate222)

  # signature = overlapping_region(dn_img1, dilate)
  imagem222 = cv2.bitwise_not(signature222)
  # plt.imshow(imagem222)

  imagedn3_copy = imagem222.copy()
  gray = cv2.cvtColor(imagem222, cv2.COLOR_BGR2GRAY)

  # Set threshold level
  threshold_level = 150

  # Find coordinates of all pixels below threshold
  handwritten_pixels_coords = np.column_stack(np.where(gray < threshold_level))
  handwritten_pixels_coords_list = handwritten_pixels_coords.tolist()
  print(len(handwritten_pixels_coords_list))

  # Create mask of all pixels lower than threshold level
  mask = gray < threshold_level

  # Color the pixels in the mask
  imagedn3_copy = imagem222.copy()
  imagedn3_copy[mask] = (204, 119, 0)

  plt.figure(figsize = (5, 5))
  plt.imsave(f'signature.jpg', imagem222, cmap='gray')
  return imagem222
  # plt.imshow(imagedn3_copy)
  # cv2.waitKey()
signature_result = signature(img_masked, dn_img1)


##-------------- EVALUATION --------------
def evaluate(signature_result, truth):
  gray_image_img1 = cv2.cvtColor(truth, cv2.COLOR_RGB2GRAY)
  gray_image_signature = cv2.cvtColor(signature_result, cv2.COLOR_RGB2GRAY)

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                          sharex=True, sharey=True)
  ax = axes.ravel()

  mse_none = mean_squared_error(gray_image_img1, gray_image_img1)/100
  ssim_none = ssim(gray_image_img1, gray_image_img1, multichannel=True)

  mse_noise = mean_squared_error(gray_image_img1, gray_image_signature)/100
  ssim_noise = ssim(gray_image_img1, gray_image_signature, multichannel=True)

  ax[0].imshow(gray_image_img1, cmap=plt.cm.gray)
  ax[0].set_xlabel(f'MSE: {mse_none:}, SSIM: {ssim_none}')
  ax[0].set_title('Original image')

  ax[1].imshow(gray_image_signature, cmap=plt.cm.gray)
  ax[1].set_xlabel(f'MSE: {mse_noise:}, SSIM: {ssim_noise:}')
  ax[1].set_title('Image with noise')

  plt.tight_layout()

  plt.show()
  fig.savefig('SSIM_MSE.png', bbox_inches='tight')

  out_sam = sam(gray_image_img1, gray_image_signature)/100
  print('Spectral Angle Mapping: ',out_sam)
  # out_psnr = psnr(in_img1, in_img2)

  #PYSPTOOLS SID FUNCTION
  def SID(s1, s2):
      p = (s1 / np.sum(s1)) + np.spacing(1)
      q = (s2 / np.sum(s2)) + np.spacing(1)
      return np.sum(p * np.log(p / q) + q * np.log(q / p))

  import pysptools
  sid = SID(gray_image_img1, gray_image_signature)
  print('Spectral Information Divergance: ',sid)
evaluate(signature_result, truth)
