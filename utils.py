import random
import cv2
import numpy as np
import tifffile as tiff
import earthpy.plot as ep
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

def get_rand_patch(img, mask, sz=160, channel = None):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :param  Channels  0: Buildings , 1: Roads & Tracks, 2: Trees , 3: Crops, 4: Water
    :return: patch with shape (sz, sz, num_channels)
    
    
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass
    if channel=='all':
        return patch_img, patch_mask
    
    if channel !='all':
        patch_mask = patch_mask[:,:,channel]
        return patch_img, patch_mask



def get_patches(x_dict, y_dict, n_patches, sz=160, channel = 'all'):
    """
    :param  Channels  0: Buildings , 1: Roads & Tracks, 2: Trees , 3: Crops, 4: Water or 'all'
    
    """
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz, channel)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)

def load_data(path = './data/'):
    """
    :param path: the path of the dataset which includes  mband and  gt_mband folders
    :return: X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_VALIDATION, Y_DICT_VALIDATION
    """
    trainIds = [str(i).zfill(2) for i in range(1, 25)]  # all availiable ids: from "01" to "24"

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in trainIds:

        img_m = normalize(tiff.imread(path + 'mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
        mask = tiff.imread(path + 'gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        #print(img_id + ' read')
    print('Images are read')
    return  X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_VALIDATION, Y_DICT_VALIDATION

def plot_train_data(X_DICT_TRAIN, Y_DICT_TRAIN, image_number = 12):
    
    labels =['Orginal Image with the 8 bands', 'Ground Truths: Buildings', 'Ground Truths: Roads & Tracks', 'Ground Truths: Trees' , 'Ground Truths: Crops', 'Ground Truths: Water']
    
    image_number = str(image_number).zfill(2)
    number_of_GTbands = Y_DICT_TRAIN[image_number].shape[2]
    f, axarr = plt.subplots(1, number_of_GTbands + 1, figsize=(25,25))

    band_indices = [0, 1, 2]
    print('Image shape is: ',X_DICT_TRAIN[image_number].shape)
    print("Ground Truth's shape is: ",Y_DICT_TRAIN[image_number].shape)

    ep.plot_rgb(X_DICT_TRAIN[image_number].transpose([2,0,1]),
                rgb=band_indices,
                title=labels[0],
                stretch=True,
                ax=axarr[0])
    
    for i in range(0, number_of_GTbands):
        axarr[i+1].imshow(Y_DICT_TRAIN[image_number][:,:,i])
        #print(labels[i+1])
        axarr[i+1].set_title(labels[i+1])

    plt.show()
    
    
def Abs_sobel_thresh(image,orient='x',thresh=(40,250) ,sobel_kernel=3):
    gray=image#cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if orient=='x':
        #the operator calculates the derivatives of the pixel values along the horizontal direction to make a filter.
        sobel=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize= sobel_kernel) 
    if (orient=='y'):
        sobel=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize= sobel_kernel)
    abs_sobel=np.absolute(sobel)
    scaled_sobel=(255*abs_sobel/np.max(abs_sobel))
    grad_binary=np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])]=1
    return grad_binary



def Mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray=image#cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)

    gradmag=np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag=np.uint8(gradmag/scale_factor )
    mag_binary=np.zeros_like(gradmag)
    mag_binary[(gradmag>=mag_thresh[0])&(gradmag<=mag_thresh[1])]=1
    # Apply threshold
    return mag_binary

def Dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray=image#cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)
    abs_graddir=np.arctan(abs_sobely,abs_sobelx)
    dir_binary=np.zeros_like(abs_graddir)
    dir_binary[(abs_graddir>=thresh[0])&(abs_graddir<=thresh[1])]=1
    # Calculate gradient direction
    # Apply threshold
    return dir_binary

def Combined_thresholds(gradx,grady,mag_binary,dir_binary):
    combined=np.zeros_like(dir_binary)
    combined[(gradx==1)|(grady==1) |(mag_binary==1)|(dir_binary==1)]=1
    return combined
    

def BilateralFilter(image, kernel_size,sigmaSpace,sigmaColor): # bilateral filter can keep edges sharp while removing noises
    img=np.copy(image)
    img=cv2.bilateralFilter(img,kernel_size,sigmaColor,sigmaSpace)
    #plt.imshow(img)
    return img


def Erosion(image, filter_size = 2, iteration= 1):
    img=np.copy(image)
    kernel = np.ones((filter_size,filter_size),np.uint8)
    erosion=cv2.erode(img,kernel,iterations=iteration)
    return erosion

def Opening(image, filter_size):
    #Opening is just another name of erosion followed by dilation
    img=np.copy(image)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(filter_size,filter_size))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening
    
    
    
def Closing(image,k):# closing is useful to detect the overall contour of a figure and opening is suitable to detect subpatterns. 
    kernel = np.ones((k, k), np.uint8)
    img=np.copy(image)
    img_close = cv2.morphologyEx(img, op= cv2.MORPH_CLOSE,kernel=kernel)
    return img_close

def Denoise(image,k):
    img=np.copy(image)
    struct=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,struct)
    return img

def Binary(image, threshold, max_value = 1):
    img=np.copy(image)
    (t,masklayer)=cv2.threshold(img,threshold,max_value,cv2.THRESH_BINARY)
    return masklayer

def Gaussian_filter(image, sigma =1):
    img=np.copy(image)
    blur = filters.gaussian(img, sigma=sigma)
    return blur

def Find_threshold_otsu(image):
    t = filters.threshold_otsu(image)
    return t


def ExtractObjects(image):
    img=np.copy(image)
    #kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    #erosion=cv2.erode(img,kernel,iterations=1)
    #bliteralfilter=cv2.bilateralFilter(erosion,5,75,75)
    #(t,masklayer)=cv2.threshold(bliteralfilter,0,1,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #denoising = Denoise(img,1)
    blob_labels=measure.label(img,background=0)
    number_of_objects=np.unique(blob_labels)
    return blob_labels,number_of_objects


def post_processing(img):
    
    blur = Gaussian_filter(img, sigma=1)
    t = Find_threshold_otsu(blur)
    binary_img = Binary(blur,t)
    opened_img  = Opening(binary_img, filter_size = 3)
    blob_labels,number_of_objects = ExtractObjects(opened_img)
    
    return opened_img, number_of_objects, blob_labels    