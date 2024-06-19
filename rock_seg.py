import os, shutil, random
from glob import glob

from sklearn.model_selection import train_test_split

# function to split data into training and testing datasets
def split_data(rootdir, split=0.7, imgext='tif'):
    assert isinstance(split,float), ValueError('Training split is not a float. Check that the split value is a float greater than 0.0 but less than 1.0.')
    assert split > 0.0, ValueError('Training split less than or equal to 0.0. Check that the split value is a float greater than 0.0 but less than 1.0.')
    assert split < 1.0, ValueError('Training split is greater than or equal to 1.0. Check that the split value is a float greater than 0.0 but less than 1.0.')

    # set images and annotations directories
    imdir = os.path.join(rootdir, "images")
    andir = os.path.join(rootdir, "annotations")
    
    assert os.path.isdir(imdir), ValueError('Cannot find {}.\n Check that directory exists and re-run program'.format(imdir))
    assert os.path.isdir(andir), ValueError('Cannot find {}.\n Check that directory exists and re-run program'.format(andir))
    
    # get list of all image files in the images directory
    all_images = glob(os.path.join(imdir,"*."+imgext))
    
    # shuffle images list
    random.shuffle(all_images)

    # split images list to training/testing
    img_train,img_eval = train_test_split(all_images, test_size=(1-split))

    # convert image splits into annotation splits
    ann_train = [img.replace('images','annotations').replace('image_','annotation_').replace('tif','png') for img in img_train]
    ann_eval = [img.replace('images','annotations').replace('image_','annotation_').replace('tif','png') for img in img_eval]
    
    assert len(img_train)==len(ann_train), ValueError("{} != {}\n Number of TRAINING images is not equal to TRAINING segmentation masks".format(len(img_train), len(ann_train)))
    assert len(img_eval)==len(ann_eval), ValueError("{} != {}\n Number of EVALUATION images is not equal to EVALUATION segmentation masks".format(len(img_eval), len(ann_eval)))

    return imdir,andir,img_train,ann_train,img_eval,ann_eval