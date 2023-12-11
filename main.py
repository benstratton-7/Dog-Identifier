import os
import xml.etree.ElementTree as et
from PIL import Image
import numpy as np
import environments
# import sys
# np.set_printoptions(threshold=sys.maxsize)

data_location = environments.data_location_path
annotations_dir_name = 'test_annotations'
images_dir_name = 'test_images'
annotations_dir = os.path.join(data_location, annotations_dir_name)
images_dir = os.path.join(data_location, images_dir_name)

def preprocess_image(im_path, b_box):
    im = Image.open(im_path)
    xmin = b_box["xmin"]
    xmax = b_box["xmax"]
    ymin = b_box["ymin"]
    ymax = b_box["ymax"]
    cropped = im.crop((xmin, ymin, xmax, ymax))
    neww, newh = (224, 224)
    new_im = cropped.resize((neww, newh))
    return np.array(new_im)

# Initialize empty lists to store images and labels
images_data = {}

# Loop through each annotation file
for subject_dir in os.listdir(annotations_dir):
    subject_path = os.path.join(annotations_dir, subject_dir)
    for annotation_file in os.listdir(subject_path):
        annotation_path = os.path.join(subject_path, annotation_file)
        try:
            tree = et.parse(annotation_path)
            root = tree.getroot()

            image_filename = root.find('filename').text
            breed = root.find('.//object/name').text
            image_path = os.path.normpath(os.path.join(images_dir, subject_dir, image_filename + '.jpg'))
            xmin = int(root.find('.//object/bndbox/xmin').text)
            ymin = int(root.find('.//object/bndbox/ymin').text)
            xmax = int(root.find('.//object/bndbox/xmax').text)
            ymax = int(root.find('.//object/bndbox/ymax').text)
            b_box = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
            image = preprocess_image(image_path, b_box)

            images_data[image_filename] = {'breed': breed, 'image_path': image_path, 'image': image, 'bounding_box': b_box}
        except Exception as e:
            print(f"Error with: {e}")