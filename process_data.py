import os
import xml.etree.ElementTree as et
from PIL import Image
import numpy as np
import environments

data_location = environments.data_location_path
annotations_dir_name = 'test_annotations'
images_dir_name = 'test_images'
annotations_dir = os.path.join(data_location, annotations_dir_name)
images_dir = os.path.join(data_location, images_dir_name)
size_of_processed_image = 224
normalized_image_width, normalized_image_height = (size_of_processed_image, size_of_processed_image)
images_data = {}


def preprocess_image(im_path, b_box = 'None'):
    if b_box == 'None':
        im = Image.open(im_path)
        im = im.convert('RGB')
        cropped = im
    else:
        im = Image.open(im_path)
        im = im.convert('RGB')
        xmin = b_box["xmin"]
        xmax = b_box["xmax"]
        ymin = b_box["ymin"]
        ymax = b_box["ymax"]
        cropped = im.crop((xmin, ymin, xmax, ymax))
    new_im = cropped.resize((normalized_image_width, normalized_image_height))
    arrayed = np.array(new_im)
    normalized = arrayed / 255.0
    reshaped = normalized.reshape(-1)
    return reshaped

def process_data():
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
                images_data[image_filename] = {'image_data': image, 'breed': breed, 'bounding_box': b_box}
            except Exception as e:
                print(f"Error with: {e}")