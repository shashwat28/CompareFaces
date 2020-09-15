"""
This module is for the Extraction of the Human Faces from a Original Images
"""
import os
import cv2


def scaled_image(img):
    '''method to determine and return list of coordinates of all frontal faces
    in the image'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


def extract_bounded_faces_from_dir(image_dir: str):
    '''Iterate through the Images of the directory'''
    for image_name in os.listdir(image_dir):
        input_path = os.path.join(image_dir, image_name)
        extract_bounded_faces(input_path, image_name)


def extract_bounded_faces(input_path: str, image_name: str):
    '''Takes path of the image and exports the image with Image_name__X'''
    image_to_scale = cv2.imread(input_path)
    scaled_img = scaled_image(image_to_scale)
    extract_bounded_faces_from_image(
        image_to_scale, scaled_img, image_name, FACE_CUTOUT_DIR)


def extract_bounded_faces_from_image(img, faces: list, image_path: str, processed_dir: str):
    '''Cropping out the faces from the subject image using coordinates from
    the list returned of the scaled_image method'''

    if(image_path is None or len(image_path) == 0):
        image_path = "img"

    name_counter = 0
    for (center_x, center_y, width, height) in faces:
        face_cutout = img[center_y:center_y + height,
                          center_x:center_x + width]
        cv2.imwrite(os.path.join(processed_dir,
                                 image_path + '__{}.jpg'.format(name_counter)),
                    face_cutout)
        name_counter = name_counter+1
    cv2.waitKey(0)


if __name__ == "__main__":

    # giving path to the data source directory
    PROJECT_PATH = "./../../"
    MODEL_PATH = "model/face_detection/haarcascade_frontalface_default.xml"
    IMAGE_DIR = PROJECT_PATH + "data/raw/images"
    FACE_CUTOUT_DIR = PROJECT_PATH + "data/processed/images"

    # loading the haar cascade pretrained classifier
    face_cascade = cv2.CascadeClassifier(PROJECT_PATH + MODEL_PATH)

    extract_bounded_faces_from_dir(IMAGE_DIR)
