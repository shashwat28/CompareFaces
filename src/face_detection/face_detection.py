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


def extract_bounded_faces(img, faces, FACE_CUTOUT_DIR):
    '''cropping out the faces from the subject image using coordinates from
    the list returned of the scaled_image method'''
    name_counter = 0
    for (center_x, center_y, width, height) in faces:
        cv2.rectangle(img, (center_x, center_y),
                      (center_x + width, center_y+height),
                      (255, 0, 0), 2)
        face_cutout = img[center_y:center_y + height,
                          center_x:center_x + width]
        cv2.imwrite(os.path.join(FACE_CUTOUT_DIR,
                                 'img{}.jpg'.format(name_counter)),
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

    # iterate through the names of contents of the folder
    for image_path in os.listdir(IMAGE_DIR):
        input_path = os.path.join(IMAGE_DIR, image_path)
        image_to_scale = cv2.imread(input_path)
        scaled_img = scaled_image(image_to_scale)
        extract_bounded_faces(image_to_scale, scaled_img, FACE_CUTOUT_DIR)
