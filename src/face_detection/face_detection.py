"""
This module is for the Extraction of the Human Faces from a Original Images
"""
import os
import cv2


def scaled_image(img, classifier) -> list:
    """Finds the location of human faces in a Images

    Args:
        img (OPEN CV Image): Input Image
        classifier (cv2.CascadeClassifier): Classifier that distinguish faces

    Returns:
        list: List of Coordinates of the found Faces
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    return faces


def extract_bounded_faces_from_dir(image_dir: str, **kwargs):
    """Iterate through a directory to find Images and Extract human faces from them

    Args:
        image_dir (str): Relative Path of directory contatining input images
        **export_path (str): Export Path for Generated Images,
        default = data/processed/images
        **model_path (str): Path of Face Classifier Model,
        default = model/face_detection/haarcascade_frontalface_default.xmls
        **image_classifier (cv2.CascadeClassifier): Face Classifier Model,
        default = cv2.CascadeClassifier(model_path)
    """
    model_path = kwargs.get(
        "model_path",
        "model/face_detection/haarcascade_frontalface_default.xml")
    classifier = kwargs.get(
        "image_classifier", cv2.CascadeClassifier(model_path))
    export_path = kwargs.get("export_path", "data/processed/images")

    for image_name in os.listdir(image_dir):
        input_path = os.path.join(image_dir, image_name)
        extract_bounded_faces(input_path, image_name,
                              export_path, image_classifier=classifier)


def extract_bounded_faces(
        input_path: str, result_prefix: str, export_path: str, **kwargs):
    """Takes the path for the input image and Extract human faces from them

    Args:
        input_path (str): path of the Input image
        result_prefix (str): Prefix for the extracted images
        export_path (str): Path of Directory where extracted images
         will be saved, default = data/processed/images
        **model_path (str): Path of Face Classifier Model,
        default = model/face_detection/haarcascade_frontalface_default.xmls
        **image_classifier (cv2.CascadeClassifier): Face Classifiers,
        default = cv2.CascadeClassifier(model_path)
    """
    model_path = kwargs.get(
        "model_path",
        "model/face_detection/haarcascade_frontalface_default.xml")
    classifier = kwargs.get(
        "image_classifier", cv2.CascadeClassifier(model_path))
    export_path = "data/processed/images" if (
        export_path is None and len(export_path) == 0) else export_path

    image_to_scale = cv2.imread(input_path)
    scaled_img = scaled_image(image_to_scale, classifier)
    crop_bounded_box_from_image(
        image_to_scale, scaled_img, result_prefix, export_path)


def crop_bounded_box_from_image(
        img, bounding_boxes: list, result_prefix: str, processed_dir: str):
    """Crop and Export a Image with given dimension

    Args:
        img (OPEN CV Image): Input Image
        bounding_boxes (list): List of Coordinates of the Bounding Boxes
        result_prefix (str): Prefix for the generated cropped images
        processed_dir (str): Path of directory where cropped images are stored
    """
    if(result_prefix is None or len(result_prefix) == 0):
        result_prefix = "img"

    name_counter = 0
    for (center_x, center_y, width, height) in bounding_boxes:
        face_cutout = img[center_y:center_y + height,
                          center_x:center_x + width]
        result_filename = create_nested_directories(
            processed_dir, ["BoundedFaces", result_prefix])
        result_filename = os.path.join(
            result_filename, "{}.jpg".format(name_counter))
        if not cv2.imwrite(result_filename, face_cutout):
            print(result_filename + " not saved")
        name_counter = name_counter+1


def create_nested_directories(parent_path: str, dir_name_list: list) -> str:
    """Create a series of Nested directories inside the parent_path

    Args:
        parent_path (str): Directory where generated directories
        will be created
        dir_name_list (list): List of directorie names where each
        directory is created inside another

    Returns:
        str: Path to innermost created directory
    """
    copy = parent_path + ""
    for dir_name in dir_name_list:
        if not create_directory(copy, dir_name):
            break
        copy = os.path.join(copy, dir_name)
    return copy


def create_directory(
        parent_path: str, dir_name: str, verbosity: bool = False) -> bool:
    """Creates a "dir_name" directory in "parent_path" directory,
    suppresses  FileExistsError and,
    return True, if required directory is generated/exists,
    otherwise return False

    Args:
        parent_path (str): Path to Target Directory,
        where new directory is needed to be created
        dir_name (str): Name of new directory
        verbosity (bool, optional): Prints error if Found, Defaults to False.

    Returns:
        bool: Returns True, if desired directory is created/exists,
        otherwise False
    """
    try:
        os.mkdir(os.path.join(parent_path, dir_name), mode=0o771)
    except FileExistsError:
        pass
    except FileNotFoundError:
        if verbosity:
            print("Target directory does not exist")
        return False
    return True


if __name__ == "__main__":
    PROJECT_PATH = ""
    MODEL_PATH = os.path.join(
        PROJECT_PATH,
        "model/face_detection/haarcascade_frontalface_default.xml")
    IMAGE_SOURCE_PATH = os.path.join(PROJECT_PATH, "data/raw/images")
    create_nested_directories(os.path.join(
        PROJECT_PATH, 'data'), ["processed", "images"])
    EXPORT_PATH = os.path.join(PROJECT_PATH, "data/processed/images")

    extract_bounded_faces_from_dir(IMAGE_SOURCE_PATH)
