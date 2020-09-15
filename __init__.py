"""
Entry point for the Project
"""
from src.face_detection import face_detection

face_detection.extract_bounded_faces_from_dir("data/raw/images")
