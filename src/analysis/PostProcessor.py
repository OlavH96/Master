from src.analysis.PredictionAnalyser import extract_score
import os

from PIL import Image, ImageOps
import src.util.ImageLoader as ImageLoader

def remove_from_folder(originals, orig_names, pred_names, detected_images_path):
    filenames = os.listdir(detected_images_path)
    print(len(filenames))
    print(filenames[0])
    for o, p in zip(orig_names, pred_names):
        print(o,p)

