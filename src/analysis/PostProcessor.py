from src.util.Filenames import extract_score, extract_hash, remove_path, md5hash, strip_path_modifier
import os

from PIL import Image, ImageOps
import src.util.ImageLoader as ImageLoader
from pathlib import Path
import glob
import shutil
import src.util.Files as Files
import src.util.Filenames as Filenames


def copy_files(path, newpath):
    Files.mkdir(newpath)
    p = strip_path_modifier(path)
    for f in glob.glob(path):
        f = remove_path(f)
        shutil.copyfile(
            src=os.path.join(p, f),
            dst=os.path.join(newpath, f)
        )


def do_create_backup(path):
    stripped = strip_path_modifier(path)
    new_path = stripped + '_backup'
    copy_files(path, new_path)
    return new_path


def do_remove(orig_name, hashed, filenames, hashes, path, do_delete=False):
    path = strip_path_modifier(path)
    try:
        i = hashes.index(hashed)
        f = filenames[i]
        if do_delete:
            os.remove(os.path.join(path, f))
    except ValueError:
        print("Could not find file", hashed, orig_name)
    else:
        print("Removed", orig_name, f, do_delete)


def remove_from_folder(orig_names, pred_names, detected_images_path, limit, create_backup=True, purge=False):
    if create_backup:
        detected_images_path = do_create_backup(detected_images_path)
        print("Created backup in dir", detected_images_path)

    filenames = glob.glob(detected_images_path)
    filenames = [remove_path(f) for f in filenames]

    hashes = [md5hash(f) for f in filenames]

    for o, p in zip(orig_names, pred_names):
        score = extract_score(p)
        try:
            hashed = extract_hash(o)
        except:
            print("Could not extract hash", o)

        if score > limit:
            do_remove(o, hashed, filenames, hashes, detected_images_path, do_delete=purge)
