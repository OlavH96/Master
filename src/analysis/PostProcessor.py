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


def do_create_backup(path, purge_overfitted=False):
    stripped = strip_path_modifier(path)
    new_path = stripped + '_backup'
    if purge_overfitted:
        new_path += '_no_overfitted'
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
        print("Could not find file", hashed, orig_name, filenames[0])
    else:
        print("Removed", orig_name, f, do_delete)


def remove_from_folder(orig_names, pred_names, detected_images_path, limit, limit_lower, create_backup=True, purge=False, purge_overfitted=False):
    if create_backup:
        detected_images_path = do_create_backup(detected_images_path, purge_overfitted)
        print("Created backup in dir", detected_images_path)
    print("path is", detected_images_path)
    filenames = glob.glob(detected_images_path)
    filenames = [remove_path(f) for f in filenames]
    
    hashes = [md5hash(f) for f in filenames]
    print("num files is", len(filenames))
    print("num orig, pred is", len(orig_names), len(pred_names))
    for o, p in zip(orig_names, pred_names):
        score = extract_score(p)
        try:
            hashed = extract_hash(o)
        except:
            print("Could not extract hash", o)
        else:
            if score > limit and purge:
                do_remove(o, hashed, filenames, hashes, detected_images_path, do_delete=purge)
            if score < limit_lower and purge_overfitted:
                do_remove(o, hashed, filenames, hashes, detected_images_path, do_delete=purge_overfitted)

