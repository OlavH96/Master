# TODO: Refactor to have name and image be tuple
import numpy as np

import src.util.ImageLoader as ImageLoader
import matplotlib.pyplot as plt
from pathlib import Path


def order_sorter():
    return lambda x: order(x[-1])


def score_sorter():
    return lambda x: extract_score(x[-1])


def remove_path(name):
    return name.split('/')[-1]


def order(name):
    """
    Extract index of image from my poor naming scheme
    :param name: name from which to extract index
    :return: index of name
    """
    if name.startswith('pred'):
        split = name.split('_')
        return int(split[-2])
    split = name.split('_')
    x = split[-1].split('.')[0]
    return int(x)


def arrange_files(images, names, sort_function):
    # https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w
    return zip(*sorted(zip(images, names), key=sort_function))


def sort_by_order(images, names):
    return arrange_files(images, names, order_sorter())


def align_sort(to_align, align_with, align_function):
    # NOTE: to_align must be sorted 0-n
    with_indexes = [align_function(x) for x in align_with]
    aligned = [to_align[i] for i in with_indexes]
    return aligned


def load_and_preprocess(path):
    origPath = str(path / 'orig*')
    predPath = str(path / 'pred*')

    originals, orig_names = ImageLoader.load_images(origPath)
    predictions, pred_names = ImageLoader.load_images(predPath)

    orig_names = [remove_path(n) for n in orig_names]
    pred_names = [remove_path(n) for n in pred_names]

    originals, orig_names = sort_by_order(originals, orig_names)
    predictions, pred_names = sort_by_order(predictions, pred_names)

    return originals, orig_names, predictions, pred_names


def extract_score(pred):
    split = pred.split('_')
    n = split[-1].split('.')
    return float('.'.join(n[:2]))


def sort_by_score(originals, orig_names, predictions, pred_names, highest_first=True):
    sorted_predictions, sorted_pred_names = arrange_files(predictions, pred_names, score_sorter())
    sorted_originals, sorted_orig_names = arrange_files(originals, orig_names, order_sorter())

    aligned = align_sort(list(zip(sorted_originals, sorted_orig_names)), sorted_pred_names, order)
    sorted_originals, sorted_orig_names = zip(*aligned)

    if highest_first:
        return list(reversed(sorted_originals)), list(reversed(sorted_orig_names)), list(reversed(sorted_predictions)), list(reversed(sorted_pred_names))

    return sorted_originals, sorted_orig_names, sorted_predictions, sorted_pred_names


def plot_images(originals, predictions, orig_names, pred_names, n=100):
    for i, (o, p, o_name, p_name) in enumerate(zip(originals, predictions, orig_names, pred_names)):
        p_score = extract_score(p_name)
        p_index = order(p_name)
        o_index = order(o_name)
        print(p_score)
        print(i, o_name, p_name)
        plt.imshow(np.array(o))
        plt.title(f'Original #{o_index}')
        plt.show()
        plt.imshow(np.array(p))
        plt.title(f'Prediction #{p_index}, score: {p_score}')
        plt.show()
        if i == n:
            return

if __name__ == '__main__':
    path = Path.cwd()
    path = path.parent.parent
    path = path / 'predictions'

    originals, orig_names, predictions, pred_names = load_and_preprocess(path)
    originals, orig_names, predictions, pred_names = sort_by_score(originals, orig_names, predictions, pred_names)

    plot_images(originals, predictions, orig_names, pred_names, n=len(originals))
