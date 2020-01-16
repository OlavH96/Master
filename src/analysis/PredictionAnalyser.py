import numpy as np
import os

import src.util.ImageLoader as ImageLoader
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize


def order_sorter():
    return lambda x: order(x[-1])


def score_sorter():
    return lambda x: extract_score(x[-1])


def indexer(to_index_from):
    return lambda x: to_index_from.index(x)


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


def load_and_preprocess(path, autoremove_missing_files=False):
    origPath = str(path / 'orig*')
    predPath = str(path / 'pred*')

    originals, orig_names_with_path = ImageLoader.load_images(origPath)
    predictions, pred_names_with_path = ImageLoader.load_images(predPath)

    orig_names = [remove_path(n) for n in orig_names_with_path]
    pred_names = [remove_path(n) for n in pred_names_with_path]

    originals, orig_names = sort_by_order(originals, orig_names)
    predictions, pred_names = sort_by_order(predictions, pred_names)

    orig_orders = [order(o) for o in orig_names]
    pred_orders = [order(p) for p in pred_names]
    missing_originals = set(orig_orders) - set(pred_orders)
    missing_predictions = set(pred_orders) - set(orig_orders)

    if autoremove_missing_files and (missing_originals or missing_predictions):
        print(f"Removing missing files. Originals removed: {sorted(list(missing_originals))}, Predictions removed: {sorted(list(missing_predictions))}")
        for f in orig_names_with_path:
            orig = remove_path(f)
            if order(orig) in missing_originals:
                os.remove(f)
        for f in pred_names_with_path:
            orig = remove_path(f)
            if order(orig) in missing_predictions:
                os.remove(f)
        return load_and_preprocess(path, autoremove_missing_files=False)
    else:
        assert not missing_originals and not missing_predictions, f"Missing predictions for originals: {sorted(list(missing_originals)) if missing_originals else None}. Missing originals for predictions: {sorted(list(missing_predictions)) if missing_predictions else None} "
    print("Load and preprocessing completed")
    return originals, orig_names, predictions, pred_names


def extract_model_name(name):
    s = name.split('.')[0].split('_')
    return '_'.join(s[1:])


def extract_score(pred):
    split = pred.split('_')
    n = split[-1].split('.')
    return float('.'.join(n[:2]))


def sort_by_score(originals, orig_names, predictions, pred_names, highest_first=True):
    sorted_predictions, sorted_pred_names = arrange_files(predictions, pred_names, score_sorter())
    sorted_originals, sorted_orig_names = arrange_files(originals, orig_names, order_sorter())

    aligned = align_sort(list(zip(sorted_originals, sorted_orig_names)), sorted_pred_names, indexer(pred_names))
    sorted_originals, sorted_orig_names = zip(*aligned)

    if highest_first:
        return list(reversed(sorted_originals)), list(reversed(sorted_orig_names)), list(
            reversed(sorted_predictions)), list(reversed(sorted_pred_names))

    return sorted_originals, sorted_orig_names, sorted_predictions, sorted_pred_names


def plot_images(originals, predictions, orig_names, pred_names, savedir, n=100, save_by_order=True, show_plot=False, save_fig=True):

    model_name = extract_model_name(orig_names[0])
    save_path = Path(savedir) / model_name
    if not save_path.exists():
        save_path.mkdir()
    else:
        # Delete existing files
        files = os.listdir(save_path)
        [os.remove(str(save_path / file)) for file in files]

    for i, (o, p, o_name, p_name) in enumerate(zip(originals, predictions, orig_names, pred_names)):
        p_score = extract_score(p_name)
        p_index = order(p_name)
        o_index = order(o_name)
        assert p_index == o_index
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Image #{o_index}')

        ax1.imshow(np.array(o))
        ax1.title.set_text(f'Original')
        ax2.imshow(np.array(p))
        score = "{0:.5f}".format(p_score)
        ax2.title.set_text(f'Prediction, score: {score}')
        if save_fig:
            plt.savefig(f'{save_path}/Comparison_n{i if save_by_order else p_index}_{o_index}_{p_index}.png')
        if show_plot:
            plt.show()
        plt.close(fig)
        if i == n - 1:
            return


def create_score_plot(originals, predictions, orig_names, pred_names):
    fig, ax = plt.subplots()
    scores = [extract_score(p) for p in pred_names]
    max_score = max(scores)
    ymax = ax.bbox.ymax
    ymin = ax.bbox.ymin
    ydiff = ymax - ymin
    xmax = ax.bbox.xmax
    xmin = ax.bbox.xmin
    xdiff = xmax - xmin

    ax.plot(scores)

    for i, (o, p, o_name, p_name) in enumerate(zip(originals, predictions, orig_names, pred_names)):
        p_score = extract_score(p_name)
        score_ratio = p_score / max_score
        index_ratio = (i + 1) / len(originals)

        im = np.array(p).astype(np.float) / 255
        o_im = np.array(o).astype(np.float) / 255
        im = resize(im, (16, 16))
        o_im = resize(o_im, (16, 16))

        w = im.shape[0]

        xpos = index_ratio * xdiff
        x_offset = xpos + xmin
        ypos = score_ratio * ydiff
        y_offset = ypos + ymin

        fig.figimage(im, x_offset, y_offset)
        if p_score > np.average(scores) + np.std(scores):
            fig.figimage(o_im, x_offset + w, y_offset)

    plt.show()


if __name__ == '__main__':
    path = Path.cwd()
    predictions_path = path / 'predictions' / 'model_rgb_64_fully-connected_10_2020-01-13_36'
    save_path = str(path / 'src' / 'analysis' / 'images')

    originals, orig_names, predictions, pred_names = load_and_preprocess(predictions_path, autoremove_missing_files=True)
    # plot_images(originals, predictions, orig_names, pred_names, n=len(originals))

    originals, orig_names, predictions, pred_names = sort_by_score(originals, orig_names, predictions, pred_names,
                                                                   highest_first=True)

    plot_images(originals, predictions, orig_names, pred_names, savedir=save_path, n=100)
    create_score_plot(originals, predictions, orig_names, pred_names)
