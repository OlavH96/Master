import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import src.util.Files as Files
import src.util.Arguments as Arguments
import src.util.ImageLoader as ImageLoader
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize


def order_sorter():
    return lambda x: order(x[-1])


def score_sorter():
    return lambda x: extract_score(x[-1])


def indexer(to_index_from: []):
    return lambda x: to_index_from.index(x)


def remove_path(name: str):
    return name.split('/')[-1]


def order(name: str):
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


def load_and_preprocess(path: str, autoremove_missing_files: bool = False):
    origPath = str(path + '/orig*')
    predPath = str(path + '/pred*')

    originals, orig_names_with_path = ImageLoader.load_images(origPath, num=6000)
    predictions, pred_names_with_path = ImageLoader.load_images(predPath, num=6000)

    orig_names = [remove_path(n) for n in orig_names_with_path]
    pred_names = [remove_path(n) for n in pred_names_with_path]
    print("Originals",len(orig_names), orig_names[0])
    print("Predictions",len(pred_names), pred_names[0])
    originals, orig_names = sort_by_order(originals, orig_names)
    predictions, pred_names = sort_by_order(predictions, pred_names)

    orig_orders = [order(o) for o in orig_names]
    pred_orders = [order(p) for p in pred_names]

    print("order", orig_names[0], order(orig_names[0]), orig_orders[0])    

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


def plot_images(originals, predictions, orig_names, pred_names, save_path, n=100, save_by_order=True, show_plot=False, save_fig=True):

    print(f"Plotting {len(orig_names)} images")
    for i, (o, p, o_name, p_name) in enumerate(zip(originals, predictions, orig_names, pred_names)):
        p_score = extract_score(p_name)
        p_index = order(p_name)
        o_index = order(o_name)
        print(f"Creating image {p_index}-{o_index}, {p_score}")
        assert p_index == o_index
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Image #{o_index}')

        ax1.imshow(np.array(o))
        ax1.title.set_text(f'Original')
        ax2.imshow(np.array(p))
        score = "{0:.5f}".format(p_score)
        ax2.title.set_text(f'Prediction, score: {score}')
        if save_fig:
            plt.savefig(f'{save_path}/Comparison_n{i if save_by_order else p_index}_{o_index}_{p_score}.png')
        if show_plot:
            plt.show()
        plt.close(fig)
        if i == n - 1:
            return


def create_score_plot(originals, predictions, orig_names, pred_names, save_path):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    plt.xlabel("Image Number")
    plt.ylabel("Score")
    scores = [extract_score(p) for p in pred_names]
    ax.plot(scores, linewidth=5)
    max_score = max(scores)

    for i, (o, p, o_name, p_name) in enumerate(zip(originals, predictions, orig_names, pred_names)):
        p_score = extract_score(p_name)

        im = np.array(p).astype(np.float) / 255
        o_im = np.array(o).astype(np.float) / 255
        new_size = (32, 32)
        im = resize(im, new_size)
        o_im = resize(o_im, new_size)

        # https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points/53851017
        if i % 2 != 0:
            ab = AnnotationBbox(OffsetImage(o_im), (i, max_score), frameon=False)
            ax.add_artist(ab)
            plt.vlines(x=i, ymin=p_score, ymax=max_score)
        else:
            ab = AnnotationBbox(OffsetImage(o_im), (i, 0), frameon=False)
            ax.add_artist(ab)
            plt.vlines(x=i, ymin=0, ymax=p_score)

        ab = AnnotationBbox(OffsetImage(im), (i, p_score), frameon=False)
        ax.add_artist(ab)

    plt.savefig(f'{save_path}/ScorePlot.png')


def create_dir_for_images(path: str, image_names: [str]) -> Path:
    model_name = extract_model_name(image_names[0])
    p = Path(path) / model_name
    return Files.makedir_else_cleardir(p)


def probability(n):
    out = []

    step = 1 / n
    current = 1
    for i in range(n // 2):
        out.append(current)
        current -= step
    for i in range(n // 2):
        out.append(current)
        current += step

    while len(out) != n:
        out.append(1)

    out = np.array(out)
    out = out / sum(out)

    assert len(out) == n
    return out


if __name__ == '__main__':
    args = Arguments.analyser_arguments()

    predictions_path = args.images_dir

    originals, orig_names, predictions, pred_names = load_and_preprocess(predictions_path, autoremove_missing_files=args.autoremove)

    save_path = create_dir_for_images(args.save_dir, orig_names)
    print("Saving images to ", save_path)
    originals, orig_names, predictions, pred_names = sort_by_score(originals, orig_names, predictions, pred_names, highest_first=True)

    # plot_images(originals, predictions, orig_names, pred_names, save_path=save_path, n=args.num, show_plot=args.visual, save_fig=not args.visual)

    num_scored = 50

    r = sorted(list(np.random.choice(a=list(range(len(orig_names))), size=num_scored, replace=False, p=probability(len(orig_names)))))
    originals = [originals[i] for i in r]
    orig_names = [orig_names[i] for i in r]
    predictions = [predictions[i] for i in r]
    pred_names = [pred_names[i] for i in r]

    create_score_plot(originals, predictions, orig_names, pred_names, save_path)
