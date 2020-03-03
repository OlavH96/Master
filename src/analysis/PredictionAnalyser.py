import numpy as np
import os

import matplotlib

matplotlib.use('Agg')

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
import src.util.Files as Files
import src.util.Arguments as Arguments
import src.util.ImageLoader as ImageLoader
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from src.util.Filenames import remove_path, extract_score, extract_model_name
from src.analysis.PostProcessor import remove_from_folder
from sklearn.metrics import roc_auc_score, roc_curve


def order_sorter():
    return lambda x: order(x[-1])


def score_sorter():
    return lambda x: extract_score(x[-1])


def indexer(to_index_from: []):
    return lambda x: to_index_from.index(x)


def order(name: str):
    """
    Extract index of image from my poor naming scheme
    :param name: name from which to extract index
    :return: index of name
    """
    if name.startswith('pred'):
        split = name.split('_')
        if len(str(split[-2])) > 10:  # New file format, -2 is hash
            return int(split[-3])
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


def load_and_preprocess(path: str, autoremove_missing_files: bool = False, num_files=1000):
    origPath = str(path + '/orig*')
    predPath = str(path + '/pred*')

    originals, orig_names_with_path = ImageLoader.load_images(origPath, num=num_files)
    predictions, pred_names_with_path = ImageLoader.load_images(predPath, num=num_files)

    orig_names = [remove_path(n) for n in orig_names_with_path]
    pred_names = [remove_path(n) for n in pred_names_with_path]
    print("Originals", len(orig_names), orig_names[0])
    print("Predictions", len(pred_names), pred_names[0])

    originals, orig_names = sort_by_order(originals, orig_names)
    predictions, pred_names = sort_by_order(predictions, pred_names)

    orig_orders = [order(o) for o in orig_names]
    pred_orders = [order(p) for p in pred_names]

    print("order orig", orig_names[0], order(orig_names[0]), orig_orders[0])
    print("order pred", pred_names[0], order(pred_names[0]), pred_orders[0])

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


def sort_by_score(originals, orig_names, predictions, pred_names, highest_first=True):
    sorted_predictions, sorted_pred_names = arrange_files(predictions, pred_names, score_sorter())
    sorted_originals, sorted_orig_names = arrange_files(originals, orig_names, order_sorter())

    aligned = align_sort(list(zip(sorted_originals, sorted_orig_names)), sorted_pred_names, indexer(pred_names))
    sorted_originals, sorted_orig_names = zip(*aligned)

    if highest_first:
        return list(reversed(sorted_originals)), list(reversed(sorted_orig_names)), list(
            reversed(sorted_predictions)), list(reversed(sorted_pred_names))

    return sorted_originals, sorted_orig_names, sorted_predictions, sorted_pred_names


def plot_images(originals, predictions, orig_names, pred_names, save_path, n=100, save_by_order=True):
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
        plt.savefig(f'{save_path}/Comparison_n{i if save_by_order else p_index}_{o_index}_{p_score}.png')
        plt.close(fig)
        if i == n - 1:
            return


def create_score_plot(originals, predictions, orig_names, pred_names, save_path, avg, std, show_originals=False, title="", rocauc=None):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    plt.xlabel("Image Number")
    plt.ylabel("Score")
    scores = [extract_score(p) for p in pred_names]
    ax.plot(scores, linewidth=5)
    max_score = max(scores)
    y_true = extract_true(pred_names)  # 1 if anomaly, 0 if not

    not_over_limit = len(originals) < 100

    for i, (o, p, o_name, p_name, anomaly) in enumerate(zip(originals, predictions, orig_names, pred_names, y_true)):
        p_score = extract_score(p_name)

        im = np.array(p).astype(np.float) / 255
        o_im = np.array(o).astype(np.float) / 255
        new_size = (32, 32)
        im = resize(im, new_size)
        o_im = resize(o_im, new_size)

        # https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points/53851017
        if show_originals and not_over_limit:
            top = i % 2 != 0
            y = max_score if top else 0
            ymin = p_score if top else 0
            ymax = max_score if top else p_score

            if rocauc:
                ab = AnnotationBbox(OffsetImage(o_im), (i, y), frameon=True, bboxprops=dict(edgecolor='red' if anomaly else 'green'))
            else:
                ab = AnnotationBbox(OffsetImage(o_im), (i, y), frameon=False)
            ax.add_artist(ab)
            plt.vlines(x=i, ymin=ymin, ymax=ymax, colors='grey')

        ab = AnnotationBbox(OffsetImage(o_im if not show_originals else im), (i, p_score), frameon=False)
        ax.add_artist(ab)
    if rocauc:
        fpr, tpr, thresholds = rocauc
        for f, t, l in list(zip(fpr, tpr, thresholds))[2:-1:2]:
            plt.hlines(xmax=len(originals), xmin=0, y=l, colors="red", label="Average")
            f = '{:.2f}'.format(f)
            t = '{:.2f}'.format(t)
            plt.text(0, l, f"FPR {f}, TPR {t}", fontsize=12)

    else:
        plt.hlines(xmax=len(originals), xmin=0, y=avg, colors="red", label="Average")
        plt.hlines(xmax=len(originals), xmin=0, y=avg + std, colors="blue", label="Limit")
        plt.hlines(xmax=len(originals), xmin=0, y=avg - std, colors="blue", label="Limit")
    plt.title(title)
    plt.savefig(f'{save_path}/ScorePlot.png')


def create_dir_for_images(dirname: str, path: str, image_names: [str], extra="") -> Path:
    model_name = dirname

    p = Path(path) / (model_name + extra)
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


def do_plotting(originals, predictions, orig_names, pred_names, n, save_dir, avg, std, pred_dir="", title="", rocauc=None):
    from_dir = pred_dir.split('_')
    extra = ""
    for i, l in enumerate(from_dir[::-1]):
        if l.isdigit():
            break
        if not i == 0:
            l += '_'
        extra = l + extra
    extra = '_' + extra
    dirname = remove_path(pred_dir)
    save_path = create_dir_for_images(dirname, save_dir, orig_names, extra=extra)
    print("Saving images to ", save_path)
    plot_images(originals, predictions, orig_names, pred_names, save_path=save_path, n=n)

    # r = sorted(list(np.random.choice(a=list(range(len(orig_names) - 2)), size=n, replace=False, p=probability(len(orig_names) - 2))))
    # e_o = originals[0]
    # e_p = predictions[0]
    # e_o_n = orig_names[0]
    # e_p_n = pred_names[0]

    # e_o_b = originals[-1]
    # e_p_b = predictions[-1]
    # e_o_n_b = orig_names[-1]
    # e_p_n_b = pred_names[-1]

    # originals = [originals[i] for i in r]
    # orig_names = [orig_names[i] for i in r]
    # predictions = [predictions[i] for i in r]
    # pred_names = [pred_names[i] for i in r]

    # originals.insert(0, e_o)
    # orig_names.insert(0, e_o_n)
    # predictions.insert(0, e_p)
    # pred_names.insert(0, e_p_n)

    # originals.append(e_o_b)
    # orig_names.append(e_o_n_b)
    # predictions.append(e_p_b)
    # pred_names.append(e_p_n_b)
    if n > 50 and n < len(originals) // 2:
        originals = originals[:n // 2] + originals[-(n // 2):]
        orig_names = orig_names[:n // 2] + orig_names[-(n // 2):]
        predictions = predictions[:n // 2] + predictions[-(n // 2):]
        pred_names = pred_names[:n // 2] + pred_names[-(n // 2):]

    create_score_plot(originals, predictions, orig_names, pred_names, save_path, avg, std, show_originals=True, title=title, rocauc=rocauc)


def do_scoring(orig, pred):
    res = roc_auc_score(orig, pred)
    print("ROCAUC", res)
    return res


def extract_true(names):
    return [1 if "anomaly" in n else 0 for n in names]


if __name__ == '__main__':
    args = Arguments.analyser_arguments()
    print(args)

    originals, orig_names, predictions, pred_names = load_and_preprocess(args.images_dir, autoremove_missing_files=args.autoremove, num_files=args.num)
    originals, orig_names, predictions, pred_names = sort_by_score(originals, orig_names, predictions, pred_names, highest_first=True)

    scores = [extract_score(p) for p in pred_names]
    print(scores[:10])
    average = np.average(scores)
    stddev = np.std(scores)
    limit = average + stddev
    print("average", average)
    print("stddev", stddev)
    print("limit", limit)

    y_true = extract_true(pred_names)
    y_score = np.array(scores)
    score = do_scoring(y_true, y_score)

    if args.create_plots:
        do_plotting(
            originals, predictions, orig_names, pred_names,
            n=args.plot_num,
            save_dir=args.save_dir,
            avg=average,
            std=stddev,
            pred_dir=args.images_dir,
            title=f'ROC Score {"{:.2f}".format(score)}' if args.known else f"Average score {average}, stddev {stddev}",
            rocauc=roc_curve(y_true, y_score)
        )
    if args.detected_dir:
        remove_from_folder(
            orig_names=orig_names,
            pred_names=pred_names,
            detected_images_path=args.detected_dir,
            limit=average + stddev,
            limit_lower=average - stddev,
            create_backup=args.backup,
            purge=args.purge,
            purge_overfitted=args.purge_overfitted
        )
