import argparse

FC = "fc"
VAE = "vae"
CONVVAE = "conv-vae"
CONV = "conv"
FCS = "fc-s"
FCSS = "fc-ss"


def model_choices_dict():
    return {
        VAE: 'vae',
        CONVVAE: 'conv-vae',
        CONV: 'conv',
        FC: 'fully-connected',
        FCS: 'fully-connected-small',
        FCSS: 'fully-connected-smaller'
    }


def model_choices_list():
    return list(model_choices_dict().values())


def get_model_choice(key):
    assert key in list(model_choices_dict().keys())
    return model_choices_dict()[key]


def anomaly_arguments():
    parser = argparse.ArgumentParser(description="Train and run anomaly detection")

    parser.add_argument(
        '--train',
        action='store_true',
        dest='do_training',
        default=False,
        help='Run with training'
    )

    parser.add_argument(
        '--predict',
        action='store_true',
        dest='do_predict',
        default=False,
        help='Run with predictions'
    )

    parser.add_argument(
        '--num-predictions',
        type=int,
        dest='num_predictions',
        default=10,
        help='Number of predictions'
    )

    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default='model.h5',
        help='Run predictions with saved model'
    )

    parser.add_argument(
        '--max-x',
        dest='max_x',
        type=int,
        default=64,
        help='Max X of trained and predicted images'
    )

    parser.add_argument(
        '--max-y',
        dest='max_y',
        type=int,
        default=64,
        help='Max Y of trained and predicted images'
    )

    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=10,
        help='Epochs for training'
    )

    parser.add_argument(
        '--path',
        dest='path',
        type=str,
        default='detected_images/362*',
        help='Path to examples'
    )

    parser.add_argument(
        '--validation-path',
        dest='validation_path',
        type=str,
        default='',
        help='Path to validation data'
    )

    parser.add_argument(
        '--predict-path',
        dest='pred_path',
        type=str,
        default=None,
        help='Path to predict data, if different from train and validation'
    )

    parser.add_argument(
        '--steps',
        dest='steps',
        type=int,
        default=0,
        help='Steps per epoch, default is length of --path dir'
    )

    parser.add_argument(
        '--architecture',
        dest='model_type',
        choices=model_choices_list(),
        type=str,
        default=model_choices_dict().get('fc'),
        help='Type of anomaly model architecture'
    )

    parser.add_argument(
        '--color',
        dest='color',
        choices=['RGB', 'HSV'],
        type=str,
        default='RGB',
        help='Color mode for loaded images.'
    )
    
    parser.add_argument(
        '--template',
        action='store_true',
        dest='template',
        default=False,
        help='Dont use model, compare with static image instead'
    )

    return parser.parse_args()


def visualizer_arguments():
    parser = argparse.ArgumentParser(description='Run video analysis.')

    parser.add_argument(
        '--no-visual',
        dest='visual',
        action='store_false',
        default=True,
        help='Run without displaying video feed'
    )

    parser.add_argument(
        '--do-not-use-cached',
        dest='cached',
        action='store_false',
        default=True,
        help='Run using downloaded observations and videos'
    )
    parser.add_argument(
        '--extract-detected',
        dest='extract',
        action='store_true',
        default=False,
        help='Extract and save detected objects'
    )

    parser.add_argument(
        '--extraction-certainty',
        dest='extract_limit',
        type=float,
        default=0.5,
        help='Prediction certainty for extracting image'
    )

    parser.add_argument(
        '--number-of-videos',
        dest='num_vids_to_analyze',
        type=int,
        default=1000,
        help='Number of videos to analyze'
    )

    parser.add_argument(
        '--skip',
        dest='num_vids_to_skip',
        type=int,
        default=0,
        help='Number of videos to skip'
    )

    parser.add_argument(
        '--raw-videos',
        dest='raw_videos_path',
        type=str,
        default='',
        help='Analyze raw videos with no metadata'
    )

    parser.add_argument(
        '--save-images-dir',
        dest='save_images_dir',
        type=str,
        default='./detected_images',
        help='Which directory to save detected images to.'
    )
    

    return parser.parse_args()


def downloader_arguments():
    parser = argparse.ArgumentParser(description='Download videos from API')
    parser.add_argument(
        '--num-videos',
        dest='num_videos',
        type=int,
        default=100,
        help='Number of videos to download'
    )

    parser.add_argument(
        '--save',
        dest='save',
        action='store_true',
        default=False,
        help='Should save and store videos to json'
    )

    return parser.parse_args()


def analyser_arguments():
    parser = argparse.ArgumentParser(description='Analyze predicted images VS actual images.')
    parser.add_argument(
        '--images-dir',
        dest='images_dir',
        type=str,
        help='Directory containing originals and predictions',
        required=True
    )

    parser.add_argument(
        '--detected-dir',
        dest='detected_dir',
        type=str,
        help='Directory containing detected images to clean using results of analyis',
        required=False
    )

    parser.add_argument(
        '--purge',
        dest='purge',
        action='store_true',
        default=False,
        help='Purge images in --detected-dir, OR the backup dir if --backup, which have too high loss.',
    )

    parser.add_argument(
        '--purge-overfitted',
        dest='purge_overfitted',
        action='store_true',
        default=False,
        help='Purge images in --detected-dir, OR the backup dir if --backup, which have too low loss.',
    )

    parser.add_argument(
        '--backup',
        dest='backup',
        action='store_true',
        default=False,
        help='Create new dir for cleaned results instead of removing from the original folder.',
    )

    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        type=str,
        help='Directory to save results',
        required=True
    )

    parser.add_argument(
        '--no-visual',
        dest='visual',
        action='store_true',
        default=False,
        help='Run without utilizing visual display'
    )

    parser.add_argument(
        '--autoremove',
        dest='autoremove',
        action='store_true',
        default=False,
        help='Delete unmatched orignals / predictions automatically'
    )

    parser.add_argument(
        '--plot',
        dest='create_plots',
        action='store_true',
        default=False,
        help='Create and save plots?'
    )

    parser.add_argument(
        '--known',
        dest='known',
        action='store_true',
        default=False,
        help='--images-dir arg is labeled data.'
    )

    parser.add_argument(
        '--knn',
        dest='knn',
        action='store_true',
        default=False,
        help='Analyze with KNN'
    )

    parser.add_argument(
        '--num',
        dest='num',
        type=int,
        default=100,
        help='Number of analyzed images'
    )

    parser.add_argument(
        '--plot-num',
        dest='plot_num',
        type=int,
        default=50,
        help='Number of images to plot'
    )

    return parser.parse_args()
