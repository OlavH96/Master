import argparse


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
        '--steps',
        dest='steps',
        type=int,
        default=0,
        help='Steps per epoch, default is length of --path dir'
    )

    parser.add_argument(
        '--architecture',
        dest='model_type',
        choices=['conv', 'fully-connected', 'vae'],
        type=str,
        default='fully-connected',
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
