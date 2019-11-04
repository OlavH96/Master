from roadai.api import Api
import configparser
import os
from pathlib import Path
import logging as logger
import pickle

logger.getLogger().setLevel(logger.INFO)
root_dir = Path.cwd()#.parent

config = configparser.ConfigParser()
config.read(root_dir / 'config.ini')
username = config['RoadAI']['username']
password = config['RoadAI']['password']
output_dir = config['Directories']['downloaded_videos']
output_dir = root_dir / output_dir

saved_observations_dir = config['Directories']['saved_observations']
saved_observations_dir = root_dir / saved_observations_dir

api = Api()
auth = api.login(username, password)


def download_videos(observations):
    logger.info(f'Downloading {len(observations)} videos')
    if not output_dir.exists():
        output_dir.mkdir()
    for i, obs in enumerate(observations):
        video_url = obs['video_url_hd']
        output_video_name = f"{obs['_id']}.mp4"
        if video_url:
            # the API does not support downloading the video files with authentication properly, use custom curl command instead
            download_command = f'curl -H "X-Auth-Token: {auth["authToken"]}" -H "X-User-Id: {auth["userId"]}" -X GET {video_url} -o {output_dir}/{output_video_name}'
            os.system(download_command)


def tie_observations_to_videos(observations):
    videos = os.listdir(output_dir)
    for i, obs in enumerate(observations):
        id = obs['_id']
        video_name = f'{id}.mp4'

        video_is_downloaded = video_name in videos
        if video_is_downloaded:
            video = output_dir / video_name
            obs['video_file'] = video


def get_observations_with_video(limit=10):
    shares = api.shares()
    # for s in shares:
    #     print(s)
    query = {
        "share_id": shares[0]["id"],
        "limit": limit
    }
    observations = api.mobileObservations(**query)
    # print(len(observations))
    # for o in observations:
    #     print(o['video_url_hd'])

    observations = list(filter(lambda o: o['video_url_hd'] != None, observations))
    # print(len(observations))
    logger.info(f'Found {len(observations)} observations with video')
    return observations


def download_videos_if_not_exists(observations):
    exists = os.listdir(output_dir)
    exists = list(map(lambda e: e.split('.')[0], exists))

    not_downloaded = list(filter(lambda o: o['_id'] not in exists, observations))
    download_videos(not_downloaded)


def save_observations_as_json(observations):
    if not saved_observations_dir.exists():
        saved_observations_dir.mkdir()
    for o in observations:
        with open(f"{saved_observations_dir}/{o['_id']}.pkl", 'wb') as f:
            pickle.dump(o, f, 0)


def load_observations_from_json():
    observations = []
    for filename in os.listdir(saved_observations_dir):
        with open(saved_observations_dir / filename, 'rb') as file:
            observations.append(pickle.load(file))

    return observations
