from roadai.api import Api
import configparser
import os
from pathlib import Path


def download_videos(observations, output_dir):
    if not output_dir.exists():
        output_dir.mkdir()
    for i, obs in enumerate(observations):
        video_url = obs['video_url_hd']
        output_video_name = f"{obs['_id']}.mp4"
        if video_url:
            # the API does not support downloading the video files with authentication properly, use custom curl command instead
            download_command = f'curl -H "X-Auth-Token: {auth["authToken"]}" -H "X-User-Id: {auth["userId"]}" -X GET {video_url} -o {output_dir}/{output_video_name}'
            os.system(download_command)


def tie_observations_to_videos(observations, downloaded_videos_path):
    videos = os.listdir(downloaded_videos_path)
    for i, obs in enumerate(observations):
        id = obs['_id']
        video_name = f'{id}.mp4'

        video_is_downloaded = video_name in videos
        if video_is_downloaded:
            video = downloaded_videos_path / video_name
            obs['video_file'] = video
def get_observations_with_video():

    root_dir = Path.cwd().parent

    config = configparser.ConfigParser()
    config.read(root_dir / 'config.ini')
    username = config['RoadAI']['username']
    password = config['RoadAI']['password']
    output_dir = config['Directories']['downloaded_videos']
    output_dir = root_dir / output_dir

    api = Api()
    auth = api.login(username, password)

    shares = api.shares()
    # for share in shares:
    #     print(share)

    query = {
        "share_id": shares[2]["id"],
        "limit": 10
    }
    observations = api.mobileObservations(**query)
    observations = list(filter(lambda o: o['video_url_hd'] != None, observations))
    # download_videos(observations, output_dir)
    tie_observations_to_videos(observations, output_dir)
    return observations
