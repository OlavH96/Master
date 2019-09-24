from roadai.api import Api
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
username = config['RoadAI']['username']
password = config['RoadAI']['password']

api = Api()
auth = api.login(username, password)
# api.set_auth(auth_token=auth['authToken'], user_id=auth['userId'])

shares = api.shares()
# for share in shares:
# 	print(share)

query = {
    "share_id": shares[0]["id"],
    "limit": 10
}
observations = api.mobileObservations(**query)
print("### Query ###")
print(observations)
print(observations[1])
# api.download_file(observations[1], "video.mp4")
for obs in observations:
    if 'video_url_hd' in obs:
        print(obs['video_url_hd'])
        api.download_file(obs, "video.mp4")
