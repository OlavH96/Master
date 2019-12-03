import requests
import math

skiltpunkt = 95


def lat_plus_meters(lat: float, meters: float) -> float:
    earth = 6378.137
    pi = math.pi
    m = (1 / ((2 * pi / 360) * earth)) / 1000
    return lat + (meters * m)


def lng_plus_meters(lat: float, lng: float, meters) -> float:
    earth = 6378.137
    pi = math.pi
    m = (1 / ((2 * pi / 360) * earth)) / 1000
    return lng + (meters * m) / math.cos(lat * (pi / 180))


def get_road_objects(lat: float, lng: float, range_meters: float = 20):
    """
    :param lat: Latitude of center point
    :param lng: Longitude of center point
    :param range_meters: Range of search in meters, determines radius of square surrounding the center point
    :return: List of road objects
    """
    min_lng = lng_plus_meters(lat, lng, -range_meters)
    max_lng = lng_plus_meters(lat, lng, range_meters)
    min_lat = lat_plus_meters(lat, -range_meters)
    max_lat = lat_plus_meters(lat, range_meters)
    inkluder = '?inkluder=metadata,egenskaper'
    srid = '&srid=wgs84'
    kartutsnitt = f'&kartutsnitt={min_lng},{min_lat},{max_lng},{max_lat}'
    ekstra = f'{inkluder}{srid}{kartutsnitt}'
    url = f'https://www.vegvesen.no/nvdb/api/v2/vegobjekter/{skiltpunkt}{ekstra}'
    print(url)
    r = requests.get(url)

    for e in r.json()['objekter']:
        print(e)


if __name__ == '__main__':
    lat = 63.435691
    lng = 10.417320
    get_road_objects(lat, lng)
