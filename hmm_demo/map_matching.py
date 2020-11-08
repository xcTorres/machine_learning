# @Time    : 14/10/20 15:36
# @Author  :  xcTorres
# @FileName: map_matching.py

import numpy as np
import pandas as pd
import requests
from hmm_demo import concurrent_request
from math import pi, radians, cos, sin, asin, sqrt, exp
from hmmlearn import hmm
from geojson import Feature, LineString, dump

NEAREST_BASE_URL = 'http://localhost:5010/nearest/'
ROUTE_BASE_URL = 'http://localhost:5010/table/'
DEFAULT_NEAREST_NUMBER = 30
RADIUS_PRECISION = 30
RADIUS_MULTIPLIER = 3

CODE_OK = 200


def calculate_flying_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    try:

        if lon1 < -180 or lon1 > 180 or lon2 < -180 or lon2 > 180:
            return None

        if lat1 < -90 or lat1 > 90 or lat2 < -90 or lat2 > 90:
            return None

        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers is 6371
        km = 6371 * c
        return km * 1000
    except Exception as e:
        return None


def emission_proba(dis):
    var = float(RADIUS_PRECISION)**2
    denom = (2*pi*var)**.5
    num = exp(-(dis)**2/(2*var))
    return num/denom


def transition_proba(dis):
    return 1/10.0 * exp(-1 * dis / 10.0)


def map_matching(coord_list):
    print("observation number: {}".format(len(coord_list)))

    params_list = []
    for i, coor in enumerate(coord_list):
        params = {
            'number': DEFAULT_NEAREST_NUMBER,
            'generate_hints': False,
            'coordinates': [coor[:2]]
        }
        params_list.append(params)
    nearest_response = concurrent_request.fetch(NEAREST_BASE_URL, params_list)

    candidate_waypoint_list = []
    candidate_distance_list = []
    for near in nearest_response:
        distance_mapping = {}
        if near['code'] != "Ok":
            candidate_distance_list.append(distance_mapping)
            continue
        for way_point in near['waypoints']:
            if way_point['distance'] > RADIUS_PRECISION * RADIUS_MULTIPLIER:
                continue
            if way_point['location'] not in candidate_waypoint_list:
                candidate_waypoint_list.append(way_point['location'])
                size = len(candidate_waypoint_list)
                distance_mapping[size - 1] = round(way_point['distance'], 6)
            else:
                idx = candidate_waypoint_list.index(way_point['location'])
                distance_mapping[idx] = round(way_point['distance'], 6)
        candidate_distance_list.append(distance_mapping)

    print('hidden state number: {}'.format(len(candidate_waypoint_list)))

    candidate_distance_mat = np.full((len(candidate_waypoint_list), len(coord_list)), np.inf)
    for i, distance_mapping in enumerate(candidate_distance_list):
        for k, v in distance_mapping.items():
            candidate_distance_mat[k][i] = v

    emission_proba_v = np.vectorize(emission_proba)
    emission_prob_mat = emission_proba_v(candidate_distance_mat)

    json_data = {
        'coordinates': candidate_waypoint_list,
        'annotations': ['distance', 'duration'],
        'generate_hints': False,
    }

    res = requests.post(ROUTE_BASE_URL, json=json_data)
    transit_proba_mat = np.array(res.json()['distances'])

    state_size = len(candidate_waypoint_list)
    for i in range(state_size):
        for j in range(state_size):
            flying_distance = calculate_flying_distance(candidate_waypoint_list[i][0], candidate_waypoint_list[i][1],
                                                        candidate_waypoint_list[j][0], candidate_waypoint_list[j][1])

            diffs = abs(transit_proba_mat[i, j] - flying_distance)
            transit_proba_mat[i, j] = transition_proba(diffs)
    # transit_proba_mat = transit_proba_mat / transit_proba_mat.sum(axis=1)[:, np.newaxis]

    model = hmm.MultinomialHMM(n_components=len(candidate_waypoint_list))
    model.startprob_ = emission_prob_mat[:, 0]
    model.transmat_ = transit_proba_mat
    model.emissionprob_ = emission_prob_mat

    oberserve_seq = [i for i in range(len(coord_list))]
    oberserve_seq = np.atleast_2d(oberserve_seq).T

    logprob, state_predict = model.decode(oberserve_seq, algorithm="viterbi")
    print('log loss: {}'.format(logprob))

    res = []
    for i in state_predict:
        res.append(candidate_waypoint_list[i])

    return res


if __name__ == '__main__':

    data = pd.read_csv('./data.csv')
    test_traj = data.iloc[110]['order_traj']

    coord_list = []
    for t in test_traj.split(';'):
        p = t.split(',')
        if int(float(p[0])) == 0:
            continue
        coord_list.append([float(p[2]), float(p[1]), int(float(p[0]))])

    print('input coordinates: {}'.format(LineString(coord_list)))
    matching_res = map_matching(coord_list)
    matching_res = Feature(geometry=LineString((matching_res)))
    print('output coordinates: {}'.format(matching_res))
    with open('matching_res.geojson', 'w') as f:
        dump(matching_res, f)


















