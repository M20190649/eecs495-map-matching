# Imports
import os
import math
import csv
import operator
import multiprocessing as mp
import itertools
import time
import json
import gmplot

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nvector as nv

from datetime import datetime
from haversine import haversine
from functools import reduce
from IPython.display import IFrame
from collections import namedtuple

# Custom classes
import link_classes as lc
import graph_classes as graph
import dist_functions as dist

# Constants
DATA_DIR = '../data'

GOOGLE_MAPS_KEY = ''
with open('config.json') as data_file:
    data = json.load(data_file)
    GOOGLE_MAPS_KEY = data['google-maps-key']

FRAME = nv.FrameE(a=6371e3, f=0)

# Utility functions
def bearing(start, end):
    """
    Computes the bearing in degrees between two geopoints

    Inputs:
        start (tuple of lat, long): starting geolocation
        end (tuple of lat, long): ending geolocation

    Outputs:
        (float): bearing in degrees between start and end
    """
    phi_1 = math.radians(start[0])
    phi_2 = math.radians(end[0])
    lambda_1 = math.radians(start[1])
    lambda_2 = math.radians(end[1])

    x = math.cos(phi_2) * math.sin(lambda_2 - lambda_1)
    y = math.cos(phi_1) * math.sin(phi_2) - (math.sin(phi_1) * math.cos(phi_2) * math.cos(lambda_2 - lambda_1))

    return (math.degrees(math.atan2(x, y)) + 360) % 360

probe_headers = ['sampleID',
             'dateTime',
             'sourceCode',
             'latitude',
             'longitude',
             'altitude',
             'speed',
             'heading']

probe_data = pd.read_csv(os.path.join(DATA_DIR, 'Partition6467ProbePoints.csv'), header=None, names=probe_headers)
probe_data.drop_duplicates(inplace=True)
probe_data['id'] = probe_data['sampleID'].map(str) + '_' + probe_data['dateTime']
probe_data['dateTime'] = pd.to_datetime(probe_data['dateTime'], format='%m/%d/%Y %I:%M:%S %p')
probe_data.sort_values(['sampleID', 'dateTime'], ascending=[True, True], inplace=True)
probe_data.reset_index(drop=True, inplace=True)

link_headers = ['linkPVID',
                'refNodeID',
                'nrefNodeID',
                'length',
                'functionalClass',
                'directionOfTravel',
                'speedCategory',
                'fromRefSpeedLimit',
                'toRefSpeedLimit',
                'fromRefNumLanes',
                'toRefNumLanes',
                'multiDigitized',
                'urban',
                'timeZone',
                'shapeInfo',
                'curvatureInfo',
                'slopeInfo']

# load raw link data
link_data = pd.read_csv(os.path.join(DATA_DIR, 'Partition6467LinkData.csv'), header=None, names=link_headers)
link_data['speedLimit'] = link_data[['fromRefSpeedLimit', 'toRefSpeedLimit']].max(axis=1)
link_data.loc[link_data['speedLimit'] == 0, 'speedLimit'] = 1
link_data['shapeArray'] = link_data['shapeInfo'].apply(lambda x: [[float(j) for j in i.split('/')[:2]] for i in x.split('|')])
link_data['location'] = link_data['shapeArray'].apply(lambda x: reduce(np.add, x) / len(x))
link_data.head()

# create link data lookup dictionary
links = []
link_db = lc.LinkDatabase()
with open(os.path.join(DATA_DIR, 'Partition6467LinkData.csv'), 'r') as csvfile:
    rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
    for r in rdr:
        rl = lc.RoadLink(r)
        links.append(rl)
        link_db.insert_link(rl)

# create road link graph
node_id_map = {}
with open(os.path.join(DATA_DIR, 'Partition6467LinkData.csv'), 'r') as csvfile:
    rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
    for r in rdr:
        try:
            node_id_map[r['refNodeID']] += [r['linkPVID']]
        except KeyError:
            node_id_map[r['refNodeID']] = [r['linkPVID']]

road_graph = graph.RoadNetwork(node_id_map)
with open(os.path.join(DATA_DIR, 'Partition6467LinkData.csv'), 'r') as csvfile:
    rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
    for r in rdr:
        road_graph.insert(graph.RoadNetworkNode(node_id_map, \
                                                [[float(j) for j in i.split('/')[:2]] for i in r['shapeInfo'].split('|')], \
                                                r['linkPVID'], \
                                                r['nrefNodeID']))

Candidate = namedtuple('Candidate', ['linkPVID', 'location', 'speedLimit'])

def nearest_n_segments(lat, long, n):
    """
    Uses link_db to find nearest n road segments

    Inputs:
        lat (float): latitude of probe point
        lon (float): longitude of probe point
        n (int): number of roads road segments to return

    Output:
        (list of Candidate tuples): Candidate tuples of n-nearest road segments

    """
    # find nearest n links
    output = []
    try:
        link_search = [(x, haversine(x.avgLatLong, (lat, long))) for x in link_db.get_links(lat, long)]
        link_search.sort(key=operator.itemgetter(1))
        link_search = link_search[0:n]

        # extract only link PVIDs from search
        output = [(int(x[0].linkPVID), x[1]) for x in link_search]
    except KeyError:
        pass

    # format output with additional data
    for i in range(len(output)):
        link_information = link_data[link_data['linkPVID'] == output[i][0]]
        formatted_output = Candidate(int(link_information['linkPVID']), \
                                     list(link_information['location'])[0], \
                                    float(link_information['speedLimit']))
        output[i] = formatted_output

    return output

def observation_probability(p, c):
    """
    Compute the likelihood that a GPS point p matches a candidate point c based on the distance between the two.

    Inputs:
        p (tuple of lat, long): probe point
        c (tuple of lat, long): candidate point

    Output:
        (float): probability of the above
    """
    distance = haversine(p, c)
    return (1 / math.sqrt(2 * math.pi * 20)) * math.exp((distance**2) / (2 * 20**2))

def transmission_probability(p_prev, p_curr, c_prev, c_curr):
    """
    Compute the likelihood that the true path from p_prev to p_curr follows the shortest path from c_prev to c_curr

    Inputs:
        p_prev (tuple of lat, long): previous probe point
        p_curr (tuple of lat, long): current probe point
        c_prev (tuple of lat, long): candidate point for previous probe point
        c_curr (tuple of lat, long): candidate point for current probe point

    Output:
        (float): probability of the above
    """
    p_dist = haversine(p_prev, p_curr)
    c_dist = haversine(c_prev, c_curr)

    if c_dist == 0:
        return 1

    return p_dist / c_dist

def spatial_analysis(p_prev, p_curr, c_prev, c_curr):
    """
    Compute the spatial measurement value for two neighboring probe points p_prev and p_curr for
    two candidate points c_prev, c_curr

    Inputs:
        p_prev (tuple of lat, long): previous probe point
        p_curr (tuple of lat, long): current probe point
        c_prev (tuple of lat, long): candidate point for previous probe point
        c_curr (tuple of lat, long): candidate point for current probe point

    Output:
        (float): spatial measurement value
    """
    return observation_probability(p_curr, c_curr) * transmission_probability(p_prev, p_curr, c_prev, c_curr)

def cosine_dist(a, b):
    """
    Compute cosine distance between vectors a, b

    Input:
        a (numpy array): vector a
        b (numpy array): vector b

    Output:
        (float): cosine distance between a, b
    """
    numerator = np.sum(a * b)
    denominator = math.sqrt(np.sum(a ** 2)) * math.sqrt(np.sum(b ** 2))
    return numerator / denominator

def temporal_analysis(p_prev, p_curr, c_prev, c_curr, speed_prev, speed_curr, delta_t):
    """
    Compute the temporal meaurement value for two neighboring probe points and their candidate points c_prev, c_curr

    Inputs:
        c_prev (tuple of lat, long): candidate point for previous probe point
        c_curr (tuple of lat, long): candidate point for current probe point
        speed_prev (float): speed limit of previous road segment
        speed_curr (float): speed limit of current road segment
        delta_t (float): time between previous probe and current probe point measurments

    Output:
        (float): temporal measurment value
    """
    avg_speed = haversine(p_prev, p_curr) / delta_t
    return cosine_dist(np.array([avg_speed, avg_speed]), np.array([speed_prev, speed_curr]))

def st_function(p_prev, p_curr, c_prev, c_curr, speed_prev, speed_curr, delta_t):
    """
    Computes st measurement

    Inputs:
        p_prev (tuple of lat, long): previous probe point
        p_curr (tuple of lat, long): current probe point
        c_prev (tuple of lat, long): candidate point for previous probe point
        c_curr (tuple of lat, long): candidate point for current probe point
        speed_prev (float): speed limit of previous road segment
        speed_curr (float): speed limit of current road segment
        delta_t (float): time between previous probe and current probe point measurments

    Output:
        (float): st measurement
    """
    return spatial_analysis(p_prev, p_curr, c_prev, c_curr) * \
           temporal_analysis(p_prev, p_curr, c_prev, c_curr, speed_prev, speed_curr, delta_t)

def find_candidates(sample_id, location, delta_t):
    """
    Finds nearest candidates for probe data point

    Input:
        sample_id (string): sample_id
        location (tuple): (lat, long)
        delta_t (float): time diff with last probe

    Output:
        (tuple): (sample_id, (lat, long), delta_t, [Candidates])
    """
    return (sample_id, location, delta_t, nearest_n_segments(location[0], location[1], 3))

def st_matching_algorithm(sample):
    """
    Match trajectory points to road links

    Input:
        sample (pandas dataframe): probe data matching a sample id

    Output:
        (list of tuples): list of tuples (sample_id, linkPVID)
    """
    # get list of candidate points
    candidates_for_id = [None for x in range(len(sample))]
    sample_ids = [None for x in range(len(sample))]
    counter = 0

    for row in sample.itertuples():
        delta_t = 0
        try:
            delta_t = (sample.loc[row.Index, 'dateTime'] - sample.loc[row.Index - 1, 'dateTime']).seconds
        except KeyError:
            pass

        candidates_for_id[counter] = find_candidates(row.id, (row.latitude, row.longitude), delta_t)
        sample_ids[counter] = row.id
        counter += 1

    # find matched sequence
    matched_sequence = find_matched_sequence(candidates_for_id)

    # zip together sample_id and linkPVID
    return zip(sample_ids, matched_sequence)

def find_matched_sequence(candidates):
    """
    Find longest matching sequence given candidates

    Input:
        candidates (list of tuple): see find_candidates

    Output:
       (list of tuples): list of tuples (sample_id, linkPVID)
    """
    # setup variables
    parents = [{str(i.linkPVID): None for i in candidate[3]} for candidate in candidates]
    f = [{str(i.linkPVID): 0 for i in candidate[3]} for candidate in candidates]
    candidate_count = len(candidates)

    # set f[0] equal to observation probability
    for c in candidates[0][3]:
        f[0][str(c.linkPVID)] = observation_probability(candidates[0][1], c.location)

    # compute scores for each node
    for i in range(1, candidate_count):
        for cs in candidates[i][3]:
            max_val = -math.inf
            for ct in candidates[i - 1][3]:
                # check if cs is a valid neighbor of ct
                if (str(cs.linkPVID) != str(ct.linkPVID)) and (str(cs.linkPVID) not in road_graph.nodes[str(ct.linkPVID)].neighbors):
                    f[i][str(cs.linkPVID)] = max_val
                    continue

                # define all the variables then compute new score
                p_prev = candidates[i - 1][1]
                p_curr = candidates[i][1]
                c_prev = ct.location
                c_curr = cs.location
                speed_prev = ct.speedLimit
                speed_curr = cs.speedLimit
                delta_t = candidates[i][2]

                alt = f[i - 1][str(ct.linkPVID)] + st_function(p_prev, p_curr, c_prev, c_curr, speed_prev, speed_curr, delta_t)

                # check if higher than existing
                if alt > max_val:
                    max_val = alt
                    parents[i][str(cs.linkPVID)] = ct.linkPVID

                # set max value for current node
                f[i][str(cs.linkPVID)] = max_val

    # compute path
    r_list = []
    c = max(f[candidate_count - 1], key=f[candidate_count - 1].get)
    for i in range(candidate_count - 1, 0, -1):
        r_list.append(c)
        if c is not None:
            c = parents[i][str(c)]
    r_list.append(c)

    return r_list[::-1]

# time code
t0 = time.time()

# sample only first sample_size to make computation faster
sample_ids = list(probe_data['sampleID'].unique())
sample_size = len(sample_ids) # for all data

# parallelizable function
def st_matching_parallel(indicies):
    """
    Links road to probe for set of indicies

    Input:
        indicies (list of floats): indicies to select from sampleIDs
    """
    sids = sample_ids[indicies[0]:indicies[1]]
    output = [[] for x in range(len(sids))]
    counter = 0
    for sid in sids:
        output[counter] = st_matching_algorithm(probe_data[probe_data['sampleID'] == sid])
        counter += 1

    return list(itertools.chain.from_iterable(output))

# run in parallel
N_CORES = mp.cpu_count()
C_SIZE = math.ceil(sample_size / N_CORES)

pool = mp.Pool(N_CORES)
r = pool.map(st_matching_parallel, [[(C_SIZE * i), ((i + 1) * C_SIZE)] for i in range(N_CORES)])
linkings = list(itertools.chain.from_iterable(r))

# assign values to probe_data
stacked_values = np.dstack(linkings)[0]

try:
    del probe_data['linkPVID']
except KeyError:
    pass

probe_data = probe_data.merge(pd.DataFrame({'id': stacked_values[0], 'linkPVID': stacked_values[1]}), how='left', on=['id'])
probe_data['linkPVID'].fillna(0, inplace=True)
probe_data['linkPVID'] = probe_data['linkPVID'].astype(int)

# finish timing
t1 = time.time()
print(str((t1 - t0) / 60) + ' minutes for ' + str(sample_size) + ' sampleIDs with ' + \
      str(len(probe_data[probe_data['sampleID'].isin(sample_ids[0:sample_size])])) + ' points using ' + \
      str(N_CORES) + ' CPU threads.')

# remove id column
try:
    del probe_data['id']
except KeyError:
    pass

# add direction and shape array columns from link_data to probe_data
probe_data = probe_data.merge(link_data[['linkPVID', 'directionOfTravel', 'shapeArray']], how='left', on=['linkPVID'])

# add dist from ref and link
probe_data['distFromRef'] = math.nan
probe_data['distFromLink'] = math.nan

for row in probe_data.itertuples():
    if type(row.shapeArray) is list:
        probe_point = FRAME.GeoPoint(float(row.latitude), float(row.longitude), degrees=True)

        link_refFrame = FRAME.GeoPoint(row.shapeArray[0][0], row.shapeArray[0][1], degrees=True)
        link_nrefFrame = FRAME.GeoPoint(row.shapeArray[-1][0], row.shapeArray[-1][1], degrees=True)

        probe_data.loc[row.Index, 'distFromRef'] = dist.dist_to_ref(probe_point, link_refFrame)
        probe_data.loc[row.Index, 'distFromLink'] = dist.dist_to_link(probe_point, link_refFrame, link_nrefFrame)

# remove unnecessary columns
probe_data = probe_data.drop(['shapeArray'], axis=1)

# save out file
probe_data.to_csv('./st_linked_data.csv', index=False)
