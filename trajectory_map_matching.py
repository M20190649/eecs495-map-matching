# Imports
import os
import math
import csv
import operator
import multiprocessing as mp
import itertools
import time
import json

import numpy as np
import pandas as pd
import nvector as nv

from datetime import datetime
from haversine import haversine
from functools import reduce

# Custom classes
import link_classes as lc
import dist_functions as dist

# Constants
DATA_DIR = '../data'

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

# create link data lookup dictionary
links = []
link_db = lc.LinkDatabase()
with open(os.path.join(DATA_DIR, 'Partition6467LinkData.csv'), 'r') as csvfile:
    rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
    for r in rdr:
        rl = lc.RoadLink(r)
        links.append(rl)
        link_db.insert_link(rl)

# Parse out shapeInfo and make averageLocation and heading columns
link_data['shapeArray'] = link_data['shapeInfo'].apply(lambda x: [[float(j) for j in i.split('/')[:2]] for i in x.split('|')])
link_data['averageLocation'] = link_data['shapeArray'].apply(lambda x: reduce(np.add, x) / len(x))
link_data['heading'] = link_data['shapeArray'].apply(lambda x: bearing(x[0], x[-1]))
link_data['flipped_heading'] = (link_data['heading'] + 180) % 360

def nearest_n_segments(lat, long, n):
    """
    Uses link_db to find nearest n road segments

    Inputs:
        lat (float): latitude of probe point
        lon (float): longitude of probe point
        n (int): number of roads road segments to return

    Output:
        (list of tuples): (linkPVID, distance) of n-nearest road segments

    """
    # find nearest n links
    output = []
    try:
        link_search = [(x, haversine(x.refLatLon, (lat, long))) for x in link_db.get_links(lat, long)]
        link_search.sort(key=operator.itemgetter(1))
        link_search = link_search[0:n]

        # extract only link PVIDs from search
        output = [(int(x[0].linkPVID), x[1]) for x in link_search]
    except KeyError:
        pass

    return output

def closest_by_heading(road_links, probe_heading):
    """
    Returns link with closest heading for given probe_heading

    Inputs:
        road_links (list of tuples): list of nearest (linkPVID, distance) tuples
        probe_heading (float): heading of probe gps point

    Outputs:
        (float): linkPVID for closest link
    """
    if len(road_links) == 0:
        return -1

    # get relevant links
    road_link_df = pd.DataFrame({'linkPVID': [x[0] for x in road_links], 'distances': [x[1] for x in road_links]})
    link_headings = link_data[link_data['linkPVID'].isin(road_link_df['linkPVID'])]
    link_headings = link_headings.merge(road_link_df)

    # compute metric from distance and difference in angle. check both directions
    link_headings['angle_diff'] = pd.DataFrame([np.abs(link_headings['heading'] - probe_heading), \
                                                np.abs(link_headings['flipped_heading'] - probe_heading)]).min()
    link_headings['metric'] = link_headings['distances'] * link_headings['angle_diff']

    # pick one with lowest metric and return its linkPVID
    link_headings = link_headings.sort_values(by='metric')
    return link_headings.head(1)['linkPVID']

# time code
t0 = time.time()

# sample only first sample_size to make computation faster
sample_size = len(probe_data) # for all data

# add road link
probe_data['linkPVID'] = 0

# parallelizable function
def link_road_parallel(indicies):
    """
    Links road to probe for set of indicies

    Input:
        indicies (list of floats): indicies to find nearest link for
    """
    output = [(0, 0) for x in range(indicies[1] - indicies[0])]
    n = 3
    counter = 0
    for row in probe_data[indicies[0]:indicies[1]].itertuples():
        output[counter] = (row.Index, closest_by_heading(nearest_n_segments(row.latitude, row.longitude, n),
                                                         row.heading))
        counter += 1

    return output

# run in parallel
N_CORES = mp.cpu_count()
C_SIZE = math.ceil(sample_size / N_CORES)

pool = mp.Pool(N_CORES)
r = pool.map(link_road_parallel, [[(C_SIZE * i), ((i + 1) * C_SIZE)] for i in range(N_CORES)])
linkings = list(itertools.chain.from_iterable(r))

# assign values to probe_data
stacked_values = np.dstack(linkings)[0]
probe_data.loc[stacked_values[0], 'linkPVID'] = stacked_values[1]

# finish timing
t1 = time.time()
print(str((t1 - t0) / 60) + ' minutes for ' + str(sample_size) + ' data points using ' + str(N_CORES) + ' CPU threads.')

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
probe_data.to_csv('./trajectory_linked_data.csv', index=False)
