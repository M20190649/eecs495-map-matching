import os
import csv
import math
import nvector as nv

import multiprocessing as mp

DATA_DIR = "probe_data_map_matching"
FRAME = nv.FrameE(a=6371e3, f=0)

from link_classes import RoadLink
from link_classes import LinkDatabase
from dist_functions import dist_to_link
from dist_functions import dist_to_ref
from haversine import haversine





links = []
link_db = LinkDatabase()
link_headers = ["linkPVID",
                "refNodeID",
                "nrefNodeID",
                "length",
                "functionalClass",
                "directionOfTravel",
                "speedCategory",
                "fromRefSpeedLimit",
                "toRefSpeedLimit",
                "fromRefNumLanes",
                "toRefNumLanes",
                "multiDigitized",
                "urban",
                "timeZone",
                "shapeInfo",
                "curvatureInfo",
                "slopeInfo"]
with open(os.path.join(DATA_DIR, "Partition6467LinkData.csv"), 'r') as csvfile:
    rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
    for r in rdr:
        rl = RoadLink(r)
        links.append(rl)
        link_db.insert_link(rl)

print("Links Loaded")

probe_headers = ["sampleID",
                 "dateTime",
                 "sourceCode",
                 "latitude",
                 "longitude",
                 "altitude",
                 "speed",
                 "heading"]

probes = []
with open(os.path.join(DATA_DIR, "Partition6467ProbePoints.csv"), 'r') as csvfile:
    rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=probe_headers)
    for r in rdr:
        probes.append(r)

print("Probes Loaded")

# Compute distances

def find_nearest_link(probe_dict):
    probe_point = FRAME.GeoPoint(float(probe_dict['latitude']),
                                 float(probe_dict['longitude']), degrees=True)
    closest_link = (None, None, None, None)

    for i in link_db.get_links(float(probe_dict['latitude']), float(probe_dict['longitude'])):
        tmp_dist = dist_to_link(probe_point, i.refFrame, i.nrefFrame)

        if closest_link[0] is None or tmp_dist < closest_link[1]:
            closest_link = (i.linkPVID,
                            tmp_dist,
                            dist_to_ref(probe_point, i.refFrame),
                            i.direction)
    return_probe = {k:v for k, v in probe_dict.items()}
    return_probe['linkPVID'] = closest_link[0]
    return_probe['direction'] = closest_link[3]
    return_probe['distFromRef'] = closest_link[2]
    return_probe['distFromLink'] = closest_link[1]
    return return_probe

from datetime import datetime

start_time = datetime.now()

with mp.Pool(64) as pool:
    matched_probes = pool.map(find_nearest_link, probes)

print(matched_probes[:100])

print(datetime.now() - start_time)

new_headers = probe_headers + ['linkPVID', 'direction', 'distFromRef', 'distFromLink']
with open(os.path.join(DATA_DIR, "simple_match.csv"), 'w') as csvfile:
    wtr = csv.writer(csvfile, delimiter=',')
    wtr.writerow(new_headers)
    for i in matched_probes:
        wtr.writerow([i[v] for v in new_headers])






