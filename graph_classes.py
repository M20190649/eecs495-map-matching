import os
import csv

DATA_DIR = "probe_data_map_matching"

class RoadNetworkNode(object):
    """Super tiny graph node

    This class holds a link corresponding to a road network in germany.
    But, we're calling it a node becuase it makes more sense to use
    that name given how were constructing this graph for curve-to-curve
    matching.

    Attributes:
        id_map (dict of nodeId -> linkPVID) map from node ref ids to link pvids
        coordinates (List of tuples): the coordinates of the nodes on the link
        linkPVID (str): the id of the link
        neighbors (List of strings): the pvids of the adjacents
    """

    def __init__(self, id_map, coordinates, link_pvid, non_ref_id):
        self.coordinates = coordinates
        self.linkPVID = link_pvid
        self.non_ref_id = non_ref_id
        self.neighbors = []
        try:
            self.neighbors += [id_map[non_ref_id]]
        except KeyError:
            # print('Missing edge/node target: {}, skipping...'.format(non_ref_id))
            pass

    def add_neighbor(self, id_map, neighbor):
        self.neighbors.append(id_map[neighbor.non_ref_id])

class RoadNetwork(object):
    """This is a super small class to hold the road network

    This tiny class holds the road network and provides an insert method

    Attributes:
        ref_id_map (dict( ref id -> pvid)) a map to help convert node id to pvid
        nodes (dict(pvid -> RoadNetworkNode)) container for the NetworkNodes

    """

    def __init__(self, id_map):
        self.ref_id_map = id_map
        self.nodes = {}

    def insert(self, nodeObj):

        # assume pvids wont collide
        self.nodes[nodeObj.linkPVID] = nodeObj




if __name__ == '__main__':
    links = []
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
    node_id_map = {}
    with open(os.path.join(DATA_DIR, "Partition6467LinkData.csv"), 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
        for r in rdr:
            try:
                node_id_map[r['refNodeID']] += r['linkPVID']
            except KeyError:
                node_id_map[r['refNodeID']] = [r['linkPVID']]

    road_graph = RoadNetwork(node_id_map)
    with open(os.path.join(DATA_DIR, "Partition6467LinkData.csv"), 'r') as csvfile:
        rdr = csv.DictReader(csvfile, delimiter=',', fieldnames=link_headers)
        for r in rdr:
            road_graph.insert(RoadNetworkNode(node_id_map,
                                              [[float(j) for j in i.split('/')[:2]] for i in r['shapeInfo'].split('|')],
                                              r['linkPVID'],
                                              r['nrefNodeID']))

    for k, v in road_graph.nodes.items():
        if len(v.neighbors) > 0:
            print(k, v.neighbors)
