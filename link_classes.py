import os
import csv
import nvector as nv

DATA_DIR = "probe_data_map_matching"
FRAME = nv.FrameE(a=6371e3, f=0)

class RoadLink(object):
    """Tiny class to hold road link data

    The RoadLink class essentially serves to hold data but does provide a couple
    nice features in the constructor to parse out lat/lons, etc.

    Attributes:
        linkPVID (string): link id
        refNodeID (string): reference node id
        nrefNodeID (string): nonreference node id
        refLatLon (List(Float, Float)): reference node lat-lon
        nrefLatLon (List(Float, Float)): nonreference node lat-lon

        ... add whatever else we decide we want

    """
    def __init__(self, data_row):
        self.linkPVID = data_row['linkPVID']
        self.refNodeID = data_row['refNodeID']
        self.nrefNodeID = data_row['nrefNodeID']
        self.direction = data_row['directionOfTravel']
        lat_lon_points = [[float(j) for j in i.split('/')[:2]] for i in data_row['shapeInfo'].split('|')]
        self.refLatLon = lat_lon_points[0]
        self.nrefLatLon = lat_lon_points[-1]

        self.refFrame = FRAME.GeoPoint(self.refLatLon[0], self.refLatLon[1], degrees=True)
        self.nrefFrame = FRAME.GeoPoint(self.nrefLatLon[0], self.nrefLatLon[1], degrees=True)

class LinkDatabase(object):
    """Class to reduce the amount of links necessary to compare to when matching

    The LinkDatabase class allows us to dramatically reduce the size of the set we need to
    explore when matching probe points to road links. We make squares of .1x.1 to allow fast
    retrieval of the small subset of links near a probe point. Indexing is done based on
    the REFERENCE NODE coordinates, not the non-reference node coordinates.

    Attributes:

    """
    def __init__(self):
        self.link_dict = {}

    def get_links(self, probe_lat, probe_lon):
        """Get nearby links

        We round the lat and lon coords up and down to and return the links in the relevant buckets

        Args:
            probe_lat (Float): the probe point latitude coordinate
            probe_lon (Float): the probe point longitude coordinate

        Returns:
            a list of RoadLink objects
        """
        rounded_lat = round(float(probe_lat), ndigits=2)
        rounded_lon = round(float(probe_lon), ndigits=2)

        links_to_return = []
        for i in [-0.01, 0, 0.01]:
            for j in [-0.01, 0, 0.01]:
                try:
                    links_to_return += self.link_dict[(str(round(rounded_lat+i, ndigits=1)), str(round(rounded_lon+j, ndigits=1)))]
                except KeyError:
                    pass
        return links_to_return

    def insert_link(self, road_link):
        """Insert a link

        Inserts a RoadLink object into the correct bucket

        Args:
            road_link (RoadLink): the road link object to be inserted

        Returns:
            None
        """
        new_lat = str(round(road_link.refLatLon[0], ndigits=2))
        new_lon = str(round(road_link.refLatLon[1], ndigits=2))

        try:
            self.link_dict[(new_lat, new_lon)] += [road_link]
        except KeyError:
            self.link_dict[(new_lat, new_lon)] = [road_link]

if __name__ == '__main__':
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
