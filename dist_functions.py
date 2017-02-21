import nvector as nv
FRAME = nv.FrameE(a=6371e3, f=0)

def dist_to_link(probe_point, link_start, link_end):
    """Get the perpendicular distance from a point to a line determined
    by two points. Courtesy of example 10 from:

    https://pypi.python.org/pypi/nvector

    Args:
        probe_point (tuple): the lat,lon coordinates of the probe point
        link_start (tuple): the lat,lon coordinates of the link start
        link_end (tuple): the lat,lon coordinates of the link end

    Returns:
        The perpendicular distance (float)
    """
    #frame = nv.FrameE(a=6371e3, f=0)
    #pointA1 = frame.GeoPoint(link_start[0], link_start[1], degrees=True)
    #pointA2 = frame.GeoPoint(link_end[0], link_end[1], degrees=True)
    #pointB = FRAME.GeoPoint(probe_point[0], probe_point[1], degrees=True)
    pathA = nv.GeoPath(link_start, link_end)
    s_xt = pathA.cross_track_distance(probe_point, method='greatcircle').ravel()
    return abs(s_xt[0])

def dist_to_ref(probe_point, link_start):
    """Get the perpendicular distance from a point to a line determined
    by two points. Courtesy of example 5 from:

    https://pypi.python.org/pypi/nvector

    Args:
        probe_point (tuple): the lat,lon coordinates of the probe point
        link_start (tuple): the lat,lon coordinates of the link start
        link_end (tuple): the lat,lon coordinates of the link end

    Returns:
        The perpendicular distance (float)
    """
    #frame = nv.FrameE(a=6371e3, f=0)
    #pointA1 = frame.GeoPoint(link_start[0], link_start[1], degrees=True)
    #pointA2 = frame.GeoPoint(link_end[0], link_end[1], degrees=True)
    #pointB = FRAME.GeoPoint(probe_point[0], probe_point[1], degrees=True)
    path = nv.GeoPath(probe_point, link_start)
    s_AB2 = path.track_distance(method='greatcircle').ravel()
    #d_to_ref, _, _ = link_start.distance_and_azimuth(probe_point)
    return abs(s_AB2[0])
