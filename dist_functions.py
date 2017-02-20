import math

def dist_to_link(probe_point, link_start, link_end):
    """Get the perpendicular distance from a point to a line determined
    by two points

    Args:
        probe_point (tuple): the lat,lon coordinates of the probe point
        link_start (tuple): the lat,lon coordinates of the link start
        link_end (tuple): the lat,lon coordinates of the link end

    Returns:
        The perpendicular distance (float)
    """
    numer = abs((link_end[1] - link_start[1])*probe_point[0] - (link_end[0] - link_start[0])*probe_point[1] + link_end[0]*link_start[1] - link_end[1]*link_start[0])

    denom = math.sqrt((link_end[1] - link_start[1])**2 + (link_end[0] - link_start[0])**2)

    return (numer / (denom + 0.000000000001)) * 100000.0 # approximately meters


