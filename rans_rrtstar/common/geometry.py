import math
import numpy as np
import numpy.random as npr
import numpy.linalg as la

from rans_rrtstar.config import SATLIM


def rotation2d_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    M = np.array([[c, -s],
                  [s, c]])
    return M


def sample_rectangle(bounds):
    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    x = npr.uniform(xmin, xmax)
    y = npr.uniform(ymin, ymax)
    return x, y


def compute_L2_distance(fromNode, toNode):
    """
    Returns the distance between two nodes computed using the Euclidean distance metric
    Input parameters:
    fromNode   : Node representing point A
    toNode     : Node representing point B
    """

    # Use the dynamic control-based distance metric
    diffVec = (fromNode.means[-1, :, :] - toNode.means[-1, :, :])[0:2, 0]  # no heading

    # Compute the radial (Euclidean) target distance
    dist = la.norm(diffVec)
    return dist


def saturate_node_with_L2(fromNode, toNode, saturation_limit=SATLIM):
    """
    If the L2 norm of ||toNode-fromNode|| is greater than some saturation distance, find a new node newToNode along
    the vector (toNode-fromNode) such that ||newToNode-fromNode|| = saturation distance
    Inputs:
    fromNode: from/source node (type: DR_RRTStar_Node)
    toNode: desired destination node (type: DR_RRTStar_Node)
    Oupputs:
    newToNode: either the original toNode or a new one (type: DR_RRTStar_Node)
    """
    if compute_L2_distance(fromNode, toNode) > saturation_limit:
        # node is further than the saturation distance
        from_x = fromNode.means[-1, :, :][0, 0]
        from_y = fromNode.means[-1, :, :][1, 0]
        to_x = toNode.means[-1, :, :][0, 0]
        to_y = toNode.means[-1, :, :][1, 0]
        angle = math.atan2(to_y - from_y, to_x - from_x)
        new_to_x = from_x + saturation_limit * math.cos(angle)
        new_to_y = from_y + saturation_limit * math.sin(angle)
        newToNode = toNode
        newToNode.means[-1, :, :][0, 0] = new_to_x
        newToNode.means[-1, :, :][1, 0] = new_to_y
        return newToNode
    else:
        return toNode
