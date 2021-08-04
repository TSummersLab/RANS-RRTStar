#!/usr/bin/env python3
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS

Author:
Venkatraman Renganathan
Email:
vrengana@utdallas.edu

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script contains functions used for performing point/obstacle and straight line/obstacle collision checks.
Obstacles are asssumed 2D rectangles.

"""


def point_obstacle_collision_flag(state, obstaclelist, envbounds, robrad):
    """
    Performs a collision check between a point and all obstacles in a list.
    Also confirms that the point is inside of a given environment.
    Inputs:
    state: 2D state. if dimension is larger, only first two scalars are considered
    obstaclelist: list of obstacles arranged as [[ox,oy,wd,ht],[ox,oy,wd,ht], ...] (ox,oy are lower left corner, wd, ht are width and height)
    envbounds: list of environment bounds [xmin,xmax,ymin,ymax]
    robrad: robot radius. if nonzero, it is used to enlarge obstacles and shrink the environment
    Outputs:
    True if collision detected, False if safe
    """

    collision_found = False

    x, y = state[0], state[1]  # extract state x and y values
    xmin, xmax, ymin, ymax = envbounds[0], envbounds[1], envbounds[2], envbounds[3]

    # check if state is in the environment
    if not ((xmin + robrad) <= x <= (xmax - robrad) and (ymin + robrad) <= y <= (
            ymax - robrad)):  # state not in environment
        collision_found = True
        return collision_found

    for (ox, oy, wd, ht) in obstaclelist:
        left = ox
        right = ox + wd
        bottom = oy
        top = oy + ht

        if (left - robrad) <= x <= (right + robrad) and (bottom - robrad) <= y <= (
                top + robrad):  # state in an obstacle
            collision_found = True
            return collision_found

    return collision_found


def line_obstacle_collision_flag(state1, state2, obstaclelist, robrad):
    """
    Performs a collision check between a line connecting two states and a list of obstacles
    We assume that points are inside the environment and not colliding with any obstacles.

    Inputs:
    state1, state2: two states to check (must have passed the point_obstacle_collision_flag check)
    obstaclelist: list of obstacles arranged as [[ox,oy,wd,ht],[ox,oy,wd,ht], ...]
    (ox,oy are lower left corner, wd, ht are width and height)
    robrad: robot radius. if nonzero, it is used to enlarge obstacles and shrink the environment

    Outputs:
    True if collision detected, False if safe
    """

    collision_found = False

    # Get the coordinates of the line connecting two points
    x1 = state1[0]
    y1 = state1[1]
    x2 = state2[0]
    y2 = state2[1]

    for (ox, oy, wd, ht) in obstaclelist:
        # Prepare bloated version of min and max x,y positions of obstacle
        minX = ox - robrad
        minY = oy - robrad
        maxX = ox + wd + robrad
        maxY = oy + ht + robrad

        # Condition for Line to be Completely outside the rectangle: both points on the same side of the obstacle
        if (x1 <= minX and x2 <= minX or  # on the left side
                y1 <= minY and y2 <= minY or  # on the bottom side
                x1 >= maxX and x2 >= maxX or  # on the right side
                y1 >= maxY and y2 >= maxY):  # on the top side
            continue  # if both points on the same side of the obstacle --> no collision possible --> move on

        # The two state are on two different sides of the obstacle
        # Calculate the slope of the line
        lineSlope = (y2 - y1)/(x2 - x1)

        # Connect with a line to other point and check if it lies inside
        yPoint1 = lineSlope*(minX - x1) + y1
        yPoint2 = lineSlope*(maxX - x1) + y1
        xPoint1 = (minY - y1)/lineSlope + x1
        xPoint2 = (maxY - y1)/lineSlope + x1

        if (minY < yPoint1 < maxY or
                minY < yPoint2 < maxY or
                minX < xPoint1 < maxX or
                minX < xPoint2 < maxX):
            collision_found = True
            return collision_found

    return collision_found


# some examples
if __name__ == '__main__':
    obstaclelist = [[0, 0, 1, 1]]  # one square lower left: (0,0), top right: (1,1)
    envbounds = [-2, 2, -2, 2]  # [xmin,xmax,ymin,ymax]
    robrad = 0.001

    s1 = [0.5, 0.5]  # inside obstacle (NOT SAFE)
    s2 = [10, 10]  # outside evironment (NOT SAFE)
    s3 = [1.5, 1.5]  # safe point to the top-right side of the obstacle (SAFE)
    s4 = [1.5, 0.1]  # safe point to the right side of the obstacle (SAFE)
    s5 = [0, 1.1]  # safe point to the top of the obstacle (SAFE)
    # the segment s3-s4 does not intersect the obstacle (SAFE)
    # the segment s4-s5 intersects the obstacle (NOT SAFE)

    s1status = point_obstacle_collision_flag(s1, obstaclelist, envbounds, robrad)
    s2status = point_obstacle_collision_flag(s2, obstaclelist, envbounds, robrad)
    s3status = point_obstacle_collision_flag(s3, obstaclelist, envbounds, robrad)
    s4status = point_obstacle_collision_flag(s4, obstaclelist, envbounds, robrad)
    s5status = point_obstacle_collision_flag(s5, obstaclelist, envbounds, robrad)

    l1status = line_obstacle_collision_flag(s3, s4, obstaclelist, robrad)
    l2status = line_obstacle_collision_flag(s4, s5, obstaclelist, robrad)

    print('s1 is in collision: ', s1status)  # (True)
    print('s2 is in collision: ', s2status)  # (True)
    print('s3 is in collision: ', s3status)  # (False)
    print('s4 is in collision: ', s4status)  # (False)
    print('s5 is in collision: ', s5status)  # (False)

    print('line s3-s4 is in collision: ', l1status)  # (False)
    print('line s4-s5 is in collision: ', l2status)  # (True)
