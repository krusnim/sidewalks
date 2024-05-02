import shapely
import random
import math

def polyrand(p):
    minx, miny, maxx, maxy = p.bounds
    point = None
    while point is None:
        q = shapely.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if p.contains(q):
            point = q
    return point

def radiusrand(p, r):
    theta = random.random() * 2 * math.pi

    return shapely.Point(
        p.x + r * math.cos(theta),
        p.y + r * math.sin(theta)
    )