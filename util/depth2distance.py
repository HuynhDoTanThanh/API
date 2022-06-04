def depth_to_distance(depth):
    return float('{:.2f}'.format((1/depth) - 0.5 - 8*(depth**1.2) + 8))