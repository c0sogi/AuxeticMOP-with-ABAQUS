# import cadquery as cq
# from cadquery import exporters
#
# polyline_points = [(0, 0), (28, 0), (28, 20), (20, 20)]
# result = cq.Workplane('XZ').polyline(polyline_points).close().extrude(4)
# exporters.export(result, 'output.step')

import pickle
import os

print(os.path.abspath())