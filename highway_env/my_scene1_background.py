import argparse
import sys

import numpy as np

import display_all as d
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.road.graphics import RoadGraphics, WorldSurface



def add_background_traffic(road, count=10, seed=0, speed_range=(8.0, 14.0)):
    
    #set random
    rng = np.random.default_rng(seed)
    
    #loop over "count" number of count and add vehicles
    for _ in range(count):
        vehicle = IDMVehicle(
            road
        )
        
        road.vehicles.append(vehicle)
        
        