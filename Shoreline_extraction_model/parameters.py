import ee
ee.Initialize()


####################### Shoreline Analysis Parameters #################

# 1. Define the satellite image collection
image_collection = 'Sentinel'     # 'Sentinel' or 'Landsat'

# 2. Define the area of interest
'''Note: always change from null to None, and false to False'''
aoi =  ee.Geometry.Polygon(
        [[[73.3854687682273, 1.8241732981904748],
          [73.3854687682273, 1.8110692571357394],
          [73.39396600638648, 1.8110692571357394],
          [73.39396600638648, 1.8241732981904748]]], None, False);

# 3. Define start date and end date
date = ['2019-09-01', '2019-09-30']

# 4. Georeferencing
horizontal_step = 0     # (+ Positive) Move to right // (- Negative) Move to left
vertical_step = 0       # (+ Positive) Move to top // (- Negative) Move to bottom

# 5. Define custom scheme bin
bins_tsc = [-100, -50, -25, -10, 0, 10, 25, 50, 100]        # For total shoreline change
bins_rpy = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]        # For shoreline change rate per year 

########################################################################