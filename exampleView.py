#! /usr/bin/env python

from sp_viewer import SPViewer
from nupic.research.spatial_pooler import SpatialPooler

def main():
  
  # Instantiate our spatial pooler
  sp = SpatialPooler(
      inputDimensions = 32**2, # Size of image patch
      columnDimensions = 16, # Number of potential features
      potentialRadius = 10000, # Ensures 100% potential pool
      potentialPct = 1, # Neurons can connect to 100% of input
      globalInhibition = True,
      numActiveColumnsPerInhArea = 1, # Only one feature active at a time
      # All input activity can contribute to feature output
      stimulusThreshold = 0,
      synPermInactiveDec = 0.01,
      synPermActiveInc = 0.1,
      synPermConnected = 0.1, # Connected threshold
      maxBoost = 3,
      seed = 1956, # The seed that Grok uses
      spVerbosity = 1,
      addNoise = False)
  
  viewer = SPViewer(sp,
                    screenWidth = 512,
                    screenHeight = 600,
                    imagePath = 'data/Image2.jpg',
                    replayDelay = .2)
  viewer.run()


if __name__ == "__main__":
  main()
  
