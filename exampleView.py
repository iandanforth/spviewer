#! /usr/bin/env python
'''
This script wraps the SP Viewer so that you can tweak parameters easily.

Things to explore:

- Various Images (Image1-3 provided)
- The effect of Boosting
- The effect of increment and decrement sizes
- Changing the overlap percent on the patches the SP sees (0.0 - 1.0)
'''

from sp_viewer import SPViewer
from nupic.research.spatial_pooler import SpatialPooler
import pygame

def main():
  
  # Instantiate our spatial pooler
  sp = SpatialPooler(
      inputDimensions = 32**2, # Size of image patch
      columnDimensions = 16, # Number of potential features
      potentialRadius = 10000, # Ensures 100% potential pool
      potentialPct = 1, # Neurons can connect to 100% of input
      globalInhibition = True,
      numActiveColumnsPerInhArea = 3, # Only one feature active at a time
      # All input activity can contribute to feature output
      stimulusThreshold = 0,
      synPermInactiveDec = 0.1,
      synPermActiveInc = 0.1,
      synPermConnected = 0.1, # Connected threshold
      maxBoost = 3,
      seed = 1956, # The seed that Grok uses
      spVerbosity = 1)
  
  viewer = SPViewer(sp,
                    screenWidth = 512,
                    screenHeight = 600,
                    imagePath = 'data/Image2.jpg',
                    patchSide = 32,
                    patchOverlapPercent = 0,
                    epochCount = 40,
                    replayDelay = .1)
  viewer.run()

  finalWindow = viewer.screen
  pygame.image.save(finalWindow, "screenshot.jpg")


if __name__ == "__main__":
  main()
  
