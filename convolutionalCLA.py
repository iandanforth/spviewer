#! /usr/bin/env python

import numpy
import math
import pygame
import time

from PIL import Image
from pygame.color import THECOLORS
from copy import copy
from nupic.research.spatial_pooler import SpatialPooler

DEBUG = 0


class SpatialPoolerVisualization(object):
  
  def __init__(self):
    
    # Start up our display
    pygame.init()
    
    # Set up our screen
    size = self.screenWidth, self.screenHeight = 512, 512
    self.screen = pygame.display.set_mode(size)
    
    # Start with a blank white canvas
    self.screen.fill(THECOLORS['white'])
    
    # Define what counts as a connected synapse
    self.synPermConnected = 0.1

  def run(self):
    
    # Display the input image we'll be learning on
    imageName = 'Image3.jpg'
    inputImage = pygame.image.load(imageName).convert()
    
    # Far left and centered vertically
    iiX = 0
    iiY = (.5 * self.screenHeight) - (.5 * inputImage.get_height())
    self.screen.blit(inputImage, (iiX, iiY))
    
    # Display an outer bounding box
    self._drawBoundingBox(inputImage, iiX, iiY)
    
    # Get some image patches on which to train
    patchSide = 32
    patchOverlapPercent = 0.0
    imagePatches = self._getPatchesFromImage(imageName,
                                       patchSide,
                                       patchOverlapPercent)
    # Convert those to bit vectors for input into CLA
    vectorPatches = [self._convertToVector(patch[0]) for patch in imagePatches]
    inputVectorLength = patchSide**2

    # An array to store the Activity state of the neurons
    featuresCount = 16
    activeArray = numpy.zeros(featuresCount)
    
    # Instantiate our spatial pooler
    sp = SpatialPooler(
        inputDimensions = inputVectorLength, # Size of image patch
        columnDimensions = featuresCount, # Number of potential features
        potentialRadius = 10000, # Ensures 100% potential pool
        potentialPct = 1, # Neurons can connect to 100% of input
        globalInhibition = True,
        numActiveColumnsPerInhArea = 1, # Only one feature active at a time
        # All input activity can contribute to feature output
        stimulusThreshold = 0,
        synPermInactiveDec = 0.01,
        synPermActiveInc = 0.1,
        synPermConnected = self.synPermConnected, # Connected threshold
        maxBoost = 3, # Turn off boosting
        seed = 1956, # The seed that Grok uses
        spVerbosity = 1,
        addNoise = False)
    
    # Draw permanences before any input or learning
    self._drawPermanences(sp, patchSide, featuresCount)
    
    # Feed in data and visualize the evolution of the permanences
    epochCount = 40
    replaySpeed = 0
    for i in range(1, epochCount + 1):
      
      print "Epoch:", i
      columnEpochHistory = []
      for j, patch in enumerate(vectorPatches):
        # Display our sliding window
        imagePatch, patchDimensions = imagePatches[j]
        self._drawViewBox(patchDimensions, iiX, iiY)
        
        # Show the patch CLA sees in a given iteration
        # Position it near the middle and centered vertically
        patchX = 160
        patchY = (.5 * self.screenHeight) - (.5 * imagePatch.size[1])
        self._drawPatch(imagePatch, patchX, patchY)
        
        # Redraw base input image
        self.screen.blit(inputImage, (iiX, iiY))

        # Update the network
        sp.compute(patch, True, activeArray)
        
        # Draw column activations
        self._drawColumnActivity(activeArray)
        
        # Store those activations to later generate feature maps
        columnEpochHistory.append(copy(activeArray))
        
        # Slow things down for viewing
        time.sleep(replaySpeed)
      
      # Display our perms after each epoch
      self._drawPermanences(sp, patchSide, featuresCount)
      
      # Draw feature maps
      self._drawFeatureMaps(columnEpochHistory)


  def _convertPILImageToPygameSurface(self, image):
    '''
    Returns a Pygame Surface instance built using data from a PIL Image
    '''
    
    mode = image.mode
    size = image.size
    data = image.tostring()
    surf = pygame.image.frombuffer(data, size, mode)
    
    return surf

  
  def _convertToImage(self, listData, mode = '1'):
    '''
    Takes in a list and returns a new square image
    '''
  
    # Assume we're getting a square image patch
    side = int(len(listData) ** 0.5)
    # Create the new image of the right size
    im = Image.new(mode, (side, side))
    # Put the data into that patch
    im.putdata(listData)

    return im

  
  def _convertToVector(self, image):
    '''
    Returns a bit vector representation (list of ints) of a PIL image.
    '''
    # Convert the image to black and white
    image = image.convert('1')
    # Pull out the data, turn that into a list, then a numpy array,
    # then convert from 0 255 space to binary with a threshold.
    # Finnally cast the values into a type CPP likes
    vector = (numpy.array(list(image.getdata())) < 100).astype('uint32')
    
    return vector


  def _coordsToRect(self, coords):
    '''
    Returns a pygame Rect
    '''
    
    left = coords[0]
    top = coords[1]
    width = coords[2] - left
    height = coords[3] - top
    
    return pygame.Rect(left, top, width, height)


  def _drawBoundingBox(self, image, x, y):
    '''
    Draws a Pygame.rect to screen that is a 1 pixel black box around the
    given dimensions.
    '''
    color = THECOLORS['black']
    boxDimensions = (x - 1,
                     y - 1,
                     x + image.get_width() + 2,
                     y + image.get_height() + 2)
    rect = self._coordsToRect(boxDimensions)
    width = 1
    pygame.draw.rect(self.screen, color, rect, width)


  def _drawPatch(self, im, x, y):
    '''
    Draws a patch to screen and updates the display
    
      patch - a PIL image object
      x, y - coords of where to draw the patch on screen
    '''
    
    mode = im.mode
    size = im.size
    data = im.tostring()
    im = pygame.image.frombuffer(data, size, mode)
        
    # Draw in the background
    self.screen.blit(im, (x, y))

    # Display an outer bounding box
    self._drawBoundingBox(im, x, y)

    # Update the screen
    pygame.display.flip()


  def _drawPermanences(self, sp, patchSide, featuresCount):
    
    for i in range(featuresCount):
      perms = sp._permanences.getRow(i)
      # Convert perms to RGB (effective grayscale) values
      allPerms = [(v, v, v) for v in ((1 - perms) * 255).astype('int')]
      
      connectedPerms = perms >= self.synPermConnected
      connectedPerms = (numpy.invert(connectedPerms) * 255).astype('int')
      connectedPerms = [(v, v, v) for v in connectedPerms]
      
      allPermsReconstruction = self._convertToImage(allPerms, 'RGB')
      connectedReconstruction = self._convertToImage(connectedPerms, 'RGB')
      size = allPermsReconstruction.size
      
      # Convert that to a format that Pygame can use
      pRSurface = self._convertPILImageToPygameSurface(allPermsReconstruction)
      cSSurface = self._convertPILImageToPygameSurface(connectedReconstruction)

      
      # Define where we'll draw that on the screen
      xOffset = 272
      yOffSet = (.5 * self.screenHeight) - (.5 * (featuresCount * size[1]))
      
      # Line
      x = xOffset
      x2 = x + 64
      y = yOffSet + i * patchSide
      
      # Square
      #x = (i % 4 * patchSide) + xOffset
      #y = math.floor( i / 4 ) * patchSide
      
      
      # Draw in the background
      self.screen.blit(pRSurface, (x, y))
      self.screen.blit(cSSurface, (x2, y))

  
  def _drawColumnActivity(self, columnActivity):
    
    # How large a square we want to represent a column
    columnVizSize = 16
    
    totalHeight = columnVizSize * len(columnActivity)
    vertOffset = (self.screenHeight * .5) - (.5 * totalHeight)
    
    for i, value in enumerate(columnActivity):
      
      color = THECOLORS['black']
      x1 = 224
      y1 = vertOffset + (i * columnVizSize)
      x2 = x1 + columnVizSize
      y2 = y1 + columnVizSize
      
      dimensions = (x1, y1, x2, y2)
      rect = self._coordsToRect(dimensions)
      if value:
        width = 0
      else:
        width = 1
      # Clear
      pygame.draw.rect(self.screen, THECOLORS['white'], rect, 0)
      # Redraw
      pygame.draw.rect(self.screen, color, rect, width)


  def _drawFeatureMaps(self, columnEpochHistory):
    '''
    Draws a feature map per column for the previous epoch
    '''
    
    mapSide = len(columnEpochHistory) ** .5
    scaleFactor = 32 / mapSide
    mapSide = int(mapSide * scaleFactor)

    # Create maps
    featureMaps = []
    columnsHistory = zip(*columnEpochHistory)
    for columnHistory in columnsHistory:
      cH = numpy.array(columnHistory)
      cH = [(v, v, v) for v in ((1-cH) * 255).astype('int')]
      mapImage = self._convertToImage(cH, 'RGB')
      largeMapImage = mapImage.resize((mapSide, mapSide))
      featureMaps.append(largeMapImage)
      
    # Draw
    for i, fMap in enumerate(featureMaps):
      
      # Define where we'll draw that on the screen
      xOffset = 400
      yOffSet = (.5 * self.screenHeight) - (.5 * (len(featureMaps) * mapSide))
      
      # Line
      x = xOffset
      y = yOffSet + i * mapSide
      
      # Square
      #x = (i % 4 * patchSide) + xOffset
      #y = math.floor( i / 4 ) * patchSide
      
      # Draw in the background
      self.screen.blit(self._convertPILImageToPygameSurface(fMap), (x, y))
      
      # Display an outer bounding box
      color = THECOLORS['black']
      dimensions = (x-1, y-1, x+mapSide+2, y+mapSide+2)
      rect = self._coordsToRect(dimensions)
      width = 1
      pygame.draw.rect(self.screen, color, rect, width)


  def _drawViewBox(self, patchDimensions, baseX, baseY):
    '''
    Draws a rect to the screen in the same location as patch
    '''
    color = THECOLORS['black']
    dimensions = copy(patchDimensions)
    dimensions[1] += baseY
    dimensions[3] += baseY
    rect = self._coordsToRect(dimensions)
    width = 1
    pygame.draw.rect(self.screen, color, rect, width)


  def _getPatchesFromImage(self,
                          imageName,
                          patchSide = 32 ,
                          overlap = 0.0):
    '''
    Returns a list of lists representing bit vector patches of imageName
    '''
    
    # Prevent infinite loop
    assert overlap < 1
    
    # Open the training image
    inputImage = Image.open(imageName)
    if DEBUG == 1:
      inputImage.show()
    
    # Get its dimensions
    _, _, imageWidth, imageHeight = inputImage.getbbox()
    if DEBUG == 1:
      print imageWidth, imageHeight
    
    # Define the size of our patch
    x1 = 0
    y1 = 0
    x2 = patchSide
    y2 = patchSide
    
    # Divide our image into patches
    patches = []
    counter = 0
    # Loop over each row of imageHeight patchSide
    while y2 <= imageHeight:
      x1 = 0
      x2 = patchSide
      # Loop over each column of imageWidth patchSide
      while x2 <= imageWidth:
        # Get our patch and then update the coords for the next loop
        target = [x1, y1, x2, y2]
        if DEBUG == 1:
          print target
        patch = inputImage.crop(target)
        patches.append([patch, target])
        # Increment our counter
        counter += 1
        if DEBUG == 1:
          print 'This is input pattern %d' % counter
          print patch
          patch.show()

        # Move the patch over by a percent to allow for overlap of patches
        move = 1 - overlap
        move = int(math.floor(patchSide * move))
        x1 += move
        x2 += move
    
      # Move the patch down
      y1 += move
      y2 += move

    return patches



if __name__ == "__main__":
  SpatialPoolerVisualization().run()
