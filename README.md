spviewer
========

A very simple, visual example of the Spatial Pooler algorithm in operation.

Modifying the parameters set in exampleView.py can help you explore how
the spatial pooler works.

Caveats

- Currently this uses the new python implementation of the SP, which is not the
  default
- All implementations of the SP add noise to the output of the SP which can make
  things confusing. An item on my TODO list is to add an option to turn this
  noise off.
- The way the SP is used in the example is meant to replicate how convolutional
  neural networks are used. This is not how other examples like HotGym work, but
  works well for image data, and makes the SP comparable to more mainline work.


Requirements

- NuPIC
- PIL
- Pygame

Use:

python exampleView.py
