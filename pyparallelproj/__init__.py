# this is needed to get the package version at runtime
from pkg_resources import DistributionNotFound, get_distribution

from .pet_scanners import RegularPolygonPETScanner
from .phantoms import brain2d_phantom, ellipse2d_phantom
from .projectors import Projector
from .sinogram import PETSinogramParameters

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass