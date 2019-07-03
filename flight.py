#! /usr/bin/env python

"""
10/2016
J.D.Lyman

Calculates the F_light (flight) statistic for an image given a pixel location
and a segmentation map of the image. Will optionally output a distribution of
flight values with associated probabilities if an uncertainty of the location
is provided.
** All pixel indices should be given in FITS format (i.e. 1,1 is centre of
   lower-left pixel)**

$ ./flight.py -h

for help on command line arguments.

Nutshell:
- Image data is read from the specified FITS extension number
- A segmentation map is made for the image if one is not provided
  - The SExtractor configuration file SEx/flight.cfg can be altered to suit
    needs
  - If a different conv or nnw is required, replace these in the SEx dir
  - SEx/config will be written by flight.py and used for creating a seg map
- The seg map will be masked depending on the choice of method
  - Specific values for the seg map objects to be used
  - The seg map object underlying the location provided only
  - All non-zero pixels in the seg map that are directly touching the location
    provided - this makes no account of segmap values and just forms an 'island'
    around the location given (this is default)
  - Alternatively a segmask can be provided for custom jobs, should be zero for
    masked pixels and one for pixels to be used in flight calculations
- flight calculations are made on this seg mask array based on the pixel
  location chosen
- If a location uncertainty is given then flight probabilities are written to a
  text file also
- If plots are not turned off:
  - A heat map of flight values for all pixels is made overlaid on the raw
    image data
    - if ORIENTAT (degrees E of N of the y axis) is in the header, a compass
      can be plotted on the heatmap. Also can choose a colour map and a label.
    - markers can also be placed on the heatmap if desired
  - If a location uncertainty is given, a histogram of flight probabilities
    is created also

Known issues:
 - If the location uncertainty extends close to an image border, it will
   only calculate the probability within the image, thus the probabilities wont
   add to ~1
 - If the locuncert ellipse is very narrow compared to the size of a pixel, the
   calculated probabilities may be incorrect as the integration relies on
   scipy.integrate.quad, which can miss the peak if it is narrow. In the case of
   extremely small uncertainty in a single pixel, the script may output the
   probability of that pixel as 0.000 instead of 1.000, for example.

Plotting of NE compass was based on some code of pywcsgrid2 by leejjoon
(leejjoon.github.io/pywcsgrid2/)
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from distutils import spawn
from itertools import cycle

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.text import Text
import matplotlib.patheffects as PathEffects
from scipy import ndimage, integrate, stats

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)

# the file and directory path to this script, in case you call it from
# another directory, so relative paths to the script are still intact
FILEPATH = os.path.realpath(__file__)
FILEDIR = os.path.dirname(FILEPATH)

# locate the sex executable on the system, requires that it's in your PATH
SEX_EXEC = spawn.find_executable("sex")
if not SEX_EXEC:
    sys.exit("Couldn't locate `sex' executable in your PATH")

# store a few files names here for ease of alteration
SEGMAPSUFF = "_SEGMAP.fits"
CATSUFF = ".cat"
MASKSUFF = "_SEGMASK.fits"

# fraction of top/bottom pixel values to clip for plotting
CLIP = 0.05


class AnchoredCompass(AnchoredOffsetbox):
    """
    AnchoredOffsetbox class to create a NE compass on the plot given a value
    for ori (degrees E of N of the yaxis)
    """

    def __init__(self, ax, ori, loc=4, arrow_fraction=0.15, txt1="E", txt2="N",
                 pad=0.3, borderpad=0.5, prop=None, frameon=False):
        self._ax = ax
        self.ori = ori
        self._box = AuxTransformBox(ax.transData)
        self.arrow_fraction = arrow_fraction
        path_effects = [PathEffects.withStroke(linewidth=3, foreground="w")]
        kwargs = dict(mutation_scale=14,
                      shrinkA=0,
                      shrinkB=7)
        self.arrow1 = FancyArrowPatch(posA=(0, 0), posB=(1, 1),
                                      arrowstyle="-|>",
                                      arrow_transmuter=None,
                                      connectionstyle="arc3",
                                      connector=None, color="k",
                                      path_effects=path_effects,
                                      **kwargs)
        self.arrow2 = FancyArrowPatch(posA=(0, 0), posB=(1, 1),
                                      arrowstyle="-|>",
                                      arrow_transmuter=None,
                                      connectionstyle="arc3",
                                      connector=None, color="k",
                                      path_effects=path_effects,
                                      **kwargs)
        self.txt1 = Text(1, 1, txt1, rotation=0,
                         rotation_mode="anchor", path_effects=path_effects,
                         va="center", ha="center")
        self.txt2 = Text(2, 2, txt2, rotation=0,
                         rotation_mode="anchor", path_effects=path_effects,
                         va="center", ha="center")
        self._box.add_artist(self.arrow1)
        self._box.add_artist(self.arrow2)
        self._box.add_artist(self.txt1)
        self._box.add_artist(self.txt2)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)

    def _update_arrow(self, renderer):
        ax = self._ax

        x0, y0 = ax.viewLim.x0, ax.viewLim.y0
        a1, a2 = 180 - self.ori, 180 - self.ori - 90
        D = min(ax.viewLim.width, ax.viewLim.height)
        d = D * self.arrow_fraction
        x1, y1 = x0 + d * np.cos(a1 / 180. * np.pi), y0 + d * np.sin(a1 / 180. * np.pi)
        x2, y2 = x0 + d * np.cos(a2 / 180. * np.pi), y0 + d * np.sin(a2 / 180. * np.pi)
        self.arrow1.set_positions((x0, y0), (x1, y1))
        self.arrow2.set_positions((x0, y0), (x2, y2))
        d2 = d
        x1t, y1t = x0 + d2 * np.cos(a1 / 180. * np.pi), y0 + d2 * np.sin(a1 / 180. * np.pi)
        x2t, y2t = x0 + d2 * np.cos(a2 / 180. * np.pi), y0 + d2 * np.sin(a2 / 180. * np.pi)
        self.txt1.set_position((x1t, y1t))
        self.txt1.set_rotation(0)  # a1-180
        self.txt2.set_position((x2t + 20, y2t))
        self.txt2.set_rotation(0)  # a2-90

    def draw(self, renderer):
        self._update_arrow(renderer)
        super(AnchoredCompass, self).draw(renderer)


def runsex(image, chkimg_type="NONE", chkimg_name="check.fits"):
    """
    Runs SExtractor on the image based on the config file `SEx/flight.cfg`
    """
    param = os.path.join(FILEDIR, 'SEx/param')
    conv = os.path.join(FILEDIR, 'SEx/conv')
    nnw = os.path.join(FILEDIR, 'SEx/nnw')
    flightcfg = os.path.join(FILEDIR, 'SEx/flight.cfg')
    cat = 'flight.cat'
    config = os.path.join(FILEDIR, 'SEx/config')

    with open(config, 'w') as f:
        f.write(get_sex_string(flightcfg, cat, param, conv, nnw,
                               chkimg_type, chkimg_name))

    subprocess.Popen([SEX_EXEC, image, '-c', config],
                     stdout=open("/tmp/flight.sex", 'w'),
                     stderr=subprocess.STDOUT).wait()
    try:
        open(cat)
    except IOError:
        print('SExtractor didn\'t produce an object catalogue')
        print('SExtractor output:\n' + \
              open('/tmp/flightsex.tmp', 'r').read())
        raise Exception("SExtracting failed")

    shutil.move(cat, os.path.splitext(image)[0] + CATSUFF)


def get_sex_string(configsex, cat, param, conv, nnw, chkimg_type,
                   chkimg_name):
    with open(configsex) as cfgsex:
        configstring = cfgsex.read()
    return configstring.format(cat, param, conv, nnw, chkimg_type, chkimg_name)


class CalcFlight(object):
    """
    Calculate the F_light statistic (Fruchter et al. 2006) given a FITS file
    and a location. Optionally determine probabilities of F_light statistic
    with a location uncertainty. All coordinates should be passed in FITS
    format.

    Parameters
    ----------

    image : string
        Filepath to FITS file on which to perform analysis.
    loc : list
        Pixel location to calculate F_light value [x,y].
    locuncert : list, optional
        Uncertainty on `loc` in units of pixels [sig_x, sig_y]. This will
        activate probabilistic F_light. Defaults to `None`.
    sciext : int or str, optional
        The FITS extension of `image` containing the science data. Defaults to
        `SCI`.
    segmap :
        Filepath to an existing segmentation map of `image`. Defaults to `None`.
    segsect : `"t"`, `"u"` or list of ints, optional
        The method used to select objects in the segmntation map
        t = touching, the island of pixels under the location that including
                      only and all touching pixels
        u = underlying, the segmentation object underlying the location only
        [i,j,k...] = manual selection, the segmap numbers of objects to
                                       specifically use
        Defaults to `t`.
    segmask : str, optional
        Filepath to an existing segmentation mask of `image`. Defaults to
        `None`.
    savemask : bool, optional
        Save a copy of the segmenation mask with the filename
        `image`_SEGMASK.fits.  Defaults to `False`
    plots: bool, optional
        Output a heat map image of the F_light statistic with the location (or
        location uncertainty) marked. Defaults to `True`
    square_plot: bool, optional
        Force the heat map plot to be square. Defaults to `False`.
    cmap: str, optional
        The name of a matplotlib colormap to be used for the for the heat map.
        Defaults to `coolwarm_r`
    cbar: bool, optional
        Show a colour bar underneath the heatmap with the F_light
        values. Defaults to `True`.
    compass: float or bool, optional
        Show a NE compass on the heat map. If `True` will look for `ORIENTAT`
        keyword in the header. Otherwise specify as the degrees east of north
        of the y-axis. Defaults to `True`.
    label: str, optional
        Put a label of the object name on heat map. Defaults to `None`.
    linpix: float, optional
        The linear size of a pixel at the distance of the host. Defaults to
        `None`.
    linscale: float, optional
        The length of a linear scale bar to plot on heat map (linpix must be
        defined to use this). Defaults to `None`.
    scaleunit: str, optional
        Unit to include after scale (e.g. 'kpc', '"'), linpix and linscale
        should both be in this unit. Defaults to `None`.
    marker_loc: list of length-2 lists, optional
        The locations to place cosmetic markers on the heat map
        (e.g. [[1,2],[3,4]...]).  Defaults to `None`.
    marker_style: list, optional
        List of matplotlib marker styles to use as markers, these will be
        cycled through when plotting marker_loc entries. Defaults to
        `["+","x","1","d"]`.
    quiet: bool, optional
        Don't print anything unless something goes wrong. Defaults to `False`.
    """

    def __init__(self, image, loc, locuncert=None, sciext="SCI", segmap=None,
                 segsect='t', segmask=None, savemask=True, plots=True,
                 square_plot=False, cmap="coolwarm_r", cbar=True, compass=True,
                 label=None, linpix=None, linscale=None, scaleunit=None,
                 marker_loc=None, marker_style=["+", "x", "1", "d"], quiet=False):

        # grab image data from science extension of the image
        self.image = image
        try:
            self.imagearray, self.hdr = fits.getdata(image, sciext, header=True)
        except (IOError, IndexError):
            raise Exception("cannot get data/header from {0} (extension {1})"
                            .format(image, sciext))
        self.basename = os.path.splitext(image)[0]

        # convert from FITS indexing to python indexing
        self.xloc, self.yloc = loc[0] - 1, loc[1] - 1
        self.orig_location = loc

        shape = self.imagearray.shape
        if self.xloc > shape[1] or self.yloc > shape[0]:
            raise Exception("x,y location not within input image")

        self.segmap = segmap
        self.segmask = segmask
        self.segmap_arr = None
        self.segmask_arr = None

        self.segsect = segsect
        self.marker_loc = marker_loc
        self.marker_style = marker_style
        self.savemask = savemask
        self.plots = plots
        self.square_plot = square_plot
        self.cmap = cmap
        self.cbar = cbar
        self.compass = compass
        self.label = label
        self.linpix = linpix
        self.linscale = linscale
        self.scaleunit = scaleunit
        self.locuncert = locuncert
        self.quiet = quiet

    def main(self):
        if not self.quiet:
            print("running main")

        # make a segmentation map if we don't have one (and we don't have a
        # seg mask provided to us)
        if self.segmap is None and self.segmask is None:
            self.segmap = self.makesegmap()

        # make segmask from segmap
        if self.segmask is None:
            self.segmap_arr = fits.getdata(self.segmap)
            # if there isn't a segmask object at the pixel location, we can't
            # find (u)nderlying or (t)ouching sections so must exit here
            if (self.segmap_arr[int(self.yloc), int(self.xloc)] == 0
                    and self.segsect in ("t", "u")):
                print("no segmentation map object at pixel", end=' ')
                print("location {},{}".format(self.xloc, self.yloc))
                return
            self.segmask_arr = self.makesegmask()
        # or use the provided segmask
        else:
            self.segmask_arr = fits.getdata(self.segmask)
        # save the mask, if desired
        if self.savemask:
            fits.writeto(self.basename + MASKSUFF,
                         data=self.segmask_arr.astype('int'),
                         clobber=True)
        # safety check to make sure there are objects in the segmask
        if np.sum(self.segmask_arr) == 0:
            print("no segmentation map objects matching", end=' ')
            print("value(s) {}".format(", ".join(map(str, self.segsect))))
            return

        # calculate flight
        self.getflight()

        # calculate the probabilities of flight values if we have a location
        # uncertainty
        if self.locuncert is not None:
            self.getflight_prob()

        if self.plots:
            self.makeplots()

        if not self.quiet:
            print("finished main")
            print()
            print("Image:            ", self.image)
            print("Location:         ", self.orig_location)
            print("Pixel value:      ", self.locpixvalue)
            print("F_light value:     {:.3f}".format(self.flight))
            print("Weighted F_light ")
            print("     most likely:  {:.3f}".format(self.likely_flight))
            print("     with a prob:  {:.3f}".format(self.likely_prob))
            print()

    def makesegmap(self):
        if not self.quiet:
            print("making segmentation map")
        segmap = self.basename + SEGMAPSUFF
        runsex(self.image, chkimg_type="SEGMENTATION",
               chkimg_name=segmap)
        try:
            open(segmap)
        except IOError:
            raise Exception("segmentation map ({}) not created for some reason"
                            .format(segmap))
        else:
            return segmap

    def makesegmask(self):
        if not self.quiet:
            print("making segmentation mask")
        y, x = list(map(int, (self.xloc, self.yloc)))  # save transposing arrays

        if isinstance(self.segsect, str):
            if self.segsect[0] == "t":
                toucharray = ndimage.label(np.array(self.segmap_arr, float),
                                           structure=ndimage.generate_binary_structure(2, 2))[0]
                t_segmask_arr = (toucharray == toucharray[x, y])
                # if we have +i-j..etc designations, add/remove them from the
                # segmask as appropriate
                toadd = re.findall("(?<=\+)(\d+)", self.segsect)
                toremove = re.findall("(?<=\-)(\d+)", self.segsect)
                for a in toadd:
                    t_segmask_arr[np.where(self.segmap_arr == int(a))] = True
                for r in toremove:
                    t_segmask_arr[np.where(self.segmap_arr == int(r))] = False
                return t_segmask_arr

            segvals = []
            if self.segsect[0] == "u":
                segvals.append(self.segmap_arr[x, y])
                # if we have +i+j..etc designations, add them from the
                # segmask as appropriate, only adding makes sense when using u
                toadd = re.findall("(?<=\+)(\d+)", self.segsect)
                segvals.extend(list(map(int, toadd)))
        elif isinstance(self.segsect, (int, tuple, list)):
            segvals = self.segsect

        # return a mask of segmap that is True for pixels within objects
        # we're interested in
        return np.in1d(self.segmap_arr, segvals).reshape(self.segmap_arr.shape)
        # (in1d doesn't preserve shape...)

    def getflight(self):
        if not self.quiet:
            print("calculating f_light")

        # correcting axes swap for numpy
        self.locpixvalue = self.imagearray[int(self.yloc), int(self.xloc)]

        # make a copy of image array where pixels outside segmask are masked
        self.maskedarray = self.imagearray.copy()
        self.maskedarray[self.segmask_arr == 0] = 0

        # make a sorted 1d list of pixel values
        pixelvalues = self.maskedarray.copy().ravel()
        pixelvalues.sort()
        self.sortedpixelvalues = pixelvalues

        # create cumulative sum distribution normalised to total sum of pixels
        totalvalue = float(np.sum(pixelvalues))
        self.cumarray = np.cumsum(pixelvalues) / totalvalue

        # find the index of the sorted array corresponding to the location
        self.locpixnumber = np.where(pixelvalues == self.locpixvalue)[0]
        nsame = len(self.locpixnumber)
        if nsame > 1 and self.locpixvalue > 0:
            if not self.quiet:
                print("\t{} pixels with same value".format(nsame))
                print("\tsetting f_light to average for this pixel value")
            self.locpixnumber = self.locpixnumber[nsame / 2]
        elif nsame == 1:
            self.locpixnumber = self.locpixnumber[0]
        elif self.segmask_arr[int(self.yloc), int(self.xloc)] == 0:
            # i.e. we are not on the segmask
            self.locpixnumber = np.nan
            self.flight = 0
            return
        else:
            print("locpixvalue = {}".format(self.locpixvalue))
            print("pixelvalues = {}".format(pixelvalues))
            raise Exception("Couldn't find locpixvalue in pixelvalues")

        # find value of the cumulative sum at locpixnumber
        self.flight = self.cumarray[self.locpixnumber]

    def getflight_prob(self):
        if not self.quiet:
            print("calculating f_light probability distribution")

        # swap axes for numpy
        y, x = self.xloc, self.yloc
        ys, xs = self.locuncert

        # make and cutout a bounding box around x,y covering +/- 3 sigma
        bounds = (max(x - 3 * xs, 0),
                  min(math.ceil(x + 3 * xs), self.imagearray.shape[0]),
                  max(y - 3 * ys, 0),
                  min(math.ceil(y + 3 * ys), self.imagearray.shape[1]))
        minx, maxx, miny, maxy = list(map(int, bounds))

        pco = self.maskedarray[minx:maxx, miny:maxy]

        # calculated distance of the edges of pixels from our location
        x_dist, y_dist = (np.arange(pco.shape[0]) - x + minx,
                          np.arange(pco.shape[1]) - y + miny)
        x_probs, y_probs = [], []
        # integrate from these distances to distance+1 (i.e. 1 pixel) over the
        # gaussian
        for xl in x_dist:
            x_probs.append(integrate.quad(stats.norm.pdf, xl, xl + 1,
                                          args=(0, xs))[0])
        for yb in y_dist:
            y_probs.append(integrate.quad(stats.norm.pdf, yb, yb + 1,
                                          args=(0, ys))[0])
        # multiply the probabilities of the 1D gaussians to make a 2D pdf
        yy, xx = np.meshgrid(y_probs, x_probs)
        p = xx * yy

        # convert the maskedaraay to one with flight values
        sortindices = np.searchsorted(self.sortedpixelvalues, self.maskedarray)
        pco_flight_array = self.cumarray[sortindices][minx:maxx, miny:maxy]

        # clip flights to 0->1 so that all non-segmask location count to the
        # Flight=0 bin
        pco_flight_array = np.clip(pco_flight_array, a_min=0, a_max=1)

        # define unique flight values and bin the probabilities of each
        f_rav = pco_flight_array.ravel()
        p_rav = p.ravel()
        self.uniq_fl, self.inverse_fl = np.unique(f_rav, return_inverse=True)
        self.total_probs = np.bincount(self.inverse_fl, p_rav)

        # save the one with the highest probability to out put to terminal
        self.likely_prob = np.max(self.total_probs)
        self.likely_flight = self.uniq_fl[np.argmax(self.total_probs)]

        # write a text file with all the possible flights and their associated
        # probabilities
        with open(self.basename + "_flprob.txt", "w") as f:
            f.write("# Flight Probability\n")
            for val, prob in zip(self.uniq_fl, self.total_probs):
                f.write("{:5.3f}    {:5.3f}\n".format(val, prob))

    def makeplots(self):
        if not self.quiet:
            print("making plots")
        # Heatmap plot:
        plt.clf()
        # clip top/bottom pixel values for plotting
        sortedarray = np.sort(self.imagearray.ravel())
        npix = len(sortedarray)
        cutpix = int(math.ceil(CLIP * npix))
        low_cut = sortedarray[cutpix]
        high_cut = sortedarray[npix - cutpix]

        # trim the image array to just enclose all the non-zero pixels from the
        # segmentation mask. Add a 1pixel border for good measure
        z = np.where(self.segmask_arr > 0)
        # append the pixel chosen to include in map
        z = (np.append(z[0], self.yloc), np.append(z[1], self.xloc))
        # calculate min/max including locuncert if given
        if self.locuncert is not None:
            sigx, sigy = self.locuncert
            minx, maxx, miny, maxy = list(map(int,
                                              (min(min(z[0]), self.yloc - sigy) - 1,
                                               max(max(z[0]), self.yloc + sigy) + 2,
                                               min(min(z[1]), self.xloc - sigx) - 1,
                                               max(max(z[1]), self.xloc + sigx) + 2)
                                              ))
        else:
            minx, maxx, miny, maxy = (min(z[0]) - 1, max(z[0]) + 2,
                                      min(z[1]) - 1, max(z[1]) + 2)
        # if we want a square plot, then expand the limits of the shortest side
        # to make it square by padding the 1/2 the difference in each direction
        if self.square_plot:
            xsize = maxx - minx
            ysize = maxy - miny
            if xsize < ysize:
                minx -= int(math.floor(float(ysize - xsize) / 2))
                maxx += int(math.ceil(float(ysize - xsize) / 2))
            if xsize > ysize:
                miny -= int(math.floor(float(xsize - ysize) / 2))
                maxy += int(math.ceil(float(xsize - ysize) / 2))
        # we need to make sure the limits are within the image boundaries
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, self.imagearray.shape[0])
        maxy = min(maxy, self.imagearray.shape[1])

        # apply the cuts and bounding box to image array
        imageplotarray = np.clip(self.imagearray,
                                 low_cut, high_cut)[minx:maxx, miny:maxy]
        plt.imshow(imageplotarray, origin="lower", interpolation="nearest",
                   cmap="Greys")
        # create the analagous array but with flight values and mask zeros
        sortindices = np.searchsorted(self.sortedpixelvalues, self.maskedarray)
        heatplotarray = self.cumarray[sortindices]
        heatplotarray = np.ma.masked_where(self.maskedarray == 0,
                                           heatplotarray)[minx:maxx, miny:maxy]
        plt.imshow(heatplotarray, origin="lower", interpolation="nearest",
                   cmap=self.cmap)
        self.heatplotarray = heatplotarray
        ax1 = plt.axes()
        ax1.autoscale(False)
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        # add various annotations as required
        if self.label is not None:
            ax1.text(0.05, 0.92, "\\textbf{" + self.label + "}", ha="left", va="center",
                     transform=ax1.transAxes, size=30, weight="bold", color="k",
                     path_effects=[PathEffects.withStroke(linewidth=3,
                                                          foreground="w")])
        if self.compass is not False:
            if isinstance(self.compass, bool):
                try:
                    orientat = self.hdr["ORIENTAT"]
                except KeyError:
                    print("\tORIENTAT not in FITS header, skipping N-E plotting")
                    orientat = False
            elif isinstance(self.compass, (int, float)):
                orientat = self.compass
            else:
                print("\tbad type of `compass`, %s. Must be bool, int or float" \
                      % type(self.compass))
                print("\tskipping N-E plotting")
                orientat = False
            if orientat is not False:
                compass = AnchoredCompass(ax1, ori=orientat)
                ax1.add_artist(compass)
        if not None in (self.linpix, self.linscale):
            # no. pixels scale line needs to be
            npix = self.linscale / float(self.linpix)
            # length of line/axis
            plotx = (npix / self.heatplotarray.shape[1])
            ax1.plot((0.05, 0.05 + plotx), (0.05, 0.05), c="k", lw=2,
                     transform=ax1.transAxes,
                     path_effects=[PathEffects.withStroke(linewidth=4,
                                                          foreground="w")])
            ax1.text(0.05, 0.1, "%s %s" % (self.linscale, self.scaleunit),
                     ha="left", va="center", transform=ax1.transAxes,
                     size=30, color="k",
                     path_effects=[PathEffects.withStroke(linewidth=3,
                                                          foreground="w")])
        if self.marker_loc is not None:
            for loc, style in zip(self.marker_loc, cycle(self.marker_style)):
                ax1.plot(loc[0] - miny - 1, loc[1] - minx - 1, marker=style, markersize=20,
                         c="k", markerfacecolor="none", markeredgewidth=2)

        if self.locuncert is not None:
            width = 2 * self.locuncert[0]
            height = 2 * self.locuncert[1]
            yaxes = (self.yloc - minx)
            xaxes = (self.xloc - miny)
            ax1.add_artist(Ellipse((xaxes, yaxes), width, height,
                                   facecolor="none", edgecolor="k", lw=2,
                                   ls="dashed"))
        else:
            self.starmark = ax1.plot(self.xloc - miny, self.yloc - minx,
                                     marker="*", markersize=20, c="k",
                                     markerfacecolor="none", markeredgewidth=2)
        if self.cbar:
            color_bar = plt.colorbar(orientation='horizontal')
            cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
            plt.setp(cbytick_obj, color='k')
        plt.savefig(self.basename + "_flheat.eps", bbox_inches="tight")

        # make Flight probability histogram
        if self.locuncert is not None:
            plt.clf()
            ax2 = plt.axes()
            ax2.set_aspect(0.3333)
            plt.hist(self.uniq_fl, bins=20, range=(0, 1),
                     weights=self.total_probs, color="teal")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            if self.label is not None:
                ax2.text(0.03, 0.87, "\\textbf{" + self.label + "}", ha="left",
                         va="center", transform=ax2.transAxes, size=20,
                         weight="bold", color="k",
                         path_effects=[PathEffects.withStroke(linewidth=3,
                                                              foreground="w")])
            plt.xlabel("F$_\\textrm{light}$")
            plt.ylabel("Probability")
            plt.savefig(self.basename + "_flprob.eps", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform F_light analysis on a FITS file. If file paths for a seg map or mask are not provided'
                    ' they will be created running SExtractor. Coordinates should be given in FITS format.',
    )

    parser.add_argument(
        'image',
        type=str,
        help='FITS file for analysis',
    )
    parser.add_argument(
        'loc',
        type=str,
        help='image pixel coordinates of object in format `x,y`',
    )
    parser.add_argument(
        '--locuncert',
        type=str,
        default=None,
        help='the uncertainty in the specified location, given in units of pixels in the form `sigx,sigy`. The flight'
             'value will be calculated from a distribution based on this uncertainty ellipse (default: None)',
    )
    parser.add_argument(
        '-e',
        '--sciext',
        default='SCI',
        help='extension name or number of data to perform analysis on in `image`. (default: "SCI")',
    )
    parser.add_argument(
        '-s',
        '--segmap',
        type=str,
        default=None,
        help='name of segmentation map from sextractor to use. (default: None)',
    )
    parser.add_argument(
        '-m',
        '--segsect',
        type=str,
        default='t',
        help='method of selecting segmap objects to use. `t`: use all non-zero segmap pixels touching the location '
             'chosen, regardless of segmap number. `u`: only use the segmap object underlying directly the location '
             'chosen. i,j,k...: a single or comma-separated list of integer values of the segmap objects to use. to '
             'combine methods use, for e.g. t+i-j.., where the method `t` be used in addition to selecting pixels from '
             'segmap object i and removing pixels from segmap object j. (default: `t`)',
    )
    parser.add_argument(
        '-k',
        '--segmask',
        type=str,
        default=None,
        help='name of segmentation mask to use. (default: None)',
    )
    parser.add_argument(
        '--savemask',
        action='store_true',
        help='save a copy of the mask, with suffix {}'.format(MASKSUFF),
    )
    parser.add_argument(
        '-p',
        '--noplots',
        dest='plots',
        action='store_false',
        help='turn off plotting of results',
    )
    parser.add_argument(
        '--square_plot',
        action='store_true',
        help='force heat map to be square',
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='coolwarm_r',
        help='colormap to use for heat map plot. (default:`coolwarm_r`',
    )
    parser.add_argument(
        '--nocbar',
        dest='cbar',
        action='store_false',
        help='do not display a colour bar legend on heat map',
    )
    parser.add_argument(
        '--nocompass',
        dest='compass',
        action='store_false',
        help='do not display a NE compass on heat map, need a valid ORIENTAT (yaxis as deg E of N) header',
    )
    parser.add_argument(
        '--compassdeg',
        dest='compassdeg',
        type=float,
        default=None,
        help='the degrees east of north that the y-axis is, use if there is not ORIENTAT header',
    )
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help='string to print as label on heat map (default: None)',
    )
    parser.add_argument(
        '--linpix',
        type=float,
        default=None,
        help='linear size of a pixel at the distance of the host (default: None)',
    )
    parser.add_argument(
        '--linscale',
        type=float,
        default=None,
        help='size of linear scale bar to plot on the heat map - linpix must be defined and this scale will be in the'
             ' the same units (default: None)',
    )
    parser.add_argument(
        '--scaleunit',
        type=str,
        default="",
        help='unit to include after scale (e.g. "kpc", \"), linpix and  linscale should both be in this '
             'unit (default: "")',
    )
    parser.add_argument(
        '--marker_loc',
        type=str,
        default=None,
        help='additional locations to mark on the heatmap of the form `x1,y1,x2,y2..xn,yn` (default: None)',
    )
    parser.add_argument(
        '--marker_style',
        type=str,
        default="+,x,1,d",
        help='comma-separated matplotlib marker styles for marker_loc (default: "+,x,1,d")',
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='run silently (unless something goes wrong)',
    )
    args = parser.parse_args()

    splitloc = args.loc.split(",")
    if len(splitloc) != 2:
        print("location should be zero-indexed coordinates of the format `x,y`")
        sys.exit(2)
    else:
        try:
            args.loc = list(map(float, splitloc))
        except ValueError:
            print("location needs to be x,y pixel coordinates, numeric only")
            sys.exit(2)
    if args.locuncert is not None:
        splitlocuncert = args.locuncert.split(",")
        if len(splitlocuncert) != 2:
            print("locuncert should be of the format `sigx,sigy`")
            sys.exit(2)
        else:
            try:
                args.locuncert = list(map(float, splitlocuncert))
            except ValueError:
                print("locuncert needs to be `sigx,sigy`, numeric only")
                sys.exit(2)

    if args.segsect != "u" and args.segsect[0] != "t":
        splitsect = args.segsect.split(",")
        try:
            args.segsect = list(map(int, splitsect))
        except ValueError:
            print("segsect must be comma-separated list of integer values if", end=' ')
            print("defining sections to use")
            sys.exit(2)

    try:
        args.sciext = int(args.sciext)
    except ValueError:
        pass

    if args.segmap is not None:
        try:
            fits.open(args.segmap)
        except IOError:
            print("Couldn't open {} with fits, make sure it exists and is" \
                  .format(args.segmap), end=' ')
            print("a valid FITS file")
            sys.exit(2)
    if args.segmask is not None:
        try:
            fits.open(args.segmask)
        except IOError:
            print("Couldn't open {} with fits, make sure it exists and is" \
                  .format(args.segmask), end=' ')
            print("a valid FITS file")
            sys.exit(2)
    if args.marker_loc is not None:
        splitmarker = args.marker_loc.split(",")
        if len(splitmarker) % 2 != 0:
            print("marker_loc should be pairs of coordinates `x1,y1,x2,y2..`")
            sys.exit(2)
        else:
            try:
                l = list(map(float, splitmarker))
                args.marker_loc = [l[i:i + 2] for i in range(0, len(l), 2)]
            except ValueError:
                print("marker_loc needs to be `x1,y1,x2,y2..` only numeric")
                sys.exit(2)
    if args.marker_style is not None:
        args.marker_style = args.marker_style.split(",")

    if args.compassdeg:
        args.compass = args.compassdeg
    del args.compassdeg

    arg_dict = vars(args)  # make dictionary of arguments
    f = CalcFlight(**arg_dict)  # pass these to construct flight class
    f.main()  # run the class
