#! /usr/bin/env python

"""
10/2014
J.D.Lyman@warwick.ac.uk

Calculates the F_light (flight) statistic for an image given a pixel location 
and a segmentation map of the image.
** The pixel location should be zero-indexed! i.e. bottom left pixel of a FITS
   image is location 0,0**

$ ./flight.py -h

for help on command line arguments.

Nutshell:
- Image data is read from the specified FITS extension number
- A segmentation map is made for the image if one is not provided
  - The SExtractor configuration file SEx/flight.cfg can be altered to suit
    needs
  - If a different conv or nnw is required, replace these in the SEx dir
  - SEx/config will be written by Flight.py and used for creating a seg map
- The seg map will be masked depending on the choice of method
  - Specific values for the seg map objects to be used
  - The seg map object underlying the location provided only
  - All non-zero pixels in the seg map that are directly touching the location
    provided - this makes no account of segmap values and just forms an 'island'
    around the location given (this is default)
  - Alternatively a segmask can be provided for custom jobs, should be zero for
    masked pixels and one for pixels to be used in flight calculations
- The image array is trimmed to just enclose the unmasked pixels
- flight calculations are made on this trimmed array based on the pixel location
  chosen
- If plots are not turned off:
  - The form of the cumulative sum with the selected pixel marked is plotted
  - A heat map of flight values for all pixels is made overlaid on the raw
    image data
    - if ORIENTAT (degrees E of N of the y axis) is in the header, a compass
      can be plotted on the heatmap. Also can choose a colour map and a label.
    - a galaxy centre and brightest pixel can be marked on the FLight plot

Plotting of NE compass was based on some code of pywcsgrid2 by leejjoon
(leejjoon.github.io/pywcsgrid2/)
"""

#TODO when givin an astrometric uncertainty (locuncert > 0) then provide a
#     weighted Flight value + range

import os,sys
import argparse
import re
import math
import subprocess
import shutil
import warnings
from distutils import spawn

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.text import Text
import matplotlib.patheffects as PathEffects

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman'],'size':18})
rc('text', usetex=True)

try:
    import pyds9
except ImportError:
    HASPYDS9 = False
else:
    HASPYDS9 = True

# stop pyfits bitching about overwriting files #FIXME Needed?
warnings.resetwarnings()
warnings.filterwarnings('ignore', category=UserWarning, append=True)

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
    def __init__(self, ax, ori, loc=4, arrow_fraction=0.15,txt1="E", txt2="N",
                 pad=0.1, borderpad=0.5, prop=None, frameon=False):
        self._ax = ax
        self.ori = ori
        self._box = AuxTransformBox(ax.transData)
        self.arrow_fraction = arrow_fraction
        path_effects = [PathEffects.withStroke(linewidth=3,foreground="w")]
        kwargs = dict(mutation_scale=14,
                      shrinkA=0,
                      shrinkB=7)
        self.arrow1 = FancyArrowPatch(posA=(0, 0), posB=(1, 1),
                                      arrowstyle="-|>",
                                      arrow_transmuter=None,
                                      connectionstyle="arc3",
                                      connector=None,color="k",
                                      path_effects=path_effects,
                                      **kwargs)
        self.arrow2 = FancyArrowPatch(posA=(0, 0), posB=(1, 1),
                                      arrowstyle="-|>",
                                      arrow_transmuter=None,
                                      connectionstyle="arc3",
                                      connector=None,color="k",
                                      path_effects=path_effects,
                                      **kwargs)
        self.txt1 = Text(1,1, txt1, rotation=0,
                         rotation_mode="anchor",path_effects=path_effects,
                         va="center", ha="center")
        self.txt2 = Text(2,2, txt2, rotation=0,
                         rotation_mode="anchor",path_effects=path_effects,
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
        a1, a2 = 180-self.ori,180-self.ori-90
        D = min(ax.viewLim.width, ax.viewLim.height)
        d = D * self.arrow_fraction
        x1, y1 = x0+d*np.cos(a1/180.*np.pi), y0+d*np.sin(a1/180.*np.pi)
        x2, y2 = x0+d*np.cos(a2/180.*np.pi), y0+d*np.sin(a2/180.*np.pi)
        self.arrow1.set_positions((x0, y0), (x1, y1))
        self.arrow2.set_positions((x0, y0), (x2, y2))
        d2 = d
        x1t, y1t = x0+d2*np.cos(a1/180.*np.pi), y0+d2*np.sin(a1/180.*np.pi)
        x2t, y2t = x0+d2*np.cos(a2/180.*np.pi), y0+d2*np.sin(a2/180.*np.pi)
        self.txt1.set_position((x1t, y1t))
        self.txt1.set_rotation(0) # a1-180
        self.txt2.set_position((x2t, y2t))
        self.txt2.set_rotation(0) # a2-90
                                    

    def draw(self, renderer):
        self._update_arrow(renderer)
        super(AnchoredCompass, self).draw(renderer)

def runsex(image,chkimg_type="NONE",chkimg_name="check.fits"):
    param = os.path.join(FILEDIR,'SEx/param')
    conv = os.path.join(FILEDIR,'SEx/conv')
    nnw = os.path.join(FILEDIR,'SEx/nnw')
    flightcfg = os.path.join(FILEDIR,'SEx/flight.cfg')
    cat = 'flight.cat'
    config = os.path.join(FILEDIR,'SEx/config')

    with open(config,'w') as f:
        f.write(get_sex_string(flightcfg,cat,param,conv,nnw,
                           chkimg_type,chkimg_name))

    subprocess.Popen([SEX_EXEC,image,'-c',config],
                     stdout=open("/tmp/flight.sex",'w'),
                     stderr=subprocess.STDOUT).wait()
    try:
        open(cat)
    except IOError:
        print 'SExtractor didn\'t produce an object catalogue'
        print 'SExtractor output:\n'+\
                    open('/tmp/flightsex.tmp','r').read()
        raise Exception("SExtracting failed")

    shutil.move(cat,os.path.splitext(image)[0]+CATSUFF)

def get_sex_string(configsex,cat,param,conv,nnw,chkimg_type,
                   chkimg_name):
    with open(configsex) as cfgsex:
        configstring = cfgsex.read()
    return configstring.format(cat,param,conv,nnw,chkimg_type,chkimg_name)


class calcflight(object):
    """
    image: filepath to fits file on which to perform analysis
    location: [x,y] list of pixel location to calculate Flight for -- 0-indexed!
    segmap: filepath to a segmentation map to use for the image
    segsect: t = touching: the collection of pixels under the location that are
                 highlighted in seg map including only and all touching 
                 detections (i.e. no separated islands of pixels),
                 - add structure = ndimage.generate_binary_structure(2,2) as
                 argument to ndimage.label
             u = underlying: the object underlying the location only
             i = interactive: use ds9 to select seg nums requires pyds9
                 (i,j,k...) = numbers: the segmap numbers of objects to 
                 specifically use
    segmask: filepath to a segmentation mask to use for the image
    sciext: the extension of the file containing the science data
    centre: zero-indexed coordinates of the barycentre of segmap, this will
            only be used to mark it with a '+' on the output heatmap, [x,y]
    brightest: same as `centre` except will be marked with a 'X'
    savemask: save a copy of the segmask with suffix _SEGMASK.fits
    plots: output a heat map image of the Flight statistic with the location 
           marked
    square_plot: force the heat map to be square to make tessalting figures
                 better
    cmap: the name of the 'colormap' to be used by matplotlib for the heat map
    cbar: show a colour bar underneath the heatmap with the Flight values
    compass: show a NE compass on the heat map, if True will look for ORIENTAT
             keyword in header, otherwise specify as the degrees east of north
             the y-axis is
    label: plot a label of the object on heat map
    linpix: the linear size of a pixel at the distance of the host
    linscale: length of linear scale bar to plot on heat map (linpix must
              be defined to use this)
    scaleunit: unit to include after scale (e.g. 'kpc', '"'), linpix and 
               linscale should both be in this unit
    locuncert: the uncertainty in the specified location, given in units of
               pixels. Instead of flagging a pixel, it will draw a circle 
               with radius locuncert on the heatmap
    quiet: don't print anything unless something goes wrong
    """
    def __init__(self,image,location,segmap=None,segsect='t',segmask=None,
                 sciext="SCI",centre=None,brightest=None,savemask=True,
                 plots=True,square_plot=False,cmap="coolwarm_r",cbar=True,
                 compass=True,label=None,linpix=None,linscale=None,
                 scaleunit=None,locuncert=0,quiet=False):
        # grab image data from science extension of the image
        self.image = image
        try:
            self.imagearray,self.hdr = fits.getdata(image,sciext,header=True)
        except (IOError,IndexError):
            raise Exception("cannot get data/header from {0} (extension {1})"
                            .format(image,sciext))
        self.basename = os.path.splitext(image)[0]

        # round location and map to integers (fractional pixels make no 
        # difference to flight calculations) - based on 0-indexed input!
        self.xloc,self.yloc = [int(round(loc,0)) for loc in location]
        # but keep the float diffs incase we're plotting a locuncert circle
        self.xloc_fl = location[0] - self.xloc
        self.yloc_fl = location[1] - self.yloc

        shape = self.imagearray.shape
        if self.xloc > shape[1] or self.yloc > shape[0]:
            raise Exception("x,y location not within input image")

        self.segmap = segmap
        self.segmask = segmask

        self.segsect = segsect
        if self.segsect == "i" and not HASPYDS9:
            raise Exception("couldn't import pyds9: cannot use interactive \
                            segmentation selection")
        self.centre = centre
        self.brightest = brightest
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
        # make a segmentation map if we don't have one (and we don't have a 
        # seg mask provided to us)
        if self.segmap is None and self.segmask is None:
            self.makesegmap()

        # check it's the same dimensions as our image
        if self.segmask is None:
            self.segarray = fits.getdata(self.segmap)
            if self.imagearray.shape != self.segarray.shape:
                raise Exception("image and seg map must have same size")
            # if there isn't a segmap object at the pixel location, we can't
            # find (u)nderlying or (t)ouching sections so must exit here
            if self.segarray[self.yloc,self.xloc] == 0\
               and self.segsect in ("t","u"):
                print "no segmentation map object at pixel",
                print "location {},{}".format(self.xloc,self.yloc)
                return
            # make a mask from this seg map depending on the choice of sections
            self.makesegmask()
        # otherwise just load the data from the user-provided seg mask
        else:
            self.segmask = fits.getdata(self.segmask)

        # save a copy of the mask if needed
        if self.savemask:
            fits.writeto(self.basename+MASKSUFF,
                           data=self.segmask.astype('int'),clobber=True)
        if np.sum(self.segmask) == 0: # i.e. no objects in segmap
            print "no segmentation map objects matching",
            print "value(s) {}".format(", ".join(map(str,self.segsect)))
            return

        maskedimagearray = self.imagearray.copy()
        maskedimagearray[self.segmask == 0] = 0 

        # trim the image array to just enclose all the non-zero pixels from the
        # segmentation mask. Add a 1pixel border for good measure and plotting-
        # niceness
        z = np.where(self.segmask>0)
        # append the pixel chosen to include that in map
        z = (np.append(z[0],self.yloc),np.append(z[1],self.xloc))

        # calculate min/max including locuncert if included. Gold star if you
        # can figure out this car-crash:
        if self.locuncert > 0:
            #lu = int(math.ceil(self.locuncert))
            lu = self.locuncert
            minx,maxx,miny,maxy = map(int,
                                  (min(min(z[0]),self.yloc+self.yloc_fl-lu)-1,
                                   max(max(z[0]),self.yloc+self.yloc_fl+lu)+2,
                                   min(min(z[1]),self.xloc+self.xloc_fl-lu)-1,
                                   max(max(z[1]),self.xloc+self.xloc_fl+lu)+2)
                                     )
            print "\n",minx,maxx,miny,maxy
        else:
            minx,maxx,miny,maxy = (min(z[0])-1,max(z[0])+2,
                                   min(z[1])-1,max(z[1])+2)
        # if we want a square plot, then expand the limits of the shortest side
        # to make it square by padding the 1/2 the difference in each direction
        if self.square_plot:
            xsize = maxx-minx
            ysize = maxy-miny
            if xsize < ysize:
                minx -= int(math.floor(float(ysize-xsize)/2))
                maxx += int(math.ceil(float(ysize-xsize)/2))
            if xsize > ysize:
                miny -= int(math.floor(float(xsize-ysize)/2))
                maxy += int(math.ceil(float(xsize-ysize)/2))
        self.flightarray = maskedimagearray[minx:maxx,miny:maxy]
        self.trimimagearray = self.imagearray[minx:maxx,miny:maxy]
        # update coordinates of location onto flightarray coordinates
        self.xfl,self.yfl = self.xloc-miny,self.yloc-minx
        # update coordinates of centre onto flightarray coordinates
        if self.centre is not None:
            self.centre = self.centre[0]-miny,self.centre[1]-minx
        if self.brightest is not None:
            self.brightest = self.brightest[0]-miny,self.brightest[1]-minx
        try:
            self.flightarray[self.yfl,self.xfl]
        except IndexError:
            raise IndexError("location coordinates not within bounds of chosen"\
                             " segmentation sections")
        try:
            self.flightarray[self.centre[1],self.centre[0]]
        except IndexError:
            raise IndexError("centre coordinates {} are not within bounds of "\
                             "chosen segmentation sections".format(self.centre))
        try:
            self.flightarray[self.brightest[1],self.brightest[0]]
        except IndexError:
            raise IndexError("brightest coordinates {} are not within bounds of "\
                             "chosen segmentation sections".format(self.brightest))

        self.getflight()

        if self.plots:
            self.makeplots()
        if not self.quiet:
            print
            print "Image:           ",self.image
            print "Location:        ",self.xloc,self.yloc
            print "Trimmed section:  [{}:{},{}:{}]".format(minx,maxx,miny,maxy)
            print "Trimmed location:",self.xfl,self.yfl
            print "Pixel value:     ",self.locpixvalue
            print "F_light value:    {:.3f}".format(self.flight)
            print

    def makesegmap(self):
        if not self.quiet:
            print "making segmentation map"
        segmapname = self.basename+SEGMAPSUFF
        runsex(self.image,chkimg_type="SEGMENTATION",
                chkimg_name=segmapname)
        try:
            open(segmapname)
        except IOError:
            raise Exception("segmentation map ({}) not created for some reason"\
                            .format(segmapname))
        else:
            self.segmap = segmapname

    def makesegmask(self):
        if not self.quiet:
            print "making segmentation mask"
        y,x = self.xloc,self.yloc # save transposing arrays

        if isinstance(self.segsect,str):
            if self.segsect[0] == "t":
                toucharray = ndimage.label(self.segarray,
                            structure=ndimage.generate_binary_structure(2,2))[0]
                self.segmask = (toucharray == toucharray[x,y])
                # if we have +i-j..etc designations, add/remove them from the
                # segmask as appropriate
                toadd = re.findall("(?<=\+)(\d+)",self.segsect)
                toremove = re.findall("(?<=\-)(\d+)",self.segsect)
                for a in toadd:
                    self.segmask[np.where(self.segarray==int(a))] = True
                for r in toremove:
                    self.segmask[np.where(self.segarray==int(r))] = False
                return

            segvals = []
            if self.segsect[0] == "u":
                segvals.append(self.segarray[x,y])
                # if we have +i+j..etc designations, add them from the
                # segmask as appropriate, only adding makes sense when using u
                toadd = re.findall("(?<=\+)(\d+)",self.segsect)
                segvals.extend(map(int,toadd))
            elif self.segsect == "i":
                print "TODO"
                raise AttributeError("interactive not implemented yet")
        elif isinstance(self.segsect,(int,tuple,list)):
            segvals = self.segsect

        # return a mask of segmap that is True for pixels within objects
        # we're interested in
        self.segmask =  np.in1d(self.segarray,segvals)\
                          .reshape(self.segarray.shape)
                          # in1d doesn't preserve shape...

    def getflight(self):
        if not self.quiet:
            print "calculating f_light"
        # find value of location pixel (numpy+fits = counter-intuitive x,y)
        self.locpixvalue = self.flightarray[self.yfl,self.xfl]

        # make a sorted 1d list of pixel values
        pixelvalues = self.flightarray.ravel()
        pixelvalues.sort()
        self.sortedpixelvalues = pixelvalues

        # find the index of the sorted array corresponding to the location
        self.locpixnumber = np.where(pixelvalues==self.locpixvalue)[0]
        nsame = len(self.locpixnumber)
        if nsame > 1 and self.locpixvalue > 0:
            if not self.quiet:
                print "\t{} pixels with same value".format(nsame)
                print "\tsetting f_light to average for this pixel value"
            self.locpixnumber = self.locpixnumber[nsame/2]
        else:
            self.locpixnumber = self.locpixnumber[0]

        # create cumulative sum distribution normalised to total sum of pixels
        totalvalue = float(np.sum(pixelvalues))
        self.cumarray = np.cumsum(pixelvalues)/totalvalue
        # find value of this cumulative sum at locpixnumber
        self.flight = self.cumarray[self.locpixnumber]

    def makeplots(self):
        if not self.quiet:
            print "making plots" 
        plt.clf()
        # make a heatmap plot of Flight values
        # clip top/bottom pixel values for plotting
        sortedarray = np.sort(self.trimimagearray.ravel())
        npix = len(sortedarray)
        cutpix = int(math.ceil(CLIP*npix))
        low_cut = sortedarray[cutpix]
        high_cut = sortedarray[npix-cutpix]
        imageplotarray = np.clip(self.trimimagearray,low_cut,high_cut)
        plt.imshow(imageplotarray,origin="lower",interpolation="nearest",
                   cmap="Greys")
        sortindices = np.searchsorted(self.sortedpixelvalues,self.flightarray)
        heatplotarray = self.cumarray[sortindices]
        # mask zero values to prevent plotting them
        heatplotarray = np.ma.masked_where(self.flightarray==0, heatplotarray)
        plt.imshow(heatplotarray,origin="lower",interpolation="nearest",
                   cmap=self.cmap)
        self.heatplotarray = heatplotarray
        ax1 = plt.axes()
        ax1.autoscale(False)
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        if self.centre is not None:
            self.cenmark = plt.plot(self.centre[0],self.centre[1],marker="+",
                                    markersize=24,markerfacecolor="none",
                                    markeredgewidth=2,c="k")
        if self.brightest is not None:
            self.brightmark = plt.plot(self.brightest[0],self.brightest[1],
                                       marker="x",markersize=20,
                                       markerfacecolor="none",markeredgewidth=2,
                                       c="k")
        if self.label is not None:
            ax1.text(0.05,0.95,"\\textbf{"+self.label+"}",ha="left",va="center",
                 transform=ax1.transAxes,size=30,weight="bold",color="k",
                 path_effects=[PathEffects.withStroke(linewidth=3,
                                                      foreground="w")])
        if self.compass is not False:
            if isinstance(self.compass, bool):
                try:
                    orientat = self.hdr["ORIENTAT"]
                    print "1",orientat
                except KeyError:
                    print "ORIENTAT not in FITS header, skipping N-E plotting"
                    orientat = False
            elif isinstance(self.compass,(int,float)):
                orientat = self.compass
            else:
                print "bad type of `compass`, %s. Must be bool, int or float"\
                    % type(self.compass)
                print "skipping N-E plotting"
                orientat = False
            if orientat is not False:
                compass = AnchoredCompass(ax1,ori=orientat)
                ax1.add_artist(compass)
        if not None in (self.linpix,self.linscale):
            npix = self.linscale/float(self.linpix) # no. pixels scale line needs to be
            plotx = (npix/self.heatplotarray.shape[1]) # length of line/axis
            ax1.plot((0.05,0.05+plotx),(0.05,0.05),c="k",lw=2,
                     transform=ax1.transAxes,
                     path_effects=[PathEffects.withStroke(linewidth=4,
                                                          foreground="w")])
            ax1.text(0.05,0.1,"%s %s" % (self.linscale,self.scaleunit),
                     ha="left",va="center",transform=ax1.transAxes,
                     size=30,color="k",
                     path_effects=[PathEffects.withStroke(linewidth=3,
                                                      foreground="w")])
        if self.locuncert > 0:
            width = 2*self.locuncert/self.heatplotarray.shape[1]
            height = 2*self.locuncert/self.heatplotarray.shape[0]
            xaxes = float(self.xfl+self.xloc_fl)/self.heatplotarray.shape[1]
            yaxes = float(self.yfl+self.yloc_fl)/self.heatplotarray.shape[0]
            ax1.add_artist(Ellipse((xaxes,yaxes),width,height,
                                   transform=ax1.transAxes,
                                   facecolor="none",edgecolor="k",lw=2,
                                   ls="dashed"))
        else:
            self.starmark = plt.plot(self.xfl,self.yfl,marker="*",markersize=24,
                                 c="k",markerfacecolor="none",markeredgewidth=2)
        if self.cbar:
            color_bar = plt.colorbar(orientation='horizontal')
            cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
            plt.setp(cbytick_obj, color='k')
        plt.savefig(self.basename+"_flheat.eps",bbox_inches="tight")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
                      description='Perform F_light analysis on a FITS file. '
                      ' If file paths for a seg map or mask are not provided'
                      ' they will be created.')

    parser.add_argument('image',type=str,help='FITS file for analysis')
    parser.add_argument('location',type=str,help='image pixel coordinates '
                        'of object in format `x,y`, zero-indexed!')
    parser.add_argument('-s','--segmap',type=str,default=None,help='name of '
                        'segmentation map from sextractor to use. (default: '
                        'None)')
    parser.add_argument('-m','--segsect',type=str,default='u',help='method of'
                        'selecting segmap objects to use. `t`: use all non-'
                        'zero segmap pixels touching the location chosen, '
                        'regardless of segmap number. `u`: only use the segmap'
                        ' object underlying directly the location chosen. '
                        'i,j,k...: a single or comma-separated list of integer'
                        ' values of the segmap objects to use. to combine '
                        'methods use, for e.g. t+i-j.., where the method `t` '
                        'be used in addition to selecting pixels from segmap '
                        'object i and removing pixels from segmap object j. '
                        ' (default: `u`)')
    parser.add_argument('-k','--segmask',type=str,default=None,help='name of '
                        'segmentation mask to use. (default: None)')
    parser.add_argument('-e','--sciext',default='SCI',help='extension '
                        'name or number of data to perform analysis on '
                        'in `image`. (default: "SCI")')
    parser.add_argument('-c','--centre',type=str,default=None,help='image pixel'
                        ' coordinates of galaxy centre in format `x,y`, '
                        'zero-indexed!')
    parser.add_argument('-b','--brightest',type=str,default=None,help='image '
                        'pixel coordinates of brightest pixel in format `x,y`, '
                        'zero-indexed!')
    parser.add_argument('--savemask',action='store_true',help='save a copy'
                        ' off the mask, with suffix {}'.format(MASKSUFF))
    parser.add_argument('-p','--noplots',dest='plots',action='store_false',
                        help='turn off plotting of results')
    parser.add_argument('--square_plot',action='store_true',
                        help='force heat map to be square')  
    parser.add_argument('--cmap',type=str,default='Spectral',help='colormap to '
                        'use for heat map plot. (default:`coolwarm_r`')
    parser.add_argument('--nocbar',dest='cbar',action='store_false',
                        help='do not display a colour bar legend on heat map')
    parser.add_argument('--nocompass',dest='compass',action='store_false',
                        help='do not display a NE compass on heat map,'
                        'need a valid ORIENTAT (yaxis as deg E of N) header')
    parser.add_argument('--compassdeg',dest='compassdeg',type=float,
                        default=None,
                        help='the degrees east of north that the y-axis is, '
                        'use if there is not ORIENTAT header')
    parser.add_argument('--label',type=str,default=None,
                        help='string to print as label on heat map (default:'
                        ' None)')
    parser.add_argument('--linpix',type=float,default=None,
                        help='linear size of a pixel at the distance of the '
                        'host (default: None)')
    parser.add_argument('--linscale',type=float,default=None,
                        help='size of linear scale bar to plot on the heat map '
                        '- linpix must be defined and this scale will be in the'
                        ' the same units (default: None)')
    parser.add_argument('--scaleunit',type=str,default="",
                        help='unit to include after scale (e.g. "kpc", \"), ' 
                        'linpix and  linscale should both be in this unit '
                        '(default: "")')
    parser.add_argument('--locuncert',type=float,default=0,
                        help='the uncertainty in the specified location, given '
                        ' in units of pixels. Instead of flagging a pixel, it '
                        'will draw a circle with radius locuncert on the heat'
                        'map (default: 0)')
    parser.add_argument('-q','--quiet',action='store_true',
                        help='run silently (unless something goes wrong)') 
    args = parser.parse_args()

    splitloc = args.location.split(",")
    if len(splitloc) != 2:
        print "location should be zero-indexed coordinates of the format `x,y`"
        sys.exit(2)
    else:
        try:
            args.location = map(float,splitloc) # convert loc to floats
        except ValueError:
            print "location needs to be x,y pixel coordinates, numeric only"
            sys.exit(2)

    if args.segsect != "u" and args.segsect[0] != "t":
        splitsect = args.segsect.split(",")
        try:
            args.segsect = map(int,splitsect)
        except ValueError:
            print "segsect must be comma-separated list of integer values if",
            print "defining sections to use"
            sys.exit(2)

    if args.segmap is not None:
        try:
            fits.open(args.segmap)
        except IOError:
            print "Couldn't open {} with fits, make sure it exists and is",
            print "a valid FITS file".format(args.segmap)
            sys.exit(2)
    if args.segmask is not None:
        try:
            fits.open(args.segmask)
        except IOError:
            print "Couldn't open {} with fits, make sure it exists and is",
            print "a valid FITS file".format(args.segmask)
            sys.exit(2)
    if args.centre is not None:
        splitcen = args.centre.split(",")
        if len(splitcen) != 2:
            print "centre should be zero-indexed coords of the format `x,y`"
            sys.exit(2)
        else:
            try:
                args.centre = map(float, centre) # convert loc to floats
            except ValueError:
                print "location needs to be x,y pixel coordinates, numeric only"
                sys.exit(2)

    if args.compassdeg:
        args.compass = args.compassdeg
    del args.compassdeg

    arg_dict = vars(args) # make dictionary of arguments
    f = calcflight(**arg_dict) # pass these to construct flight class
    f.main() # run the class