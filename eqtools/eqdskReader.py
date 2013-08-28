# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This file is part of EqTools.
#
# EqTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EqTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

"""
This module contains the EQDSKReader class, which creates Equilibrium class
functionality for equilibria stored in eqdsk files from EFIT(a- and g-files).

Classes:
    EQDSKReader: class inheriting Equilibrium reading g- and a-files for
        equilibrium data.
"""

from core import Equilibrium
import scipy
import glob
import re
import matplotlib.pyplot as plt

class EQDSKReader(Equilibrium):
    """
    Equilibrium subclass working from eqdsk ASCII-file equilibria.

    Inherits mapping and structural data from Equilibrium, populates equilibrium
    and profile data from g- and a-files for a selected shot and time window.
    """
    def __init__(self,shot,time,gfilename=None,afilename=None,length_unit='m'):
        """
        Initializes EQDSKReader object.  Pulls data from g- and a-files for given
        shot, time slice.  By default, attempts to parse shot, time inputs into file
        name, and searches directory for appropriate files.  Optionally, the user may
        instead directly input a file path for a-file, g-file.

        INPUTS:
        shot:       shot index
        time:       time slice in ms
        gfilename:  (optional, default None) if set, ignores shot,time inputs and pulls g-file by name
        afilename:  (optional, default None) if set, ignores shot,time inputs and pulls a-file by name
        """
        """
        Create instance of EQDSKReader.

        Generates object and reads data from selected g-file (either manually set or
        autodetected based on user shot and time selection), storing as object
        attributes for usage in Equilibrium mapping methods.

        Args:
            shot: Int.  Shot index.
            time: Int.  Time index (typically ms).  Shot and Time used to autogenerate filenames.

        Kwargs:
            gfilename: String.  Manually selects ASCII file for equilibrium read.
            afilename: String.  Manually selects ASCII file for time-history read.
            length_unit: String.  Flag setting length unit for equilibrium scales.
                Defaults to 'm' for lengths in meters.
        """
        # instantiate superclass, forcing time splining to false (eqdsk only contains single time slice)
        super(EQDSKReader,self).__init__(length_unit=length_unit,tspline=False)

        # parse shot and time inputs into standard naming convention
        if len(str(time)) < 5:
            timestring = '0'*(5-len(str(time))) + str(time)
        elif len(str(time)) > 5:
            timestring = str(time)[-5:]
            print('Time window string greater than 5 digits.  Masking to last 5 digits.  \
                  If this does not match the selected EQ files, \
                  please use explicit filename inputs.')
        else:   #exactly five digits
            timestring = str(time)

        name = str(shot)+'.'+timestring

        # if explicit filename for g-file is not set, check current directory for files matching name
        # if multiple valid files or no files are found, trigger ValueError
        if gfilename is None:   #attempt to generate filename
            print('Searching directory for file g'+name+'.')
            gcurrfiles = glob.glob('g'+name+'*')
            if len(gcurrfiles) == 1:
                self._gfilename = gcurrfiles[0]
                print('File found: '+self._gfilename)
            elif len(gcurrfiles) > 1:
                raise ValueError('Multiple valid g-files detected in directory.  \
                                  Please select a file with explicit \
                                  input or clean directory.')
            else:   # no files found
                raise ValueError('No valid g-files detected in directory.  \n\
                                  Please select a file with explicit input or \n\
                                  ensure file is in directory.')
        else:   # check that given file is in directory
            gcurrfiles = glob.glob(gfilename)
            if len(gcurrfiles) < 1:
                raise ValueError('No g-file with the given name detected in directory.  \
                                  Please ensure the file is in the active directory or \
                                  that you have supplied the correct name.')
            else:
                self._gfilename = gfilename

        # and likewise for a-file name.  However, we can operate at reduced capacity
        # without the a-file.  If no file with explicitly-input name is found, or 
        # multiple valid files (with no explicit input) are found, raise ValueError.
        # otherwise (no autogenerated files found) set hasafile flag false and 
        # nonfatally warn user.
        if afilename is None:
            print('Searching directory for file a'+name+'.')
            acurrfiles = glob.glob('a'+name+'*')
            if len(acurrfiles) == 1:
                self._afilename = acurrfiles[0]
                print('File found: '+self._afilename)
                self._hasafile = True
            elif len(acurrfiles) > 1:
                raise ValueError('Multiple valid a-files detected in directory.  \
                                  Please select a file with explicit \
                                  input or clean directory.')
            else:   # no files found
                print('No valid a-files detected in directory.  \
                      Please select a file with explicit input or \
                      ensure file in in directory.  Disabling a-file \
                      read functions.')
                self._afilename = None
                self._hasafile = False
        else:   # check that given file is in directory
            acurrfiles = glob.glob(afilename)
            if len(acurrfiles) < 1:
                raise ValueError('No a-file with the given name detected in directory.  \
                                  Please ensure the file is in the active directory or \
                                  that you have supplied the correct name.')
            else:
                self._afilename = afilename

        # now we start reading the g-file
        with open(self._gfilename,'r') as gfile:
            # read the header line, containing grid size, mfit size, and type data
            line = gfile.readline().split()
            self._date = line[1]                         # (str) date of g-file generation, MM/DD/YYYY
            self._shot = int(re.split('\D',line[2])[-1]) # (int) shot index
            timestring = line[3]                         # (str) time index, with units (e.g. '875ms')
            imfit = int(line[4])                         # not sure what this is supposed to be...
            nw = int(line[5])                            # width of flux grid (dim(R))
            nh = int(line[6])                            # height of flux grid (dim(Z))

            #extract time, units from timestring
            time = re.findall('\d+',timestring)[0]
            self._tunits = timestring.split(time)[1]
            timeConvertDict = {'ms':1000.,'s':1}
            self._time = scipy.array(float(time)*timeConvertDict[self._tunits])
            
            # next line - construction values for RZ grid
            line = gfile.readline()
            line = re.findall('-?\d\.\d*E[-+]\d*',line)     # regex magic!
            xdim = float(line[0])     # width of R-axis in grid
            zdim = float(line[1])     # height of Z-axis in grid
            rzero = float(line[2])    # zero point of R grid
            rgrid0 = float(line[3])   # start point of R grid
            zmid = float(line[4])     # midpoint of Z grid

            # construct EFIT grid
            self._rGrid = scipy.linspace(rgrid0,rgrid0 + xdim,nw)
            self._zGrid = scipy.linspace(zmid - zdim/2.0,zmid + zdim/2.0,nh)
            drefit = (self._rGrid[-1] - self._rGrid[0])/(nw-1)
            dzefit = (self._zGrid[-1] - self._zGrid[0])/(nh-1)

            # read R,Z of magnetic axis, psi at magnetic axis and LCFS, and bzero
            line = gfile.readline()
            line = re.findall('-?\d\.\d*E[-+]\d*',line)
            self._rmaxis = scipy.array(float(line[0]))
            self._zmaxis = scipy.array(float(line[1]))
            self._psiAxis = scipy.array(float(line[2]))
            self._psiLCFS = scipy.array(float(line[3]))
            self._bcentr = scipy.array(float(line[4]))

            # read EFIT-calculated plasma current, psi at magnetic axis (duplicate), 
            # dummy, R of magnetic axis (duplicate), dummy
            line = gfile.readline()
            line = re.findall('-?\d\.\d*E[-+]\d*',line)
            self._cpasma = scipy.array(float(line[0]))

            # read Z of magnetic axis (duplicate), dummy, psi at LCFS (duplicate), dummy, dummy
            line = gfile.readline()
            # don't actually need anything from this line

            # start reading fpol, next nw inputs
            nrows = nw/5
            if nw % 5 != 0:     # catch truncated rows
                nrows += 1

            self._fpol = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    self._fpol.append(float(val))
            self._fpol = scipy.array(self._fpol)

            # and likewise for pressure
            self._pres = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    self._pres.append(float(val))
            self._pres = scipy.array(self._pres)

            # geqdsk written as negative for positive plasma current
            # ffprim, pprime input with correct EFIT sign
            self._ffprim = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    self._ffprim.append(float(val))
            self._ffprim = scipy.array(self._ffprim)

            self._pprime = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    self._pprime.append(float(val))
            self._pprime = scipy.array(self._pprime)

            # read the 2d [nw,nh] array for psiRZ
            # start by reading nw x nh points into 1D array,
            # then repack in column order into final array
            npts = nw*nh
            nrows = npts/5
            if npts % 5 != 0:
                nrows += 1

            psis = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    psis.append(float(val))
            self._psiRZ = scipy.array(psis).reshape((nw,nh),order='C')

            # read q(psi) profile, nw points (same basis as fpol, pres, etc.)
            nrows = nw/5
            if nw % 5 != 0:
                nrows += 1

            self._qpsi = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    self._qpsi.append(float(val))
            self._qpsi = scipy.array(self._qpsi)

            # read nbbbs, limitr
            line = gfile.readline().split()
            nbbbs = int(line[0])
            limitr = int(line[1])

            # next data reads as 2 x nbbbs array, then broken into
            # rbbbs, zbbbs (R,Z locations of LCFS)
            npts = 2*nbbbs
            nrows = npts/5
            if npts % 5 != 0:
                nrows += 1
            bbbs = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    bbbs.append(float(val))
            bbbs = scipy.array(bbbs).reshape((2,nbbbs),order='C')
            self._RLCFS = bbbs[0,:]
            self._ZLCFS = bbbs[1,:]

            # next data reads as 2 x limitr array, then broken into
            # xlim, ylim (locations of limiter)(?)
            npts = 2*limitr
            nrows = npts/5
            if npts % 5 != 0:
                nrows += 1
            lim = []
            for i in range(nrows):
                line = gfile.readline()
                line = re.findall('-?\d\.\d*E[-+]\d*',line)
                for val in line:
                    lim.append(float(val))
            lim = scipy.array(lim).reshape((2,limitr),order='C')
            self._xlim = lim[0,:]
            self._ylim = lim[1,:]

            # this is the extent of the original g-file read.
            # attempt to continue read for newer g-files; exception
            # handler sets relevant parameters to None for older g-files
            try:
                # read kvtor, rvtor, nmass
                line = gfile.readline().split()
                kvtor = int(line[0])
                rvtor = float(line[1])
                nmass = int(line[2])

                # read kvtor data if present
                if kvtor > 0:
                    nrows = nw/5
                    if nw % 5 != 0:
                        nrows += 1
                    self._presw = []
                    for i in range(nrows):
                        line = gfile.readline()
                        line = re.findall('-?\d.\d*E[-+]\d*',line)
                        for val in line:
                            self._presw.append(float(val))
                    self._presw = scipy.array(self._presw)
                    self._preswp = []
                    for i in range(nrows):
                        line = gfile.readline()
                        line = re.findall('-?\d.\d*E[-+]\d*',line)
                        for val in line:
                            self._preswp.append(float(val))
                    self._preswp = scipy.array(self._preswp)
                else:
                    self._presw = scipy.array([0])
                    self._preswp = scipy.array([0])

                # read ion mass density if present
                if nmass > 0:
                    nrows = nw/5
                    if nw % 5 != 0:
                        nrows += 1
                    self._dmion = []
                    for i in range(nrows):
                        line = gfile.readline()
                        line = re.findall('-?\d.\d*E[-+]\d*',line)
                        for val in line:
                            self._dmion.append(float(val))
                    self._dmion = scipy.array(self._dmion)
                else:
                    self._dmion = scipy.array([0])

                # read rhovn
                nrows = nw/5
                if nw % 5 != 0:
                    nrows += 1
                self._rhovn = []
                for i in range(nrows):
                    line = gfile.readline()
                    line = re.findall('-?\d.\d*E[-+]\d*',line)
                    for val in line:
                        self._rhovn.append(float(val))
                self._rhovn = scipy.array(self._rhovn)

                # read keecur; if >0 read workk
                line = gfile.readline.split()
                keecur = int(line[0])
                if keecur > 0:
                    self._workk = []
                    for i in range(nrows):
                        line = gfile.readline()
                        line = re.findall('-?\d.\d*E[-+]\d*',line)
                        for val in line:
                            self._workk.append(float(val))
                    self._workk = scipy.array(self._workk)
                else:
                    self._workk = scipy.array([0])
            except:
                self._presw = scipy.array([0])
                self._preswp = scipy.array([0])
                self._rhovn = scipy.array([0])
                self._dmion = scipy.array([0])
                self._workk = scipy.array([0])
                    
    def __str__(self):
        return 'G-file equilibrium from '+str(self._gfile)
        
    def getInfo(self):
        """
        returns namedtuple of equilibrium information
        outputs:
        namedtuple containing
            shot:   shot index
            time:   time point of g-file
            nr:     size of R-axis of spatial grid
            nz:     size of Z-axis of spatial grid
        """
        data = namedtuple('Info',['shot','time','nr','nz'])
        try:
            nr = len(self._rGrid)
            nz = len(self._zGrid)
            shot = self._shot
            time = self._time
        except TypeError:
            nr,nz,shot,time=0
            print 'failed to load data from g-file.'
        return data(shot=shot,time=time,nr=nr,nz=nz)

    def getTimeBase(self):
        #returns EFIT time base array (t)
        raise NotImplementedError()

    def getFluxGrid(self):
        """
        returns EFIT flux grid, [r,z]
        """
        return self._psiRZ.copy()

    def getRGrid(self):
        """
        returns EFIT R-axis [r]
        """
        return self._rGrid.copy()

    def getZGrid(self):
        """
        returns EFIT Z-axis [z]
        """
        return self._zGrid.copy()

    def getFluxAxis(self):
        """
        returns psi on magnetic axis
        """
        return scipy.array(self._psiAxis)

    def getFluxLCFS(self):
        """
        returns psi at separatrix
        """
        return scipy.array(self._psiLCFS)

    def getRLCFS(self):
        #returns R-positions mapping LCFS, rbbbs(t,n)
        return self._RLCFS.copy()

    def getZLCFS(self):
        #returns Z-positions mapping LCFS, zbbbs(t,n)
        return self._ZLCFS.copy()

    def getFluxVol(self):
        #returns volume contained within a flux surface as function of psi, volp(psi,t)
        raise NotImplementedError()

    def getVolLCFS(self):
        #returns plasma volume in LCFS, vout(t)
        raise NotImplementedError()

    def getRmidPsi(self):
        #returns max major radius of flux surface, rpres(t,psi)
        raise NotImplementedError()

    def getFluxPres(self):
        #returns EFIT-calculated pressure p(psi,t)
        return self._pres.copy()

    def getElongation(self):
        #returns LCFS elongation, kappa(t)
        raise NotImplementedError()

    def getUpperTriangularity(self):
        #returns LCFS upper triangularity, delta_u(t)
        raise NotImplementedError()

    def getLowerTriangularity(self):
        #returns LCFS lower triangularity, delta_l(t)
        raise NotImplementedError()

    def getShaping(self):
        #returns dimensionless shaping parameters for plasma
        #namedtuple containing {LCFS elongation, LCFS upper/lower triangularity)
        raise NotImplementedError()

    def getMagR(self):
        #returns magnetic-axis major radius, rmagx(t)
        raise NotImplementedError()

    def getMagZ(self):
        #returns magnetic-axis Z, zmagx(t)
        raise NotImplementedError()

    def getAreaLCFS(self):
        #returns LCFS surface area, areao(t)
        raise NotImplementedError()

    def getAOut(self):
        #returns outboard-midplane minor radius
        raise NotImplementedError()

    def getRmidOut(self):
        #returns outboard-midplane major radius
        raise NotImplementedError()

    def getGeometry(self):
        #returns dimensional geometry parameters for plasma
        #namedtuple containing {mag axis r,z, LCFS area, volume, outboard midplane major radius}
        raise NotImplementedError()

    def getQProfile(self):
        #returns safety factor profile q(psi,t):
        return self._qpsi.copy()

    def getQ0(self):
        #returns q-value on magnetic axis, q0(t)
        raise NotImplementedError()

    def getQ95(self):
        #returns q at 95% flux, psib(t)
        raise NotImplementedError()

    def getQLCFS(self):
        #returns q on LCFS, qout(t)
        raise NotImplementedError()

    def getQ1Surf(self):
        #returns outboard-midplane minor radius of q=1 surface, aaq1(t)
        raise NotImplementedError()
    
    def getQ2Surf(self):
        #returns outboard-midplane minor radius of q=2 surface, aaq2(t)
        raise NotImplementedError()

    def getQ3Surf(self):
        #returns outboard-midplane minor radius of q=3 surface, aaq3(t)
        raise NotImplementedError()

    def getQs(self):
        #returns specific q-profile values
        #namedtuple containing {q0, q95, q(LCFS), minor radius of q=1,2,3 surfaces}
        raise NotImplementedError()

    def getBtVac(self):
        #returns vacuum on-axis toroidal field btaxv(t)
        raise NotImplementedError()

    def getBtPla(self):
        #returns plasma on-axis toroidal field btaxp(t)
        raise NotImplementedError()

    def getBpAvg(self):
        #returns avg poloidal field, bpolav(t)
        raise NotImplementedError() 

    def getFields(self):
        #returns magnetic-field measurements from EFIT
        #dict containing {Btor on magnetic axis (plasma and vacuum), avg Bpol)
        raise NotImplementedError()

    def getIpCalc(self):
        #returns EFIT-calculated plasma current
        return self._cpasma.copy()

    def getIpMeas(self):
        #returns measured plasma current
        raise NotImplementedError()

    def getJp(self):
        #returns (r,z,t) grid of EFIT-calculated current density
        raise NotImplementedError()

    def getBetaT(self):
        #returns calculated toroidal beta, betat(t)
        raise NotImplementedError()

    def getBetaP(self):
        #returns calculated avg poloidal beta, betap(t)
        raise NotImplementedError()

    def getLi(self):
        #returns calculated internal inductance of plasma, ali(t)
        raise NotImplementedError()

    def getBetas(self):
        #returns calculated beta and inductive values
        #namedtuple of {betat,betap,li}
        raise NotImplementedError()

    def getDiamagFlux(self):
        #returns diamagnetic flux, diamag(t)
        raise NotImplementedError()

    def getDiamagBetaT(self):
        #returns diamagnetic-loop toroidal beta, betatd(t)
        raise NotImplementedError()

    def getDiamagBetaP(self):
        #returns diamagnetic-loop poloidal beta, betapd(t)
        raise NotImplementedError()

    def getDiamagTauE(self):
        #returns diamagnetic-loop energy confinement time, taudia(t)
        raise NotImplementedError()

    def getDiamagWp(self):
        #returns diamagnetic-loop plasma stored energy, wplasmd(t)
        raise NotImplementedError()

    def getDiamag(self):
        #returns diamagnetic measurements of plasma parameters
        #namedtuple of {diamag flux, betat,betap from diamag coils, tau_E from diamag, diamag stored energy)
        raise NotImplementedError()

    def getWMHD(self):
        #returns EFIT-calculated MHD stored energy wplasm(t)
        raise NotImplementedError()

    def getTauMHD(self):
        #returns EFIT-calculated MHD energy confinement time taumhd(s)
        raise NotImplementedError()

    def getPinj(self):
        #returns EFIT-calculated injected power, pbinj(t)
        raise NotImplementedError()

    def getWbdot(self):
        #returns EFIT-calculated d/dt of magnetic stored energy, wbdot(t)
        raise NotImplementedError()

    def getWpdot(self):
        #returns EFIT-calculated d/dt of plasma stored energy, wpdot(t)
        raise NotImplementedError()

    def getEnergy(self):
        #returns stored-energy parameters
        #dict of {stored energy, MHD tau_E, injected power, d/dt of magnetic, plasma stored energy)
        raise NotImplementedError()

    def getParam(self,path):
        #backup function - takes parameter name for EFIT variable, returns that variable
        #acts as wrapper for EFIT tree access from within object
        raise NotImplementedError()
        
    def getMachineCrossSection(self):
        raise NotImplementedError('no machine cross section stored in g-files.')
        
    def plotFLux(self):
        """
        streamlined plotting of flux contours directly from psi grid
        """
        plt.ion()

        try:
            psiRZ = self.getFluxGrid()
            rGrid = self.getRGrid()
            zGrid = self.getZGrid()

            RLCFS = self.getRLCFS()
            ZLCFS = self.getZLCFS()
        except ValueError:
            raise AttributeError('cannot plot EFIT flux map.')

        fluxPlot = plt.figure(figsize=(6,11))
        fluxPlot.set_xlabel('$R$ (m)')
        fluxPlot.set_ylabel('$Z$ (m)')
        fillcont = plt.contourf(rGrid,zGrid,psiRZ,50)
        cont = plt.contour(rGrid,zGrid,psiRZ,50,colors='k',linestyles='solid')
        LCFS = plt.plot(RLCFS,ZLCFS,'r',linewidth=3)
        plt.show()
                



                






