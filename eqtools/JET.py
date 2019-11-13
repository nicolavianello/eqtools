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

"""This module provides classes inheriting :py:class:`eqtools.Equilibrium` for 
working with ASDEX Upgrade experimental data.
"""

import warnings

import scipy

from .core import ModuleWarning, Equilibrium

try:
    from jet.data import sal
    from jet.data.sal import SALException
    _has_sal = True
except ImportError:
    warnings.warn(
        "sal module could not be loaded -- classes that use "
        "sal for data access will not work.",
        ModuleWarning,
    )
    _has_sal = False
try:
    import matplotlib.pyplot as plt
    _has_plt = True
except:
    warnings.warn(
        "Matplotlib.pyplot module could not be loaded -- classes that "
        "use pyplot will not work.",
        ModuleWarning,
    )
    _has_plt = False


class JETSALData(Equilibrium):
    """Inherits :py:class:`eqtools.Equilibrium` class. Machine-specific data
    handling class for JET Upgrade. Pulls JET data through the jet.sal classes
    and stores as object attributes. Each data variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for mapping are pulled on initialization (e.g. psirz grid). Additional
    data are pulled at the first request and stored for subsequent usage.
    
    Intializes JET version of the Equilibrium object.  Pulls data to
    storage in instance attributes.  Core attributes are populated using SAL classes
    data on initialization.  Additional attributes are initialized as None, 
    filled on the first request to the object.

    Args:
        shot (integer): JET shot index.
    
    Keyword Args:
        shotfile (string): Optional input for alternate shotfile, defaults to 'EQH'
            (i.e., CLISTE results are in EQH,EQI with other reconstructions
            Available (FPP, EQE, ect.).
        edition (integer): Describes the edition of the shotfile to be used
        shotfile2 (string): Describes companion 0D equilibrium data, will automatically
            reference based off of shotfile, but can be manually specified for 
            unique reconstructions, etc.
        length_unit (string): Sets the base unit used for any quantity whose
            dimensions are length to any power. Valid options are:
                
                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================
                
            Default is 'm' (all units taken and returned in meters).
        tspline (Boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic (Boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
        experiment: Used to describe the work space that the shotfile is located
            It defaults to 'AUGD' but can be set to other values
    """

    def __init__(
        self,
        shot,
        user=None,
        dda=None,
        sequence=None,
        length_unit="m",
        tspline=False,
        monotonic=True,
    ):

        if not _has_sal:
            print("sal module did not load properly")
            print("Most functionality will not be available!")

        super(JETSALData, self).__init__(
            length_unit=length_unit, tspline=tspline, monotonic=monotonic
        )

        self._DDA_PATH = "/pulse/{}/ppf/signal/{}/{}:{}"
        self._DATA_PATH = "/pulse/{}/ppf/signal/{}/{}/{}:{}"

        # defaults
        user = user or "jetppf"
        dda = dda or "efit"
        sequence = sequence or 0

        self._shot = shot
        self.user = user
        self.dda = dda

        # identify the current head sequence number if seq = 0 to ensure all data from same sequence
        # this should mitigate the very low probability event of new data being written part way through the read
        if sequence == 0:
            r = sal.list(self._DDA_PATH.format(self._shot, user, dda, sequence))
            sequence = r.revision_latest
        self.sequence = sequence
        self._defaultUnits = {}

        # initialize None for non-essential data

        # grad-shafranov related parameters
        self._fpol = None
        self._fluxPres = None  # pressure on flux surface (psi,t)
        self._ffprim = None
        self._pprime = None  # pressure derivative on flux surface (t,psi)

        # fields
        self._btaxp = None  # Bt on-axis, with plasma (t)
        self._btaxv = None  # Bt on-axis, vacuum (t)
        self._bpolav = None  # avg poloidal field (t)
        self._BCentr = None  # Bt at RCentr, vacuum (for gfiles) (t)

        # plasma current
        self._IpCalc = None  # calculated plasma current (t)
        self._IpMeas = None  # measured plasma current (t)
        self._Jp = None  # grid of current density (r,z,t)
        self._currentSign = (
            None
        )  # sign of current for entire shot (calculated in moderately kludgey manner)

        # safety factor parameters
        self._q0 = None  # q on-axis (t)
        self._q95 = None  # q at 95% flux (t)
        self._qLCFS = None  # q at LCFS (t)
        self._rq1 = None  # outboard-midplane minor radius of q=1 surface (t)
        self._rq2 = None  # outboard-midplane minor radius of q=2 surface (t)
        self._rq3 = None  # outboard-midplane minor radius of q=3 surface (t)

        # shaping parameters
        self._kappa = None  # LCFS elongation (t)
        self._dupper = None  # LCFS upper triangularity (t)
        self._dlower = None  # LCFS lower triangularity (t)

        # (dimensional) geometry parameters
        self._rmag = None  # major radius, magnetic axis (t)
        self._zmag = None  # Z magnetic axis (t)
        self._aLCFS = None  # outboard-midplane minor radius (t)
        self._RmidLCFS = None  # outboard-midplane major radius (t)
        self._areaLCFS = None  # LCFS surface area (t)
        self._RLCFS = None  # R-positions of LCFS (t,n)
        self._ZLCFS = None  # Z-positions of LCFS (t,n)
        self._RCentr = None  # Radius for BCentr calculation (for gfiles) (t)

        # machine geometry parameters
        self._Rlimiter = None  # R-positions of vacuum-vessel wall (t)
        self._Zlimiter = None  # Z-positions of vacuum-vessel wall (t)

        # calc. normalized-pressure values
        self._betat = None  # calc toroidal beta (t)
        self._betap = None  # calc avg. poloidal beta (t)
        self._Li = None  # calc internal inductance (t)

        # diamagnetic measurements
        self._diamag = None  # diamagnetic flux (t)
        self._betatd = None  # diamagnetic toroidal beta (t)
        self._betapd = None  # diamagnetic poloidal beta (t)
        self._WDiamag = None  # diamagnetic stored energy (t)
        self._tauDiamag = None  # diamagnetic energy confinement time (t)

        # energy calculations
        self._WMHD = None  # calc stored energy (t)
        self._tauMHD = None  # calc energy confinement time (t)
        self._Pinj = None  # calc injected power (t)
        self._Wbdot = None  # d/dt magnetic stored energy (t)
        self._Wpdot = None  # d/dt plasma stored energy (t)

        # load essential mapping data
        # Set the variables to None first so the loading calls will work right:
        self._time = None  # timebase
        self._psiRZ = None  # flux grid (r,z,t)
        self._rGrid = None  # R-axis (t)
        self._zGrid = None  # Z-axis (t)
        self._psiLCFS = None  # flux at LCFS (t)
        self._psiAxis = None  # flux at magnetic axis (t)
        self._fluxVol = None  # volume within flux surface (t,psi)
        self._volLCFS = None  # volume within LCFS (t)
        self._qpsi = None  # q profile (psi,t)
        self._RmidPsi = None  # max major radius of flux surface (t,psi)

        # AUG SV file flag
        self._SSQ = None

        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        # self.getTimeBase()  # check
        # self._timeidxend = self.getTimeBase().size
        # self.getFluxGrid()  # loads _psiRZ, _rGrid and _zGrid at once. check
        # self.getFluxLCFS()  # check
        # self.getFluxAxis()  # check
        # self.getFluxVol()  # check
        # self._lpf = self.getFluxVol().shape[1]
        # self.getVolLCFS()  # check
        # self.getQProfile()  #

    def __str__(self):
        """string formatting for ASDEX Upgrade Equilibrium class.
        """
        try:
            nt = len(self._time)
            nr = len(self._rGrid)
            nz = len(self._zGrid)

            mes = (
                "JET data for shot "
                + str(self._shot)
                + "\n"
                + "timebase "
                + str(self._time[0])
                + "-"
                + str(self._time[-1])
                + "s in "
                + str(nt)
                + " points\n"
                + str(nr)
                + "x"
                + str(nz)
                + " spatial grid"
            )
            return mes
        except TypeError:
            return "tree has failed data load."

    def getInfo(self):
        """returns namedtuple of shot information
        
        Returns:
            namedtuple containing
                
                =====   ===============================
                shot    JET shot file
                nr      size of R-axis for spatial grid
                nz      size of Z-axis for spatial grid
                nt      size of timebase for flux grid
                =====   ===============================
        """
        try:
            nt = len(self._time)
            nr = len(self._rGrid)
            nz = len(self._zGrid)
        except TypeError:
            nt, nr, nz = 0, 0, 0
            print("tree has failed data load.")

        data = namedtuple("Info", ["shot", "nr", "nz", "nt"])
        return data(shot=self._shot, nr=nr, nz=nz, nt=nt)

    def getTimeBase(self):
        """returns time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._time is None:
            try:
                self.getFluxGrid()
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._time.copy()

    def getFluxGrid(self):
        """returns flux grid.
        
        Note that this method preserves whatever sign convention is used in AFS.
        
        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiRZ is None:
            try:
                self._packed_psi = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "psi", self.sequence
                    )
                )
                self._time = self._packed_psi.dimensions[0].data
                self._defaultUnits["_time"] = str(self._packed_psi.dimensions[0].units)

                # psi grid axis
                _rGrid = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "psir", self.sequence
                    )
                )
                _zGrid = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "psiz", self.sequence
                    )
                )
                self._defaultUnits["_rGrid"] = str.lower(_rGrid.units)
                self._defaultUnits["_zGrid"] = str.lower(_zGrid.units)
                self._psiRZ = self._packed_psi.data.transpose().reshape(
                    _rGrid.data.size,_zGrid.data.size,self._time.size)
                self._rGrid = _rGrid.data
                self._zGrid = _zGrid.data
                self._defaultUnits["_psiRZ"] = "Vs"  # HARDCODED DUE TO CALIBRATED=FALSE

            except:
                raise ValueError("data retrieval failed.")
        return self._psiRZ.copy()

    def getRGrid(self, length_unit=1):
        """returns R-axis.

        Returns:
            rGrid (Array): [nr] array of R-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._rGrid is None:
            raise ValueError("data retrieval failed.")

        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_rGrid"], length_unit
        )
        return unit_factor * self._rGrid.copy()

    def getZGrid(self, length_unit=1):
        """returns Z-axis.

        Returns:
            zGrid (Array): [nz] array of Z-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._zGrid is None:
            raise ValueError("data retrieval failed.")

        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_zGrid"], length_unit
        )
        return unit_factor * self._zGrid.copy()

    def getFluxAxis(self):
        """returns psi on magnetic axis.

        Returns:
            psiAxis (Array): [nt] array of psi on magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiAxis is None:
            try:
                _psi_axis = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "faxs", self.sequence
                    )
                )
                self._psiAxis = _psi_axis.data
                self._defaultUnits["_psiAxis"] = str.lower(_psi_axis.units)
            except:
                raise ValueError("data retrieval failed.")
        return self._psiAxis.copy()

    def getFluxLCFS(self):
        """returns psi at separatrix.

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiLCFS is None:
            try:
                psiLCFSNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "fbnd", self.sequence
                    )
                )
                self._psiLCFS = psiLCFSNode.data
                self._defaultUnits["_psiLCFS"] = str.lower(psiLCFSNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._psiLCFS.copy()

    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface.

        Keyword Args:
            length_unit (String or 3): unit for plasma volume.  Defaults to 3, 
                indicating default volumetric unit (typically m^3).

        Returns:
            fluxVol (Array): [nt,npsi] array of volume within flux surface.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        raise NotImplementedError("getFluxVol not implemented.")

    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS.

        Keyword Args:
            length_unit (String or 3): unit for LCFS volume.  Defaults to 3, 
                denoting default volumetric unit (typically m^3).

        Returns:
            volLCFS (Array): [nt] array of volume within LCFS.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "volm", self.sequence
                    )
                )
                self._volLCFS = volLCFSNode.data
                self._defaultUnits["_volLCFS"] = str.lower(volLCFSNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_volLCFS"], length_unit
        )
        return unit_factor * self._volLCFS.copy()

    def getRmidPsi(self, length_unit=1):
        """returns maximum major radius of each flux surface.

        Keyword Args:
            length_unit (String or 1): unit of Rmid.  Defaults to 1, indicating 
                the default parameter unit (typically m).

        Returns:
            Rmid (Array): [nt,npsi] array of maximum (outboard) major radius of 
            flux surface psi.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getRmidPsi not implemented.")

    def getRLCFS(self, length_unit=1):
        """returns R-values of LCFS position.

        Returns:
            RLCFS (Array): [nt,n] array of R of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._RLCFS is None:
            try:
                rgeo = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "rbnd", self.sequence
                    )
                )
                self._RLCFS = (
                    rgeo.data
                )  # construct a 2d grid of angles, take cos, multiply by radius
                self._defaultUnits["_RLCFS"] = str.lower(rgeo.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_RLCFS"], length_unit
        )
        return unit_factor * self._RLCFS.copy()

    def getZLCFS(self, length_unit=1):
        """returns Z-values of LCFS position.

        Returns:
            ZLCFS (Array): [nt,n] array of Z of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._ZLCFS is None:
            try:
                zgeo = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "zbnd", self.sequence
                    )
                )
                self._ZLCFS = (
                    zgeo.data
                )  # construct a 2d grid of angles, take sin, multiply by radius
                self._defaultUnits["_ZLCFS"] = str.lower(zgeo.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_ZLCFS"], length_unit
        )
        return unit_factor * self._ZLCFS.copy()

    def getF(self):
        """returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov 
        solutions.
        
        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fpol is None:
            try:
                fNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "f", self.sequence
                    )
                )
                self._fpol = fNode.data
                self._defaultUnits["_fpol"] = str("T m")
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._fpol.copy()

    def getFluxPres(self):
        """returns pressure at flux surface.

        Returns:
            p (Array): [nt,npsi] array of pressure on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from AUG AFS system.
        """
        if self._fluxPres is None:
            try:
                fluxPresNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "p", self.sequence
                    )
                )
                self._fluxPres = fluxPresNode.data
                self._defaultUnits["_fluxPres"] = str.lower(fluxPresNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._fluxPres.copy()

    def getFPrime(self):
        """returns F', often calculated for grad-shafranov 
        solutions.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        raise NotImplementedError("self.getFPrime not implemented.")

    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FFprime (Array): [nt,npsi] array of FF' fromgrad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        raise NotImplementedError("self.getFFPrime not implemented.")

    def getPPrime(self):
        """returns plasma pressure gradient as a function of psi.

        Returns:
            pprime (Array): [nt,npsi] array of pressure gradient on flux surface 
            psi from grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        raise NotImplementedError("self.getPPrime not implemented.")

    def getElongation(self):
        """returns LCFS elongation.

        Returns:
            kappa (Array): [nt] array of LCFS elongation.

        Raises:
            ValueError: if module cannot retrieve data from AFS.
        """
        if self._kappa is None:
            try:
                kappaNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "elon", self.sequence
                    )
                )
                self._kappa = kappaNode.data
                self._defaultUnits["_kappa"] = str.lower(kappaNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._kappa.copy()

    def getUpperTriangularity(self):
        """returns LCFS upper triangularity.

        Returns:
            deltau (Array): [nt] array of LCFS upper triangularity.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._dupper is None:
            try:
                dupperNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "triu", self.sequence
                    )
                )
                self._dupper = dupperNode.data
                self._defaultUnits["_dupper"] = str.lower(dupperNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._dupper.copy()

    def getLowerTriangularity(self):
        """returns LCFS lower triangularity.

        Returns:
            deltal (Array): [nt] array of LCFS lower triangularity.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._dlower is None:
            try:
                dlowerNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "tril", self.sequence
                    )
                )
                self._dlower = dlowerNode.data
                self._defaultUnits["_dlower"] = str.lower(dlowerNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._dlower.copy()

    def getShaping(self):
        """pulls LCFS elongation and upper/lower triangularity.
        
        Returns:
            namedtuple containing (kappa, delta_u, delta_l)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        try:
            kap = self.getElongation()
            du = self.getUpperTriangularity()
            dl = self.getLowerTriangularity()
            data = namedtuple("Shaping", ["kappa", "delta_u", "delta_l"])
            return data(kappa=kap, delta_u=du, delta_l=dl)
        except ValueError:
            raise ValueError("data retrieval failed.")

    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius.

        Returns:
            magR (Array): [nt] array of major radius of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._rmag is None:
            try:
                rmagNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "rmag", self.sequence
                    )
                )
                self._rmag = rmagNode.data
                self._defaultUnits["_rmag"] = str.lower(rmagNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_rmag"], length_unit
        )
        return unit_factor * self._rmag.copy()

    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._zmag is None:
            try:
                zmagNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "zmag", self.sequence
                    )
                )
                self._zmag = zmagNode.data
                self._defaultUnits["_zmag"] = str.lower(zmagNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_zmag"], length_unit
        )
        return unit_factor * self._zmag.copy()

    def getAreaLCFS(self, length_unit=2):
        """returns LCFS cross-sectional area.

        Keyword Args:
            length_unit (String or 2): unit for LCFS area.  Defaults to 2, 
                denoting default areal unit (typically m^2).

        Returns:
            areaLCFS (Array): [nt] array of LCFS area.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._areaLCFS is None:
            try:
                areaLCFSNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "area", self.sequence
                    )
                )
                self._areaLCFS = areaLCFSNode.data
                self._defaultUnits["_areaLCFS"] = str.lower(areaLCFSNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        # Units should be cm^2:
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_areaLCFS"], length_unit
        )
        return unit_factor * self._areaLCFS.copy()

    def getAOut(self, length_unit=1):
        """returns outboard-midplane minor radius at LCFS.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            aOut (Array): [nt] array of LCFS outboard-midplane minor radius.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._aLCFS is None:
            try:
                aLCFSNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "rmj0", self.sequence
                    )
                )
                self._aLCFS = aLCFSNode.data - 2.96
                self._defaultUnits["_aLCFS"] = str.lower(aLCFSNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_aLCFS"], length_unit
        )
        return unit_factor * self._aLCFS.copy()

    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius.

        Keyword Args:
            length_unit (String or 1): unit for major radius.  Defaults to 1, 
                denoting default length unit (typically m).
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._RmidOUT is None:
            try:
                _RmidOout = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "rmj0", self.sequence
                    )
                )
                # The units aren't properly stored in the tree for this one!
                # Should be meters.
                self._RmidOUT = _RmidOut.data
                self._defaultUnits["_RmidOUT"] = "m"
            except SALException:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_RmidOUT"], length_unit
        )
        return unit_factor * self._RmidOUT.copy()

    def getGeometry(self, length_unit=None):
        """pulls dimensional geometry parameters.
        
        Returns:
            namedtuple containing (magR,magZ,areaLCFS,aOut,RmidOut)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        try:
            Rmag = self.getMagR(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            Zmag = self.getMagZ(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            AreaLCFS = self.getAreaLCFS(
                length_unit=(length_unit if length_unit is not None else 2)
            )
            aOut = self.getAOut(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            RmidOut = self.getRmidOut(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            data = namedtuple(
                "Geometry", ["Rmag", "Zmag", "AreaLCFS", "aOut", "RmidOut"]
            )
            return data(
                Rmag=Rmag, Zmag=Zmag, AreaLCFS=AreaLCFS, aOut=aOut, RmidOut=RmidOut
            )
        except ValueError:
            raise ValueError("data retrieval failed.")

    def getQProfile(self):
        """returns profile of safety factor q.

        Returns:
            qpsi (Array): [nt,npsi] array of q on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._qpsi is None:
            try:
                qpsiNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "q", self.sequence
                    )
                )
                self._qpsi = qpsiNode.data
                self._defaultUnits["_qpsi"] = str.lower(qpsiNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._qpsi.copy()

    def getQ0(self):
        """returns q on magnetic axis,q0.

        Returns:
            q0 (Array): [nt] array of q(psi=0).

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._q0 is None:
            try:
                q0Node = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "qax", self.sequence
                    )
                )
                self._q0 = q0Node.data
                self._defaultUnits["_q0"] = str.lower(q0Node.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._q0.copy()

    def getQ95(self):
        """returns q at 95% flux surface.

        Returns:
            q95 (Array): [nt] array of q(psi=0.95).

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._q95 is None:
            try:
                q95Node = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "q95", self.sequence
                    )
                )
                self._q95 = q95Node.data
                self._defaultUnits["_q95"] = str.lower(q95Node.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._q95.copy()

    def getQLCFS(self):
        """returns q on LCFS (interpolated).
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._qLCFS is None:
            try:
                qLCFSNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "qwl", self.sequence
                    )
                )
                self._qLCFS = qLCFSNode.data
                self._defaultUnits["_LCFS"] = str(qLCFSNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._qLCFS.copy()

    def getQ1Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=1 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ1Surf not implemented.")

    def getQ2Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=2 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ2Surf not implemented.")

    def getQ3Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=3 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ3Surf not implemented.")

    def getQs(self, length_unit=1):
        """pulls q values.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQs not implemented.")

    def getBtVac(self):
        """Returns vacuum toroidal field on-axis. THIS MAY BE INCORRECT

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._btaxv is None:
            try:
                btaxvNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "btax", self.sequence
                    )
                )
                # technically Bave is the average over the volume, but for the core its a singular value
                self._btaxv = btaxvNode.data[:, 0]
                self._defaultUnits["_btaxv"] = str(btaxvNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._btaxv.copy()

    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._btaxp is None:
            try:
                btaxpNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "btax", self.sequence
                    )
                )
                # technically Bave is the average over the volume, but for the core its a singular value
                self._btaxp = btaxpNode.data[:, 0]
                self._defaultUnits["_btaxp"] = str(btaxpNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._btaxp.copy()

    def getBpAvg(self):
        """returns average poloidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getFields not implemented.")

    def getFields(self):
        """pulls vacuum and plasma toroidal field, avg poloidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getFields not implemented.")

    def getIpCalc(self):
        """returns Plasma Current, is the same as getIpMeas.

        Returns:
            IpCalc (Array): [nt] array of the reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._IpCalc is None:
            try:
                IpCalcNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "xipc", self.sequence
                    )
                )
                self._IpCalc = IpCalcNode.data
                self._defaultUnits["_IpCalc"] = str(IpCalcNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._IpCalc.copy()

    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._IpMeas is None:
            try:
                IpMeasNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "xip", self.sequence
                    )
                )
                self._IpMeas = IpMeasNode.data
                self._defaultUnits["_IpMeas"] = str(IpMeasNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._IpMeas.copy()

    def getJp(self):
        """returns the calculated plasma current density Jp on flux grid.

        Returns:
            Jp (Array): [nt,nz,nr] array of current density.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._Jp is None:
            try:
                JpNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "jphi", self.sequence
                    )
                )
                self._Jp = JpNode.data
                self._defaultUnits["_Jp"] = str(JpNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._Jp.copy()

    def getBetaT(self):
        """returns the calculated toroidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._betat is None:
            try:
                BtNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "bttm", self.sequence
                    )
                )
                self._betat = BtNode.data
                self._defaultUnits["_betat"] = BtNode.units
            except SALException:
                raise ValueError("data retrieval failed")
            return self._betat.copy()

    def getBetaP(self):
        """returns the calculated poloidal beta.

        Returns:
            BetaP (Array): [nt] array of the calculated average poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._betap is None:
            try:
                betapNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "btpm", self.sequence
                    )
                )
                self._betap = betapNode.data
                self._defaultUnits["_betap"] = str(betapNode.unit)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._betap.copy()

    def getLi(self):
        """returns the calculated internal inductance.

        Returns:
            Li (Array): [nt] array of the calculated internal inductance.

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._Li is None:
            try:
                LiNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "xlim", self.sequence
                    )
                )
                self._Li = LiNode.data
                self._defaultUnits["_Li"] = str(LiNode.unit)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._Li.copy()

    def getBetas(self):
        """pulls calculated betap, betat, internal inductance.

        """
        try:
            betat = self.getBetaT()
            betap = self.getBetaP()
            Li = self.getLi()
            data = namedtuple("Betas", ["betat", "betap", "Li"])
            return data(betat=betat, betap=betap, Li=Li)
        except ValueError:
            raise ValueError("data retrieval failed.")

    def getDiamagFlux(self):
        """returns the measured diamagnetic-loop flux.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagFlux not implemented.")

    def getDiamagBetaT(self):
        """returns diamagnetic-loop toroidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._betatd is None:
            try:
                BetaTDNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "bttd", self.sequence
                    )
                )
                self._betatd = BetaTDNode.data
                self._defaultUnits["_betatd"] = str(BetaTDNode.unit)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._betatd.copy()

    def getDiamagBetaP(self):
        """returns diamagnetic-loop avg poloidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._betapd is None:
            try:
                BetaPDNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "btpd", self.sequence
                    )
                )
                self._betatd = BetaPDNode.data
                self._defaultUnits["_betapd"] = str(BetaPDNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._betapd.copy()

    def getDiamagTauE(self):
        """returns diamagnetic-loop energy confinement time.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagTauE not implemented.")

    def getDiamagWp(self):
        """returns diamagnetic-loop plasma stored energy.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._WDiamag is None:
            try:
                Node = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "wdia", self.sequence
                    )
                )
                self._Wdiamag = Node.data
                self._defaultUnits["_Wdiamag"] = Node.units
            except SALException:
                raise ValueError("data retrieval failed")
        return self._Wdiamag.copy()

    def getDiamag(self):
        """pulls diamagnetic flux measurements, toroidal and poloidal beta,
        energy confinement time and stored energy.

        Returns:
            namedtuple containing (diamag. flux, betatd, betapd, tauDiamag, WDiamag)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            dFlux = self.getDiamagFlux()
            betatd = self.getDiamagBetaT()
            betapd = self.getDiamagBetaP()
            dWp = self.getDiamagWp()
            data = namedtuple("Diamag", ["diaFlux", "diaBetat", "diaBetap", "diaWp"])
            return data(diaFlux=dFlux, diaBetat=betatd, diaBetap=betapd, diaWp=dWp)
        except ValueError:
            raise ValueError("data retrieval failed.")

    def getWMHD(self):
        """returns calculated MHD stored energy.

        Returns:
            WMHD (Array): [nt] array of the calculated stored energy.

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._WMHD is None:
            try:
                WMHDNode = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "wp", self.sequence
                    )
                )
                self._WMHD = WMHDNode.data
                self._defaultUnits["_WMHD"] = str(WMHDNode.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._WMHD.copy()

    def getTauMHD(self):
        """returns the calculated MHD energy confinement time.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getTauMHD not implemented.")

    def getPinj(self):
        """returns the injected power.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
            .
        """
        raise NotImplementedError("self.getPinj not implemented.")

    def getWbdot(self):
        """returns the calculated d/dt of magnetic stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getWbdot not implemented.")

    def getWpdot(self):
        """returns the calculated d/dt of plasma stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getWpdot not implemented.")

    def getBCentr(self):
        """returns Vacuum toroidal magnetic field at center of plasma

        Returns:
            B_cent (Array): [nt] array of B_t at center [T]

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._BCentr is None:
            try:
                Node = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "bvac", self.sequence
                    )
                )
                self._BCentr = Node.data
                self._defaultUnits["_Bcentr"] = str(Node.units)
            except SALException:
                raise ValueError("data retrieval failed.")
        return self._BCentr.copy()

    def getRCentr(self, length_unit=1):
        """Returns Radius of BCenter measurement

        Returns:
            R: Radial position where Bcent calculated [m]
        """
        if self._RCentr is None:
            self._RCentr = 2.96  # Hardcoded from MAI file description of BTF
            self._defaultUnits["_RCentr"] = "m"
        return self._RCentr

    def getEnergy(self):
        """pulls the calculated energy parameters - stored energy, tau_E, 
        injected power, d/dt of magnetic and plasma stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getEnergy not implemented.")

    def getMachineCrossSection(self):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting 
        routines.
        
        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        if self._Rlimiter is None or self._Zlimiter is None:
            try:
                self._Rlimiter = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "rlim", self.sequence
                    )
                ).data[0, :]
                self._Zlimiter = sal.get(
                    self._DATA_PATH.format(
                        self._shot, self.user, self.dda, "zlim", self.sequence
                    )
                ).data[0, :]

            except SALException:
                raise ValueError("data retrieval failed.")
        return (self._Rlimiter, self._Zlimiter)

    def getMachineCrossSectionFull(self):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """
        x, y = self.getMachineCrossSection()
        return (x, y)

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's 
        IDL implementation efit_rz2psi.pro.

        Returns:
            currentSign (Integer): 1 for positive-direction current, -1 for negative.
        """
        if self._currentSign is None:
            self._currentSign = -1 if scipy.mean(self.getIpMeas()) > 1e5 else 1
        return self._currentSign
