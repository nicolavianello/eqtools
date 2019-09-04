# This program is distributed under the terms of the GNU
# General Purpose License (GPL).
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
This module provides classes inheriting
:py:class:`eqtools.EFIT.EFITTree` for
working with TCV LIUQE Equilibrium.
"""

import scipy
from collections import namedtuple
from .core import PropertyAccessMixin, ModuleWarning, Equilibrium
import warnings
import numpy as np

try:
    import MDSplus
    try:
        from MDSplus._treeshr import TreeException
    except:
        from MDSplus.mdsExceptions import TreeException
    _has_MDS = True
except Exception as _e_MDS:
    if isinstance(_e_MDS, ImportError):
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work.",
                      ModuleWarning)
    else:
        warnings.warn("MDSplus module could not be loaded -- classes that use"
                      "MDSplus for data access will not work. Exception raised"
                      "was of type %s, message was '%s'."
                      % (_e_MDS.__class__, _e_MDS.message),
                      ModuleWarning)
    _has_MDS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mplw
    import matplotlib.gridspec as mplgs
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.ticker import MaxNLocator

except Exception:
    warnings.warn("matplotlib modules could not be loaded -- plotting and gfile"
                  " writing will not be available.",
                  ModuleWarning)

# we need to define the green function area from the polygon
# see http://stackoverflow.com/questions/22678990/how-can-i-calculate-the-area-within-a-contour-in-python-using-the-matplotlib
# see also http://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
# for how to compute the contours without calling matplotlib contours


# def greenArea(vs):
#     a = 0
#     x0, y0 = vs[0]
#     for [x1, y1] in vs[1:]:
#         dx = x1-x0
#         dy = y1-y0
#         a += 0.5*(y0*dx - x0*dy)
#         x0 = x1
#         y0 = y1
#     return a


class TCVLIUQEMATTree(Equilibrium):
    """Inherits :py:class:`eqtools.Equilibrium` class. Machine-specific data
    handling class for TCV Machine. Pulls LIUQUE Matlab version
    data from selected MDS tree
    and shot through tcv_eq TDI funcion,
    stores as object attributes eventually transforming it in the
    equivalent quantity for EFIT. Each  variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for LIUQUE mapping are pulled on initialization (e.g. psirz grid).
    Additional
    data are pulled at the first request and stored for subsequent usage.
    Intializes TCV version of EFITTree object.  Pulls data from MDS tree for
    storage in instance attributes.  Core attributes are populated from the MDS
    tree on initialization.  Additional attributes are initialized as None,
    filled on the first request to the object.

    Args:
        shot (integer): TCV shot index.
    Keyword Args:
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
        remote (Strig): String indicating the MDSplus server to connect to. Eventually this implies
            the module can be used through MDSplus tunneling
    """
    def __init__(self, shot, tree='tcv_shot', length_unit='m', tspline=False, monotonic=True,
                 server='tcvdata.epfl.ch'):

        super(TCVLIUQEMATTree, self).__init__(length_unit=length_unit,
                                              tspline=tspline,
                                              monotonic=monotonic)

        # superceed the definition of MDStree
        self.server = server
        self._MDSTree = MDSplus.Connection(self.server)
        self._MDSTree.openTree(tree, shot)
        self._defaultUnits = {}
        # now we need to define a None all the variables which will be initialized afterwards through
        # different methods
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
        self._currentSign = None  # sign of current for entire shot (calculated in moderately kludgey manner)

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
        self._tauMHD = None  # calc energy confinement time (t)ps
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
        self.getTimeBase()
        self.getFluxGrid()
    # ---  1
    def getInfo(self):
        """returns namedtuple of shot information
        Returns:
            namedtuple containing
                =====   ===============================
                shot    TCV shot index (long)
                tree    LIUQE tree (string)
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
            print('tree has failed data load.')

        data = namedtuple('Info', ['shot', 'tree', 'nr', 'nz', 'nt'])
        return data(shot=self._shot, tree=self._tree, nr=nr, nz=nz, nt=nt)

    # ---  2
    def getTimeBase(self):
        """returns LIUQE time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._time is None:
            try:
                timenode = self._MDSTree.get(r'tcv_eq("time_psi","liuqe.m")')
                self._time = timenode.data()
                self._defaultUnits['_time'] = 's'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._time.copy()

    # ---  3
    def getFluxGrid(self):
        """returns LIUQE flux grid.

        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.
        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
         """

        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.get(r'tcv_eq("psi","liuqe.m")')
                self._psiRZ = psinode.data()  / (2.*scipy.pi)
                self._rGrid = self._MDSTree.get(r'dim_of(tcv_eq("psi","liuqe.m"),0)').data()
                self._zGrid = self._MDSTree.get(r'dim_of(tcv_eq("psi","liuqe.m"),1)').data()
                self._defaultUnits['_psiRZ'] = str(psinode.units)
                self._defaultUnits['_rGrid'] = 'm'
                self._defaultUnits['_zGrid'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # the transpose is needed as psi
        # is saved as (R, Z, t) in the pulse file
        return self._psiRZ.copy()

    # ---  4
    def getRGrid(self, length_unit=1):
        """returns LIUQE R-axis.

        Returns:
            rGrid (Array): [nr] array of R-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rGrid is None:
            raise ValueError('data retrieval failed.')
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_rGrid'],
            length_unit)
        return unit_factor * self._rGrid.copy()

    # ---  5
    def getZGrid(self, length_unit=1):
        """returns LIUQE Z-axis.

        Returns:
            zGrid (Array): [nz] array of Z-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zGrid is None:
            raise ValueError('data retrieval failed.')
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_zGrid'],
            length_unit)
        return unit_factor * self._zGrid.copy()

    # ---  6
    def getFluxAxis(self):
        """returns psi on magnetic axis.

        Returns:
            psiAxis (Array): [nt] array of psi on magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiAxis is None:
            try:
                psiAxisNode = self._MDSTree.get(r'tcv_eq("psi_axis","liuqe.m")')
                self._psiAxis = psiAxisNode.data() / (2.*scipy.pi)
                self._defaultUnits['_psiAxis'] = str(psiAxisNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiAxis.copy()

    # ---  7
    def getFluxLCFS(self):
        """returns psi at separatrix. Remember that for LIUQE.M the psi is saved as
        flux - flux(LCFS) so that the flux at the LCFS should be identically zero

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiLCFS is None:
            try:
                self._psiLCFS = np.zeros(self._time.size)
            # try:
            #     psiLCFSNode = self._MDSTree.get(r'tcv_eq("psi_surf","liuqe.m")')
            #     self._psiLCFS = psiLCFSNode.data()/(2*scipy.pi)
                self._defaultUnits['_psiLCFS'] = 'Wb'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiLCFS.copy()

    # ---  8
    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface.

        Keyword Args:
            length_unit (String or 3): unit for plasma volume.  Defaults to 3,
                indicating default volumetric unit (typically m^3).

        Returns:
            fluxVol (Array): [nt,npsi] array of volume within flux surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fluxVol is None:
            try:
                volnode = self._MDSTree.get(r'tcv_eq("vol","liuqe.m")')
                self._fluxVol = volnode.data()
                # Units aren't properly stored in the tree for this one!
                self._defaultUnits['_fluxVol'] = 'm^3'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units are m^3, but aren't stored in the tree!
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_fluxVol'], length_unit)
        return unit_factor * self._fluxVol.copy()

    # ---  9
    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS.

        Keyword Args:
            length_unit (String or 3): unit for LCFS volume.  Defaults to 3,
                denoting default volumetric unit (typically m^3).

        Returns:
            volLCFS (Array): [nt] array of volume within LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree.get(r'tcv_eq("vol_edge","liuqe.m")')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = 'm^3'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()

    # ---  10
    def getRmidPsi(self, length_unit=1):
        """returns maximum major radius of each flux surface.

        Keyword Args:
            length_unit (String or 1): unit of Rmid.  Defaults to 1, indicating
                the default parameter unit (typically m).

        Returns:
            Rmid (Array): [nt,npsi] array of maximum (outboard) major radius of
            flux surface psi.

        Raises:
            Value Error: if module cannot retrieve data from MDS tree.
        """
        if self._RmidPsi is None:
            try:
                RmidPsiNode = self._MDSTree.get(r'tcv_eq("r_out","liuqe.m")')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = str(RmidPsiNode.units)
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_RmidPsi'], length_unit)
        return unit_factor * self._RmidPsi.copy()

    # ---  11
    def getRLCFS(self, length_unit=1):
        """returns R-values of LCFS position.

        Returns:
            RLCFS (Array): [nt,n] array of R of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._RLCFS is None:
            try:
                RLCFSNode = self._MDSTree.get(r'tcv_eq("r_edge","liuqe.m")')
                self._RLCFS = RLCFSNode.data()
                self._defaultUnits['_RLCFS'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_RLCFS'], length_unit)
        return unit_factor * self._RLCFS.copy()

    # ---  12
    def getZLCFS(self, length_unit=1):
        """returns Z-values of LCFS position.

        Returns:
            ZLCFS (Array): [nt,n] array of Z of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._ZLCFS is None:
            try:
                ZLCFSNode = self._MDSTree.get(r'tcv_eq("z_edge","liuqe.m")')
                self._ZLCFS = ZLCFSNode.data()
                self._defaultUnits['_ZLCFS'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_ZLCFS'], length_unit)
        return unit_factor * self._ZLCFS.copy()

    # ---  13
    def getF(self):
        """
        returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov
        solutions.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)
        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fpol is None:
            try:
                fluxFfnode = self._MDSTree.get(r'tcv_eq("rbtor_rho","liuqe.m")')
                self._fpol = fluxFfnode.data()
                self._defaultUnits['_fpol'] = 'T*m'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._fpol.copy()

    # ---  14
    def getFluxPres(self):
        """returns pressure at flux surface. Not implemented. We have pressure
           saved on the same grid of psi

        Returns:
            p (Array): [nt,npsi] array of pressure on flux surface psi.
        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fluxPres is None:
            try:
                fluxPPresNode = self._MDSTree(r'tcv_eq("p_rho","liuqe.m")')
                self._fluxPres = fluxPPresNode.data()
                self._defaultUnits['_fluxPres'] = 'Pa'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._fluxPres.copy()

    # ---  15
    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FFprime (Array): [nt,npsi] array of FF' from
            grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._ffprim is None:
            try:
                fluxFFNode = self._MDSTree.get(r'tcv_eq("ttprime_rho","liuqe.m")')
                self._ffprim = fluxFFnode.data()
                self._defaultUnits['_ffprim'] = 'T*m^4'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._ffprim.copy()

    # ---  16
    def getPPrime(self):
        """returns plasma pressure gradient as a function of psi.

        Returns:
            pprime (Array): [nt,npsi] array of pressure
            gradient on flux surface
            psi from grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # in Liuqe pprime is not given in the appropriate flux
        # surface but it is saved as coefficients
        # ppr_coeffs. So we need to build the derivative as
        # p' = p0 + p1 * phi + p2 * phi^2 + p3 * phi^3 with
        # phi = (psi - psi_edge) / (psi_axis - psi_edge)
        # But conventionally psi_edge on TCV = 0 -->  phi = psi / psi_axis
        if self._pprime is None:
            try:
                fluxPPresNode = self._MDSTree.get(r'tcv_eq("pprime_rho","liuqe.m")')
                self._pprime = fluxPPresNode.data()
                self._defaultUnits['_fluxPres'] = 'Pa/Wb'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._pprime.copy()

    # ---  17
    def getElongation(self):
        """returns LCFS elongation.

        Returns:
            kappa (Array): [nt] array of LCFS elongation.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._kappa is None:
            try:
                kappaNode = self._MDSTree.get(r'tcv_eq("kappa_edge","liuqe.m")')
                self._kappa = kappaNode.data()
                self._defaultUnits['_kappa'] = ' '
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._kappa.copy()

    # ---  18
    def getUpperTriangularity(self):
        """returns LCFS upper triangularity.

        Returns:
            deltau (Array): [nt] array of LCFS upper triangularity.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._dupper is None:
            try:
                dupperNode = self._MDSTree.get(r'tcv_eq("delta_ed_top","liuqe.m")')
                self._dupper = dupperNode.data()
                self._defaultUnits['_dupper'] = str(dupperNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dupper.copy()

    # ---  19
    def getLowerTriangularity(self):
        """returns LCFS lower triangularity.

        Returns:
            deltal (Array): [nt] array of LCFS lower triangularity.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._dlower is None:
            try:
                dlowerNode = self._MDSTree.get(r'tcv_eq("delta_ed_bot","liuqe.m")')
                self._dlower = dlowerNode.data()
                self._defaultUnits['_dlower'] = str(dlowerNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dlower.copy()

    # ---  21
    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius.

        Returns:
            magR (Array): [nt] array of major radius of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rmag is None:
            try:
                rmagNode = self._MDSTree.get(r'tcv_eq("r_axis","liuqe.m")')
                self._rmag = rmagNode.data()
                self._defaultUnits['_rmag'] = 'm'
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()

    # ---  22
    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zmag is None:
            try:
                zmagNode = self._MDSTree.get(r'tcv_eq("z_axis","liuqe.m")')
                self._zmag = zmagNode.data()
                self._defaultUnits['_zmag'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_zmag'], length_unit)
        return unit_factor * self._zmag.copy()

    # ---  23
    def getAreaLCFS(self, length_unit=2):
        """returns LCFS cross-sectional area.

        Keyword Args:
            length_unit (String or 2): unit for LCFS area.  Defaults to 2,
                denoting default areal unit (typically m^2).

        Returns:
            areaLCFS (Array): [nt] array of LCFS area.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._areaLCFS is None:
            try:
                areaLCFSNode = self._MDSTree.get(r'tcv_eq("area_edge","liuqe.m")')
                self._areaLCFS = areaLCFSNode.data()
                self._defaultUnits['_areaLCFS'] = 'm^2'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Units should be cm^2:
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_areaLCFS'], length_unit)
        return unit_factor * self._areaLCFS.copy()

    # ---  24
    def getAOut(self, length_unit=1):
        """
        returns outboard-midplane minor radius at LCFS.
        In LIUQE it is the last values
        of \results::r_max_psi

        Keyword Args:
            length_unit (String or 1): unit for minor radius.
            Defaults to 1,
            denoting default length unit (typically m).

        Returns:
            aOut (Array): [nt] array of LCFS outboard-midplane minor radius.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """

        # defin a simple way to fin the closest index to zero
        if self._aLCFS is None:
            try:
                _dummy = self._MDSTree.get(r'tcv_eq("r_out_mid","liuqe.m")')
                # remember that it is the minor Radius and
                # getRmidPsi() give the absolute value
                RMaj = 0.88/0.996
                self._aLCFS = _dummy.data()[:, _dummy.data().shape[1] - 1] - RMaj
                self._defaultUnits['_aLCFS']='m'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_aLCFS'], length_unit)
        return unit_factor * self._aLCFS.copy()

    # ---  25
    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius. It uses getA

        Keyword Args:
            length_unit (String or 1): unit for major radius.  Defaults to 1,
                denoting default length unit (typically m).

        Returns:
            RmidOut (Array): [nt] array of major radius of LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._RmidLCFS is None:
            try:
                # this variable is not saved in the pulse file.
                # we compute this by adding the Major radius
                # of the machine to the computed AOut()
                # almost 0.88
                RMaj = 0.88/0.996
                self._RmidLCFS = self.getAOut()+RMaj
                # The units aren't properly stored in the tree for this one!
                # Should be meters.
                self._defaultUnits['_RmidLCFS'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits['_RmidLCFS'], length_unit)
        return unit_factor * self._RmidLCFS.copy()

    # ---  27
    def getQProfile(self):
        """returns profile of safety factor q.

        Returns:
            qpsi (Array): [nt,npsi] array of q on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qpsi is None:
            try:
                qpsiNode = self._MDSTree.get(r'tcv_eq("q","liuqe.m")')
                self._qpsi = qpsiNode.data()
                self._defaultUnits['_qpsi'] = str(qpsiNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    # ---  28
    def getQ0(self):
        """returns q on magnetic axis,q0.

        Returns:
            q0 (Array): [nt] array of q(psi=0).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._q0 is None:
            try:
                q0Node = self._MDSTree.get(r'tcv_eq("q_axis","liuqe.m")')
                self._q0 = q0Node.data()
                self._defaultUnits['_q0'] = str(q0Node.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q0.copy()

    # ---  29
    def getQ95(self):
        """returns q at 95% flux surface.

        Returns:
            q95 (Array): [nt] array of q(psi=0.95).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._q95 is None:
            try:
                q95Node = self._MDSTree.get(r'tcv_eq("q_95","liuqe.m")')
                self._q95 = q95Node.data()
                self._defaultUnits['_q95'] = str(q95Node.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q95.copy()

    # ---  30
    def getQLCFS(self):
        """returns q on LCFS (interpolated).

        Returns:
            qLCFS (Array): [nt] array of q* (interpolated).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qLCFS is None:
            try:
                qLCFSNode = self._MDSTree.get(r'tcv_eq("q_edge","liuqe.m")')
                self._qLCFS = qLCFSNode.data()
                self._defaultUnits['_qLCFS'] = str(qLCFSNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._qLCFS.copy()

    # ---  31
    def getQ1Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=1 surface.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1,
                denoting default length unit (typically m).

        Returns:
            qr1 (Array): [nt] array of minor radius of q=1 surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # if self._rq1 is None:
        #     try:
        #         rq1Node = self._MDSTree.getNode(self._root + self._afile + ':aaq1')
        #         self._rq1 = rq1Node.data()
        #         self._defaultUnits['_rq1'] = str(rq1Node.units)
        #     except (TreeException, AttributeError):
        #         raise ValueError('data retrieval failed.')
        # unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq1'], length_unit)
        # return unit_factor * self._rq1.copy()

    # ---  32
    def getQ2Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=2 surface.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1,
                denoting default length unit (typically m).

        Returns:
            qr2 (Array): [nt] array of minor radius of q=2 surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # if self._rq2 is None:
        #     try:
        #         rq2Node = self._MDSTree.getNode(self._root + self._afile + ':aaq2')
        #         self._rq2 = rq2Node.data()
        #         self._defaultUnits['_rq2'] = str(rq2Node.units)
        #     except (TreeException, AttributeError):
        #         raise ValueError('data retrieval failed.')
        # unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq2'], length_unit)
        # return unit_factor * self._rq2.copy()

    # ---  33
    def getQ3Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=3 surface.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1,
                denoting default length unit (typically m).

        Returns:
            qr3 (Array): [nt] array of minor radius of q=3 surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # if self._rq3 is None:
        #     try:
        #         rq3Node = self._MDSTree.getNode(self._root + self._afile + ':aaq3')
        #         self._rq3 = rq3Node.data()
        #         self._defaultUnits['_rq3'] = str(rq3Node.units)
        #     except (TreeException, AttributeError):
        #         raise ValueError('data retrieval failed.')
        # unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq3'], length_unit)
        # return unit_factor * self._rq3.copy()

    # ---  34
    def getQs(self, length_unit=1):
        """pulls q values.

        Returns:
            namedtuple containing (q0,q95,qLCFS,rq1,rq2,rq3).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # try:
        #     q0 = self.getQ0()
        #     q95 = self.getQ95()
        #     qLCFS = self.getQLCFS()
        #     rq1 = self.getQ1Surf(length_unit=length_unit)
        #     rq2 = self.getQ2Surf(length_unit=length_unit)
        #     rq3 = self.getQ3Surf(length_unit=length_unit)
        #     data = namedtuple('Qs', ['q0', 'q95', 'qLCFS', 'rq1', 'rq2', 'rq3'])
        #     return data(q0=q0, q95=q95, qLCFS=qLCFS, rq1=rq1, rq2=rq2, rq3=rq3)
        # except ValueError:
        #     raise ValueError('data retrieval failed.')

    # ---  35
    def getBtVac(self):
        """Returns vacuum toroidal field on-axis. We use MDSplus.Connection
        for a proper use of the TDI function tcv_eq()

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._btaxv is None:
            try:
                RMaj = 0.88/0.996
                bt = self._MDSTree.get('tcv_eq("BZERO")').data()/RMaj
                btTime = self._MDSTree.get('dim_of(tcv_eq("BZERO"))').data()
                self._btaxv = scipy.interp(self.getTimeBase(), btTime, bt)
                self._defaultUnits['_btaxv'] = 'T'
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxv.copy()

    # ---  36
    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Returns:
            BtPla (Array): [nt] array of toroidal field including plasma effects.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # if self._btaxp is None:
        #     try:
        #         btaxpNode = self._MDSTree.getNode(self._root+self._afile+':btaxp')
        #         self._btaxp = btaxpNode.data()
        #         self._defaultUnits['_btaxp'] = str(btaxpNode.units)
        #     except (TreeException,AttributeError):
        #         raise ValueError('data retrieval failed.')
        # return self._btaxp.copy()
        
    # ---  39
    def getIpCalc(self):
        """returns EFIT-calculated plasma current.

        Returns:
            IpCalc (Array): [nt] array of EFIT-reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpCalc is None:
            try:
                IpCalcNode = self._MDSTree.get('tcv_eq("i_p","liuqe.m")')
                self._IpCalc = IpCalcNode.data()
                self._defaultUnits['_IpCalc'] = str(IpCalcNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpCalc.copy()
        
    # ---  40
    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpMeas is None:
            try:
                # conn = MDSplus.Connection(self.server)
                # conn.openTree('tcv_shot', self._shot)
                # ip = conn.get('tcv_ip()').data()
                # ipTime = conn.get('dim_of(tcv_ip())').data()
                # conn.closeTree(self._tree, self._shot)
                ipNode = self._MDSTree.get(r'\magnetics::iplasma:trapeze')
                ip = ipNode.data()
                ipTime = self._MDSTree.get(r'dim_of(\magnetics::iplasma:trapeze)').data()
                self._IpMeas = scipy.interp(self.getTimeBase(), ipTime, ip)
                self._defaultUnits['_IpMeas'] = 'A'
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpMeas.copy()

    # ---  42
    def getBetaT(self):
        """returns LIUQE-calculated toroidal beta.

        Returns:
            BetaT (Array): [nt] array of LIUQE-calculated average toroidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betat is None:
            try:
                betatNode = self._MDSTree.get('tcv_eq("beta_tor","liuqe.m")')
                self._betat = betatNode.data()
                self._defaultUnits['_betat'] = str(betatNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betat.copy()

    # ---  43
    def getBetaP(self):
        """returns LIUQE-calculated poloidal beta.

        Returns:
            BetaP (Array): [nt] array of LIUQE-calculated average poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betap is None:
            try:
                betapNode = self._MDSTree.get('tcv_eq("beta_pol","liuqe.m")')
                self._betap = betapNode.data()
                self._defaultUnits['_betap'] = str(betapNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betap.copy()

    # ---  44
    def getLi(self):
        """returns LIUQE-calculated internal inductance.

        Returns:
            Li (Array): [nt] array of LIUQE-calculated internal inductance.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Li is None:
            try:
                LiNode = self._MDSTree.get('tcv_eq("l_i_3","liuqe.m")')

                self._Li = LiNode.data()
                self._defaultUnits['_Li'] = str(LiNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Li.copy()

    # ---  45
    def getBetas(self):
        """pulls calculated betap, betat, internal inductance

        Returns:
            namedtuple containing (betat,betap,Li)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            betat = self.getBetaT()
            betap = self.getBetaP()
            Li = self.getLi()
            data = namedtuple('Betas', ['betat', 'betap', 'Li'])
            return data(betat=betat, betap=betap, Li=Li)
        except ValueError:
            raise ValueError('data retrieval failed.')

    # ---  50
    # def getDiamagWp(self):
    #     """returns diamagnetic-loop plasma stored energy.
    #
    #     Returns:
    #         Wp (Array): [nt] array of measured plasma stored energy.
    #
    #     Raises:
    #         ValueError: if module cannot retrieve data from MDS tree.
    #     """
    #     if self._WDiamag is None:
    #         try:
    #             WDiamagNode = self._MDSTree.getNode(self._root+'::total_energy')
    #             self._WDiamag = WDiamagNode.data()
    #             self._defaultUnits['_WDiamag'] = str(WDiamagNode.units)
    #         except (TreeException, AttributeError):
    #             raise ValueError('data retrieval failed.')
    #     return self._WDiamag.copy()

    # ---  53
    def getWMHD(self):
        """returns EFIT-calculated MHD stored energy.

        Returns:
            WMHD (Array): [nt] array of EFIT-calculated stored energy.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._WMHD is None:
            try:
                WMHDNode = elf._Connection.get('tcv_eq("w_mhd","liuqe.m")')
                self._WMHD = WMHDNode.data()
                self._defaultUnits['_WMHD'] = str(WMHDNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WMHD.copy()

    # ---  53
    # def getTauMHD(self):
    #     """returns LIUQE-calculated MHD energy confinement time.
    #
    #     Returns:
    #         tauMHD (Array): [nt] array of LIUQE-calculated energy confinement time.
    #
    #     Raises:
    #         ValueError: if module cannot retrieve data from MDS tree.
    #     """
    #     if self._tauMHD is None:
    #         try:
    #             tauMHDNode = self._MDSTree.getNode(self._root+'::tau_e')
    #             self._tauMHD = tauMHDNode.data()
    #             self._defaultUnits['_tauMHD'] = str(tauMHDNode.units)
    #         except (TreeException, AttributeError):
    #             raise ValueError('data retrieval failed.')
    #     return self._tauMHD.copy()

    def getBCentr(self):
        """returns Vacuum toroidal magnetic field at center of plasma

        Returns:
            B_cent (Array): [nt] array of B_t at center [T]

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """

        if self._BCentr is None:
            try:
                self._BCentr = self.getBtVac() * self.getRCentr()
                self._defaultUnits['_btaxv'] = 'T'
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')

        return self._BCentr

    def getRCentr(self, length_unit=1):
        """Returns Radius of BCenter measurement

        Returns:
            R: Radial position where Bcent calculated [m]
        """
        if self._RCentr is None:
            self._RCentr =  0.88/0.996 #Hardcoded from MAI file description of BTF
            self._defaultUnits['_RCentr'] = 'm'
        return self._RCentr

    # ---  59
    def getMachineCrossSection(self):
        """Pulls TCV cross-section data from tree, converts to plottable
        vector format for use in other plotting routines

        Returns:
            (`x`, `y`)

            * **x** (`Array`) - [n] array of x-values for machine cross-section.
            * **y** (`Array`) - [n] array of y-values for machine cross-section.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # pull cross-section from tree
        try:
            self._Rlimiter = self._MDSTree.get('static("r_t")').data()
            self._Zlimiter = self._MDSTree.get('static("z_t")').data()
        except MDSplus._treeshr.TreeException:
            raise ValueError('data load failed.')

        return (self._Rlimiter,self._Zlimiter)

    # ---  60
    def getMachineCrossSectionPatch(self):
        """Pulls TCV cross-section data from tree, converts it directly to
        a matplotlib patch which can be simply added to the approriate axes
        call in plotFlux()

        Returns:
            tiles matplotlib Patch, vessel matplotlib Patch

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # pull cross-section from tree
        try:
            Rv_in = self._MDSTree.get('static("r_v:in")').data()
            Rv_out = self._MDSTree.get('static("r_v:out")').data()
            Zv_in = self._MDSTree.get('static("z_v:in")').data()
            Zv_out = self._MDSTree.get('static("z_v:out")').data()
        except MDSplus._treeshr.TreeException:
            raise ValueError('data load failed.')

        # this is for the vessel
        verticesIn = [r for r in zip(Rv_in, Zv_in)]
        verticesIn.append(verticesIn[0])
        codesIn = [Path.MOVETO] + (len(verticesIn) - 1) * [Path.LINETO]
        verticesOut = [r for r in zip(Rv_out, Zv_out)][::-1]
        verticesOut.append(verticesOut[0])
        codesOut = [Path.MOVETO] + (len(verticesOut) - 1) * [Path.LINETO]
        vessel_path = Path(verticesIn + verticesOut, codesIn + codesOut)
        vessel_patch = PathPatch(vessel_path, facecolor=(0.6, 0.6, 0.6),
                                 edgecolor='black')
        # this is for the tiles
        x, y = self.getMachineCrossSection()
        verticesIn = [r for r in zip(x, y)][::- 1]
        verticesIn.append(verticesIn[0])
        codesIn = [Path.MOVETO] + (len(verticesIn)-1) * [Path.LINETO]
        verticesOut = [r for r in zip(Rv_in, Zv_in)]
        verticesOut.append(verticesOut[0])
        codesOut = [Path.MOVETO] + (len(verticesOut) - 1) * [Path.LINETO]
        tiles_path = Path(verticesIn + verticesOut, codesIn + codesOut)
        tiles_patch = PathPatch(tiles_path, facecolor=(0.75, 0.75, 0.75),
                                edgecolor='black')

        return (tiles_patch , vessel_patch)

    # ---  61
    def plotFlux(self, fill=True, mask=False):
        """Plots LIQUE TCV flux contours directly from psi grid.
        Returns the Figure instance created and the time slider widget (in case
        you need to modify the callback). `f.axes` contains the contour plot as
        the first element and the time slice slider as the second element.
        Keyword Args:
            fill (Boolean):
                Set True to plot filled contours.  Set False (default)
                to plot white-background
                color contours.
        """
        try:
            psiRZ = self.getFluxGrid()
            rGrid = self.getRGrid(length_unit='m')
            zGrid = self.getZGrid(length_unit='m')
            t = self.getTimeBase()

            RLCFS = self.getRLCFS(length_unit='m')
            ZLCFS = self.getZLCFS(length_unit='m')
        except ValueError:
            raise AttributeError('cannot plot EFIT flux map.')
        try:
            limx, limy = self.getMachineCrossSection()
        except NotImplementedError:
            if self._verbose:
                print('No machine cross-section implemented!')
            limx = None
            limy = None
        try:
            macx, macy = self.getMachineCrossSectionFull()
        except:
            macx = None
            macy = None

        # event handler for arrow key events in plot windows.
        # Pass slider object
        # to update as masked argument using lambda function
        # lambda evt: arrow_respond(my_slider,evt)
        def arrowRespond(slider, event):
            if event.key == 'right':
                slider.set_val(min(slider.val+1, slider.valmax))
            if event.key == 'left':
                slider.set_val(max(slider.val-1, slider.valmin))

        # make time-slice window
        fluxPlot = plt.figure(figsize=(6, 11))
        gs = mplgs.GridSpec(2, 1, height_ratios=[30, 1])
        psi = fluxPlot.add_subplot(gs[0,0])
        psi.set_aspect('equal')
        try:
            tilesP, vesselP = self.getMachineCrossSectionPatch()
            psi.add_patch(tilesP)
            psi.add_patch(vesselP)
        except NotImplementedError:
            if self._verbose:
                print('No machine cross-section implemented!')
        psi.set_xlim([0.6, 1.2])
        psi.set_ylim([-0.8, 0.8])

        timeSliderSub = fluxPlot.add_subplot(gs[1, 0])
        title = fluxPlot.suptitle('')

        # dummy plot to get x,ylims
        psi.contour(rGrid, zGrid, psiRZ[0], 10, colors='k')

        # generate graphical mask for limiter wall
        if mask:
            xlim = psi.get_xlim()
            ylim = psi.get_ylim()
            bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]),
                           (xlim[1], ylim[1]), (xlim[1], ylim[0]),
                           (xlim[0], ylim[0])]
            poly_verts = [(limx[i], limy[i]) for i in
                          range(len(limx) - 1, -1, -1)]

            bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
            poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

            path = mpath.Path(bound_verts + poly_verts,
                              bound_codes + poly_codes)
            patch = mpatches.PathPatch(path,
                                       facecolor='white',
                                       edgecolor='none')

        def updateTime(val):
            psi.clear()
            t_idx = int(timeSlider.val)

            psi.set_xlim([0.5, 1.2])
            psi.set_ylim([-0.8, 0.8])

            title.set_text('LIUQE Reconstruction, $t = %(t).2f$ s' % {'t':t[t_idx]})
            psi.set_xlabel('$R$ [m]')
            psi.set_ylabel('$Z$ [m]')
            if macx is not None:
                psi.plot(macx, macy, 'k', linewidth=3, zorder=5)
            elif limx is not None:
                psi.plot(limx,limy,'k',linewidth=3,zorder=5)
            # catch NaNs separating disjoint sections of R,ZLCFS in mask
            maskarr = scipy.where(scipy.logical_or(RLCFS[t_idx] > 0.0,scipy.isnan(RLCFS[t_idx])))
            RLCFSframe = RLCFS[t_idx,maskarr[0]]
            ZLCFSframe = ZLCFS[t_idx,maskarr[0]]
            psi.plot(RLCFSframe,ZLCFSframe,'r',linewidth=3,zorder=3)
            if fill:
                psi.contourf(rGrid,zGrid,psiRZ[t_idx],50,zorder=2)
                psi.contour(rGrid,zGrid,psiRZ[t_idx],50,colors='k',linestyles='solid',zorder=3)
            else:
                psi.contour(rGrid,zGrid,psiRZ[t_idx],50,colors='k')
            if mask:
                patchdraw = psi.add_patch(patch)
                patchdraw.set_zorder(4)

            psi.add_patch(tilesP)
            psi.add_patch(vesselP)
            psi.set_xlim([0.5, 1.2])
            psi.set_ylim([-0.8, 0.8])

            fluxPlot.canvas.draw()

        timeSlider = mplw.Slider(timeSliderSub,'t index',0,len(t)-1,valinit=0,valfmt="%d")
        timeSlider.on_changed(updateTime)
        updateTime(0)

        plt.ion()
        fluxPlot.show()

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's
        IDL implementation efit_rz2psi.pro.

        Returns:
            currentSign (Integer): 1 for positive-direction current, -1 for negative.
        """
        if self._currentSign is None:
            self._currentSign = -1 if scipy.mean(self.getIpMeas()) > 1e5 else 1
        return self._currentSign


class TCVLIUQEMATTreeProp(TCVLIUQEMATTree, PropertyAccessMixin):
    """TCVLIUQETree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass
