"""
Poly-reference Least Square Frequency Domain (pLSCF) Module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni
"""

from __future__ import annotations

import logging
import typing

from pyoma2_headless.algorithms.data.mpe_params import pLSCFMPEParams
from pyoma2_headless.algorithms.data.result import pLSCFResult
from pyoma2_headless.algorithms.data.run_params import pLSCFRunParams
from pyoma2_headless.functions import fdd, gen, plscf

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
class pLSCF(
    BaseAlgorithm[pLSCFRunParams, pLSCFMPEParams, pLSCFResult, typing.Iterable[float]]
):
    """
    Implementation of the poly-reference Least Square Complex Frequency (pLSCF) algorithm for modal analysis.
    """

    RunParamCls = pLSCFRunParams
    MPEParamCls = pLSCFMPEParams
    ResultCls = pLSCFResult

    def run(self) -> pLSCFResult:
        """
        Execute the pLSCF algorithm to perform modal analysis on the provided data.
        """
        Y = self.data.T
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov
        ordmax = self.run_params.ordmax
        ordmin = self.run_params.ordmin
        sc = self.run_params.sc
        hc = self.run_params.hc

        if method == "per":
            sgn_basf = -1
        elif method == "cor":
            sgn_basf = +1

        freq, Sy = fdd.SD_est(Y, Y, self.dt, nxseg, method=method, pov=pov)

        Ad, Bn = plscf.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)

        Fns, Xis, Phis, Lambds = plscf.pLSCF_poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )

        hc_xi_max = hc["xi_max"]
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]

        Lambds, mask1 = gen.HC_conj(Lambds)
        lista = [Fns, Xis, Phis]
        Fns, Xis, Phis = gen.applymask(lista, mask1, Phis.shape[2])

        Xis, mask2 = gen.HC_damp(Xis, hc_xi_max)
        lista = [Fns, Phis]
        Fns, Phis = gen.applymask(lista, mask2, Phis.shape[2])

        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask3, Phis.shape[2])
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask4, Phis.shape[2])

        Lab = gen.SC_apply(
            Fns,
            Xis,
            Phis,
            ordmin,
            ordmax - 1,
            1,
            sc["err_fn"],
            sc["err_xi"],
            sc["err_phi"],
        )

        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            Ad=Ad,
            Bn=Bn,
            Fn_poles=Fns,
            Xi_poles=Xis,
            Phi_poles=Phis,
            Lab=Lab,
        )

    def mpe(
        self,
        result: pLSCFResult,
        sel_freq: typing.List[float],
        order: typing.Union[int, str] = "find_min",
        rtol: float = 5e-2,
    ) -> pLSCFResult:
        """
        Extract the modal parameters at the selected frequencies and order.
        """
        super().mpe(result)

        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.order_in = order
        self.mpe_params.rtol = rtol

        Fn_pol = result.Fn_poles
        Sm_pol = result.Xi_poles
        Ms_pol = result.Phi_poles
        Lab = result.Lab

        Fn_pLSCF, Xi_pLSCF, Phi_pLSCF, order_out = plscf.pLSCF_mpe(
            sel_freq, Fn_pol, Sm_pol, Ms_pol, order, Lab=Lab, rtol=rtol
        )

        result.order_out = order_out
        result.Fn = Fn_pLSCF
        result.Xi = Xi_pLSCF
        result.Phi = Phi_pLSCF
        return result


# =============================================================================
# MULTI SETUP
# =============================================================================
class pLSCF_MS(pLSCF[pLSCFRunParams, pLSCFMPEParams, pLSCFResult, typing.Iterable[dict]]):
    """
    A multi-setup extension of the pLSCF class.
    """

    RunParamCls = pLSCFRunParams
    MPEParamCls = pLSCFMPEParams
    ResultCls = pLSCFResult

    def run(self) -> pLSCFResult:
        """
        Execute the pLSCF algorithm on multi-setup data.
        """
        Y = self.data
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov
        ordmax = self.run_params.ordmax
        ordmin = self.run_params.ordmin
        sc = self.run_params.sc
        hc = self.run_params.hc

        if method == "per":
            sgn_basf = -1
        elif method == "cor":
            sgn_basf = +1

        freq, Sy = fdd.SD_PreGER(Y, self.fs, nxseg=nxseg, method=method, pov=pov)

        Ad, Bn = plscf.pLSCF(Sy, self.dt, ordmax, sgn_basf=sgn_basf)

        Fns, Xis, Phis, Lambds = plscf.pLSCF_poles(
            Ad, Bn, self.dt, nxseg=nxseg, methodSy=method
        )

        hc_xi_max = hc["xi_max"]
        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]

        Lambds, mask1 = gen.HC_conj(Lambds)
        lista = [Fns, Xis, Phis]
        Fns, Xis, Phis = gen.applymask(lista, mask1, Phis.shape[2])

        Xis, mask2 = gen.HC_damp(Xis, hc_xi_max)
        lista = [Fns, Phis]
        Fns, Phis = gen.applymask(lista, mask2, Phis.shape[2])

        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask3, Phis.shape[2])
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            lista = [Fns, Xis, Phis, Lambds]
            Fns, Xis, Phis, Lambds = gen.applymask(lista, mask4, Phis.shape[2])

        Lab = gen.SC_apply(
            Fns,
            Xis,
            Phis,
            ordmin,
            ordmax - 1,
            1,
            sc["err_fn"],
            sc["err_xi"],
            sc["err_phi"],
        )

        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            Ad=Ad,
            Bn=Bn,
            Fn_poles=Fns,
            Xi_poles=Xis,
            Phi_poles=Phis,
            Lab=Lab,
        )
