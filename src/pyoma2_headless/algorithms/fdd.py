"""
Frequency Domain Decomposition (FDD) Algorithm Module.
Part of the pyOMA2 package.
Authors:
Dag Pasca
Diego Margoni
"""

from __future__ import annotations

import logging
import typing

from pyoma2_headless.algorithms.base import BaseAlgorithm
from pyoma2_headless.algorithms.data.mpe_params import EFDDMPEParams, FDDMPEParams
from pyoma2_headless.algorithms.data.result import EFDDResult, FDDResult
from pyoma2_headless.algorithms.data.run_params import EFDDRunParams, FDDRunParams
from pyoma2_headless.functions import fdd

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
# FREQUENCY DOMAIN DECOMPOSITION
class FDD(BaseAlgorithm[FDDRunParams, FDDMPEParams, FDDResult, typing.Iterable[float]]):
    """
    Frequency Domain Decomposition (FDD) algorithm for operational modal analysis.
    """

    RunParamCls = FDDRunParams
    MPEParamCls = FDDMPEParams
    ResultCls = FDDResult

    def run(self) -> FDDResult:
        """
        Executes the FDD algorithm on the input data.
        """
        Y = self.data.T
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov

        freq, Sy = fdd.SD_est(Y, Y, self.dt, nxseg, method=method, pov=pov)
        Sval, Svec = fdd.SD_svalsvec(Sy)

        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            S_val=Sval,
            S_vec=Svec,
        )

    def mpe(self, result: FDDResult, sel_freq: typing.List[float], DF: float = 0.1) -> FDDResult:
        """
        Performs Modal Parameter Estimation (mpe) on selected frequencies.
        """
        super().mpe(result)

        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.DF = DF

        S_val = result.S_val
        S_vec = result.S_vec
        freq = result.freq

        Fn_FDD, Phi_FDD = fdd.FDD_mpe(
            Sval=S_val, Svec=S_vec, freq=freq, sel_freq=sel_freq, DF=DF
        )

        result.Fn = Fn_FDD
        result.Phi = Phi_FDD
        return result


# ------------------------------------------------------------------------------
# ENHANCED FREQUENCY DOMAIN DECOMPOSITION EFDD
class EFDD(FDD[EFDDRunParams, EFDDMPEParams, EFDDResult, typing.Iterable[float]]):
    """
    Enhanced Frequency Domain Decomposition (EFDD) Algorithm Class.
    """

    RunParamCls = EFDDRunParams
    MPEParamCls = EFDDMPEParams
    ResultCls = EFDDResult

    def mpe(
        self,
        result: EFDDResult,
        sel_freq: typing.List[float],
        DF1: float = 0.1,
        DF2: float = 1.0,
        cm: int = 1,
        MAClim: float = 0.85,
        sppk: int = 3,
        npmax: int = 20,
    ) -> EFDDResult:
        """
        Performs Modal Parameter Estimation (mpe) on selected frequencies using EFDD results.
        """
        # EFDD does not need the FDD mpe, so we call the base mpe directly.
        BaseAlgorithm.mpe(self, result)

        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.DF1 = DF1
        self.mpe_params.DF2 = DF2
        self.mpe_params.cm = cm
        self.mpe_params.MAClim = MAClim
        self.mpe_params.sppk = sppk
        self.mpe_params.npmax = npmax

        Fn_FDD, Xi_FDD, Phi_FDD, forPlot = fdd.EFDD_mpe(
            result.Sy,
            result.freq,
            self.dt,
            sel_freq,
            self.run_params.method_SD,
            method=self.method,
            DF1=DF1,
            DF2=DF2,
            cm=cm,
            MAClim=MAClim,
            sppk=sppk,
            npmax=npmax,
        )

        result.Fn = Fn_FDD.reshape(-1)
        result.Xi = Xi_FDD.reshape(-1)
        result.Phi = Phi_FDD
        result.forPlot = forPlot
        return result


# ------------------------------------------------------------------------------
# FREQUENCY SPATIAL DOMAIN DECOMPOSITION FSDD
class FSDD(EFDD):
    """
    Frequency-Spatial Domain Decomposition (FSDD) Algorithm Class.
    """
    method: str = "FSDD"


# =============================================================================
# MULTI SETUP
# =============================================================================
class FDD_MS(FDD[FDDRunParams, FDDMPEParams, FDDResult, typing.Iterable[dict]]):
    """
    Frequency Domain Decomposition (FDD) Algorithm for Multi-Setup Analysis.
    """
    RunParamCls = FDDRunParams
    MPEParamCls = FDDMPEParams
    ResultCls = FDDResult

    def run(self) -> FDDResult:
        """
        Executes the FDD algorithm on multi-setup data.
        """
        Y = self.data
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov

        freq, Sy = fdd.SD_PreGER(Y, self.fs, nxseg=nxseg, method=method, pov=pov)
        Sval, Svec = fdd.SD_svalsvec(Sy)

        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            S_val=Sval,
            S_vec=Svec,
        )


class EFDD_MS(EFDD[EFDDRunParams, EFDDMPEParams, EFDDResult, typing.Iterable[dict]]):
    """
    Enhanced Frequency Domain Decomposition (EFDD) Algorithm for Multi-Setup Analysis.
    """
    method = "EFDD"
    RunParamCls = EFDDRunParams
    MPEParamCls = EFDDMPEParams
    ResultCls = EFDDResult

    def run(self) -> FDDResult:
        """
        Executes the EFDD algorithm on multi-setup data.
        """
        Y = self.data
        nxseg = self.run_params.nxseg
        method = self.run_params.method_SD
        pov = self.run_params.pov

        freq, Sy = fdd.SD_PreGER(Y, self.fs, nxseg=nxseg, method=method, pov=pov)
        Sval, Svec = fdd.SD_svalsvec(Sy)

        return self.ResultCls(
            freq=freq,
            Sy=Sy,
            S_val=Sval,
            S_vec=Svec,
        )
