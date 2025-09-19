"""
Stochastic Subspace Identification (SSI) Algorithm Module.
Part of the pyOMA2 package.

Authors:
    Dag Pasca
    Diego Margoni
"""

from __future__ import annotations

import logging
import typing

import numpy as np
from scipy import signal, stats
from tqdm import trange

from pyoma2.algorithms.base import BaseAlgorithm
from pyoma2.algorithms.data.mpe_params import SSIMPEParams
from pyoma2.algorithms.data.result import ClusteringResult, SSIResult
from pyoma2.algorithms.data.run_params import Clustering, FDDRunParams, SSIRunParams
from pyoma2.functions import clus, fdd, gen, ssi

logger = logging.getLogger(__name__)


# =============================================================================
# SINGLE SETUP
# =============================================================================
# (REF)DATA-DRIVEN STOCHASTIC SUBSPACE IDENTIFICATION
class SSI(BaseAlgorithm[SSIRunParams, SSIMPEParams, SSIResult, typing.Iterable[float]]):
    """
    Perform Stochastic Subspace Identification (SSI) on single-setup measurement data.
    """

    RunParamCls = SSIRunParams
    MPEParamCls = SSIMPEParams
    ResultCls = SSIResult
    method: typing.Literal["dat", "cov", "cov_R", "IOcov"] = "cov"

    def run(self) -> SSIResult:
        """
        Execute the SSI-ref algorithm on the provided measurement data.
        """
        Y = self.data.T
        U = self.run_params.U.T if self.run_params.U is not None else None

        br = self.run_params.br
        method_hank = self.run_params.method or self.method
        ordmin = self.run_params.ordmin
        ordmax = self.run_params.ordmax
        step = self.run_params.step
        sc = self.run_params.sc
        hc = self.run_params.hc
        calc_unc = self.run_params.calc_unc
        nb = self.run_params.nb

        if self.run_params.ref_ind is not None:
            ref_ind = self.run_params.ref_ind
            Yref = Y[ref_ind, :]
        else:
            Yref = Y

        freq, Sy = None, None
        if self.run_params.spetrum is True:
            fdd_run_params = self.run_params.fdd_run_params or FDDRunParams()
            freq, Sy = fdd.SD_est(
                Y, Yref, self.dt, fdd_run_params.nxseg, fdd_run_params.method_SD, fdd_run_params.pov
            )

        H, T = ssi.build_hank(
            Y=Y, Yref=Yref, br=br, method=method_hank, calc_unc=calc_unc, nb=nb, U=U
        )

        Obs, A, C, G = ssi.SSI_fast(H, br, ordmax, step=step)

        hc_xi_max = hc["xi_max"]
        Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = ssi.SSI_poles(
            Obs, A, C, ordmax, self.dt, step=step, calc_unc=calc_unc, H=H, T=T, xi_max=hc_xi_max, HC=True
        )

        hc_mpc_lim = hc["mpc_lim"]
        hc_mpd_lim = hc["mpd_lim"]
        hc_CoV_max = hc["CoV_max"]

        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask3, Phis.shape[2]
            )

        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask4, Phis.shape[2]
            )

        if calc_unc and hc_CoV_max is not None:
            Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_CoV_max)
            to_mask = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(
                to_mask, mask5, Phis.shape[2]
            )

        Lab = gen.SC_apply(
            Fns, Xis, Phis, ordmin, ordmax, step, sc["err_fn"], sc["err_xi"], sc["err_phi"]
        )

        return SSIResult(
            Obs=Obs, A=A, C=C, G=G, H=H, Lambds=Lambds, Fn_poles=Fns, Xi_poles=Xis,
            Phi_poles=Phis, Lab=Lab, Fn_poles_std=Fn_std, Xi_poles_std=Xi_std,
            Phi_poles_std=Phi_std, freq=freq, Sy=Sy
        )

    def mpe(
        self,
        result: SSIResult,
        sel_freq: typing.List[float],
        order_in: typing.Union[int, str] = "find_min",
        rtol: float = 5e-2,
    ) -> SSIResult:
        """
        Extract modal parameters at specified frequencies (stationary MPE).
        """
        super().mpe(result)

        self.mpe_params.sel_freq = sel_freq
        self.mpe_params.order_in = order_in
        self.mpe_params.rtol = rtol

        Fn_pol, Xi_pol, Phi_pol, Lab = result.Fn_poles, result.Xi_poles, result.Phi_poles, result.Lab
        step = self.run_params.step
        Fn_pol_std, Xi_pol_std, Phi_pol_std = result.Fn_poles_std, result.Xi_poles_std, result.Phi_poles_std

        Fn, Xi, Phi, order_out, Fn_std, Xi_std, Phi_std = ssi.SSI_mpe(
            sel_freq, Fn_pol, Xi_pol, Phi_pol, order_in, step, Lab=Lab, rtol=rtol,
            Fn_std=Fn_pol_std, Xi_std=Xi_pol_std, Phi_std=Phi_pol_std
        )

        result.order_out, result.Fn, result.Xi, result.Phi = order_out, Fn, Xi, Phi
        result.Fn_std, result.Xi_std, result.Phi_std = Fn_std, Xi_std, Phi_std
        return result

    def add_clustering(self, *clusterings: Clustering) -> None:
        """
        Add clustering configuration(s) to the algorithm.
        """
        self.clusterings = {
            **getattr(self, "clusterings", {}),
            **{alg.name: alg.steps for alg in clusterings},
        }

    def run_all_clustering(self, result: SSIResult) -> SSIResult:
        """
        Execute all added clustering configurations sequentially.
        """
        for name in self.clusterings.keys():
            result = self.run_clustering(result, name=name)
        logger.info("All clustering configurations executed.")
        return result

    def run_clustering(self, result: SSIResult, name: str) -> SSIResult:
        """
        Perform clustering on identified poles using a specified configuration.
        """
        try:
            steps = self.clusterings[name]
        except KeyError as e:
            raise AttributeError(
                f"'{name}' is not a valid clustering configuration. "
                f"Valid names: {list(self.clusterings.keys())}"
            ) from e

        logger.info("Running clustering '%s'...", name)

        Fns, Xis, Lambds, Phis = result.Fn_poles, result.Xi_poles, result.Lambds, result.Phi_poles
        Fn_std, Xi_std, Phi_std = result.Fn_poles_std, result.Xi_poles_std, result.Phi_poles_std

        calc_unc = self.run_params.calc_unc
        ordmin, ordmax, step = self.run_params.ordmin, self.run_params.ordmax, self.run_params.step

        step1, step2, step3 = steps
        freq_lim = step3.freqlim

        hc_dict, sc_dict = step1.hc_dict, step1.sc_dict
        pre_cluster, pre_clus_typ, pre_clus_dist = step1.pre_cluster, step1.pre_clus_typ, step1.pre_clus_dist
        transform = step1.transform

        if freq_lim is not None:
            Fns, mask_flim = gen.HC_freqlim(Fns, freq_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(
                to_mask, mask_flim, Phis.shape[2]
            )

        if step1.hc:
            if hc_dict["xi_max"] is not None:
                Xis, mask2 = gen.HC_damp(Xis, hc_dict["xi_max"])
                to_mask = [Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std]
                Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask2, Phis.shape[2])
            if hc_dict["mpc_lim"] is not None:
                mask3 = gen.HC_MPC(Phis, hc_dict["mpc_lim"])
                to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask3, Phis.shape[2])
            if hc_dict["mpd_lim"] is not None:
                mask4 = gen.HC_MPD(Phis, hc_dict["mpd_lim"])
                to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask4, Phis.shape[2])
            if calc_unc and hc_dict["CoV_max"] is not None:
                Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_dict["CoV_max"])
                to_mask = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(to_mask, mask5, Phis.shape[2])

        if step1.sc:
            Lab = gen.SC_apply(Fns, Xis, Phis, ordmin, ordmax, step, sc_dict["err_fn"], sc_dict["err_xi"], sc_dict["err_phi"])
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, Lab, Phis.shape[2])

        order = np.full(Lambds.shape, np.nan)
        for kk in range(Lambds.shape[1]):
            order[: kk * step, kk] = kk * step

        MPC = np.apply_along_axis(gen.MPC, axis=2, arr=Phis)
        MPD = np.apply_along_axis(gen.MPD, axis=2, arr=Phis)

        non_nan_index = np.argwhere(~np.isnan(Fns.flatten(order="f")))
        features = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std, MPC, MPD, order]
        Fn_fl, Xi_fl, Phi_fl, Lambd_fl, Fn_std_fl, Xi_std_fl, Phi_std_fl, MPC_fl, MPD_fl, order_fl = clus.vectorize_features(features, non_nan_index)

        if pre_cluster:
            data_dict = {"Fns": Fns, "Xis": Xis, "Phis": Phis, "Lambdas": Lambds, "MPC": MPC, "MPD": MPD}
            feat_arr = clus.build_feature_array(pre_clus_dist, data_dict, ordmax, step, transform)
            if pre_clus_typ == "GMM":
                labels_all, dlim = clus.GMM(feat_arr, dist=True) if step3.merge_dist == "deder" else clus.GMM(feat_arr)
            elif pre_clus_typ == "kmeans":
                labels_all = clus.kmeans(feat_arr)
            elif pre_clus_typ == "FCMeans":
                labels_all = clus.FCMeans(feat_arr)
            else:
                raise ValueError(f"Unsupported pre-clustering type: {pre_clus_typ}")
            stab_lab = np.argwhere(labels_all == 0)
            filtered = clus.filter_fl_list([Fn_fl, Xi_fl, Phi_fl, Lambd_fl, Fn_std_fl, Xi_std_fl, Phi_std_fl, MPC_fl, MPD_fl, order_fl], stab_lab)
            Fn_fl, Xi_fl, Phi_fl, Lambd_fl, Fn_std_fl, Xi_std_fl, Phi_std_fl, MPC_fl, MPD_fl, order_fl = filtered

        if step1.hc == "after" or step1.sc == "after":
            list_array1d = [Fn_fl, Xi_fl, Lambd_fl, MPC_fl, MPD_fl, order_fl]
            Fns, Xis, Lambds, MPC, MPD, order = clus.oned_to_2d(list_array1d, order_fl, Fns.shape, step)
            Phis = clus.oned_to_2d([Phi_fl], order_fl, Phis.shape, step)[0]
            if step1.hc == "after":
                if hc_dict["xi_max"] is not None:
                    Xis, mask2 = gen.HC_damp(Xis, hc_dict["xi_max"])
                    to_mask = [Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std]
                    Fns, Lambds, Phis, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask2, Phis.shape[2])
                if hc_dict["mpc_lim"] is not None:
                    mask3 = gen.HC_MPC(Phis, hc_dict["mpc_lim"])
                    to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask3, Phis.shape[2])
                if hc_dict["mpd_lim"] is not None:
                    mask4 = gen.HC_MPD(Phis, hc_dict["mpd_lim"])
                    to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask4, Phis.shape[2])
                if calc_unc and hc_dict["CoV_max"] is not None:
                    Fn_std, mask5 = gen.HC_CoV(Fns, Fn_std, hc_dict["CoV_max"])
                    to_mask = [Fns, Xis, Phis, Lambds, Xi_std, Phi_std]
                    Fns, Xis, Phis, Lambds, Xi_std, Phi_std = gen.applymask(to_mask, mask5, Phis.shape[2])
            if step1.sc == "after":
                Lab = gen.SC_apply(Fns, Xis, Phis, ordmin, ordmax, step, sc_dict["err_fn"], sc_dict["err_xi"], sc_dict["err_phi"])
                to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
                Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, Lab, Phis.shape[2])
            non_nan_index = np.argwhere(~np.isnan(Fns.flatten(order="f")))
            features = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std, MPC, MPD, order]
            Fn_fl, Xi_fl, Phi_fl, Lambd_fl, Fn_std_fl, Xi_std_fl, Phi_std_fl, MPC_fl, MPD_fl, order_fl = clus.vectorize_features(features, non_nan_index)

        dist_feat, weights, sqrtsqr, method = step2.distance, step2.weights, step2.sqrtsqr, step2.algo
        min_size = int(0.1 * ordmax / step) if step2.min_size == "auto" else step2.min_size
        dc, n_clusters = step2.dc, step2.n_clusters

        data_dict = {"Fn_fl": Fn_fl, "Xi_fl": Xi_fl, "Phi_fl": Phi_fl, "Lambda_fl": Lambd_fl, "MPC_fl": MPC_fl, "MPD_fl": MPD_fl}
        dtot = clus.build_tot_dist(dist_feat, data_dict, len(Fn_fl), weights, sqrtsqr)
        dsim = clus.build_tot_simil(dist_feat, data_dict, len(Fn_fl), weights)

        if method == "hdbscan":
            labels_clus = clus.hdbscan(dtot, min_size)
        elif method == "optics":
            labels_clus = clus.optics(dtot, min_size)
        elif method == "hierarc":
            labels_clus = clus.hierarc(dtot, dc, step2.linkage, n_clusters, ordmax, step, Fns, Phis)
        elif method == "spectral":
            labels_clus = clus.spectral(dsim, n_clusters, ordmax)
        elif method == "affinity":
            labels_clus = clus.affinity(dsim)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        post_proc, merge_dist, select = step3.post_proc, step3.merge_dist, step3.select
        if merge_dist == "auto":
            x = dtot[np.triu_indices_from(dtot, k=0)]
            xs = np.linspace(x.min(), x.max(), 500)
            pdf = stats.gaussian_kde(x)(xs)
            merge_dist = xs[signal.argrelmax(pdf)[0][0]]
        elif merge_dist == "deder":
            merge_dist = dlim

        labels = labels_clus.copy()
        unique_labels = set(labels)
        unique_labels.discard(-1)
        clusters = {label: np.where(labels == label)[0] for label in unique_labels}

        for post_i in post_proc:
            if post_i == "fn_med" and calc_unc: clusters, labels = clus.post_fn_med(clusters, labels, (Fn_fl, Fn_std_fl))
            if post_i == "fn_IQR": clusters, labels = clus.post_fn_IQR(clusters, labels, Fn_fl)
            if post_i == "damp_IQR": clusters, labels = clus.post_xi_IQR(clusters, labels, Xi_fl)
            if post_i == "min_size": clusters, labels = clus.post_min_size(clusters, labels, min_size)
            if post_i == "min_size_pctg": clusters, labels = clus.post_min_size_pctg(clusters, labels, step3.min_pctg)
            if post_i == "min_size_kmeans": clusters, labels = clus.post_min_size_kmeans(labels)
            if post_i == "min_size_gmm": clusters, labels = clus.post_min_size_gmm(labels)
            if post_i == "merge_similar": clusters, labels = clus.post_merge_similar(clusters, labels, dtot, merge_dist)
            if post_i == "1xorder": clusters, labels = clus.post_1xorder(clusters, labels, dtot, order_fl)
            if post_i == "MTT": clusters, labels = clus.post_MTT(clusters, labels, (Fn_fl, Xi_fl))
            if post_i == "ABP": clusters, labels = clus.post_adjusted_boxplot(clusters, labels, (Fn_fl, Xi_fl))

        clusters, labels = clus.reorder_clusters(clusters, labels, Fn_fl)
        if freq_lim is not None: clusters, labels = clus.post_freq_lim(clusters, labels, freq_lim, Fn_fl)

        medoids = {label: indices[np.argmin(dtot[np.ix_(indices, indices)].sum(axis=1))] for label, indices in clusters.items() if len(indices) > 0}
        medoid_indices = list(medoids.values())
        medoid_distances = dtot[np.ix_(medoid_indices, medoid_indices)]

        flattened_results = (Fn_fl, Xi_fl, Phi_fl.squeeze(), order_fl)
        Fn_out, Xi_out, Phi_out, order_out = clus.output_selection(select, clusters, flattened_results, medoid_indices)

        Fn_std_out, Xi_std_out, Phi_std_out = (None, None, None)
        if calc_unc:
            flattened_unc = (Fn_std_fl, Xi_std_fl, Phi_std_fl.squeeze(), order_fl)
            Fn_std_out, Xi_std_out, Phi_std_out, _ = clus.output_selection(select, clusters, flattened_unc, medoid_indices)
            Phi_std_out = Phi_std_out.T

        logger.debug("Saving clustering '%s' result.", name)
        risultati = dict(Fn=Fn_out, Xi=Xi_out, Phi=Phi_out.T, Fn_fl=Fn_fl, Xi_fl=Xi_fl, Phi_fl=Phi_fl.squeeze(),
                         Fn_std_fl=Fn_std_fl, Xi_std_fl=Xi_std_fl, Phi_std_fl=Phi_std_fl, order_fl=order_fl, labels=labels,
                         dtot=dtot, medoid_distances=medoid_distances, order_out=order_out, Fn_std=Fn_std_out,
                         Xi_std=Xi_std_out, Phi_std=Phi_std_out)
        result.clustering_results[name] = ClusteringResult(**risultati)
        return result


# =============================================================================
# MULTISETUP
# =============================================================================
class SSI_MS(SSI[SSIRunParams, SSIMPEParams, SSIResult, typing.Iterable[dict]]):
    """
    Perform Stochastic Subspace Identification (SSI) on multi-setup measurement data.
    """
    method: typing.Literal["dat", "cov_R", "cov"] = "cov"

    def run(self) -> SSIResult:
        """
        Execute the SSI algorithm across multiple experimental setups.
        """
        Y = self.data
        br, method_hank = self.run_params.br, self.run_params.method or self.method
        ordmin, ordmax, step = self.run_params.ordmin, self.run_params.ordmax, self.run_params.step
        sc, hc = self.run_params.sc, self.run_params.hc

        Obs, A, C = ssi.SSI_multi_setup(Y, self.fs, br, ordmax, step=1, method_hank=method_hank)

        hc_xi_max = hc["xi_max"]
        Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = ssi.SSI_poles(
            Obs, A, C, ordmax, self.dt, step=step, calc_unc=False, HC=True, xi_max=hc_xi_max
        )

        hc_mpc_lim, hc_mpd_lim = hc["mpc_lim"], hc["mpd_lim"]
        if hc_mpc_lim is not None:
            mask3 = gen.HC_MPC(Phis, hc_mpc_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask3, Phis.shape[2])
        if hc_mpd_lim is not None:
            mask4 = gen.HC_MPD(Phis, hc_mpd_lim)
            to_mask = [Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std]
            Fns, Xis, Phis, Lambds, Fn_std, Xi_std, Phi_std = gen.applymask(to_mask, mask4, Phis.shape[2])

        Lab = gen.SC_apply(Fns, Xis, Phis, ordmin, ordmax, step, sc["err_fn"], sc["err_xi"], sc["err_phi"])

        return SSIResult(
            Obs=Obs, A=A, C=C, H=None, Lambds=Lambds, Fn_poles=Fns, Xi_poles=Xis,
            Phi_poles=Phis, Lab=Lab, Fn_poles_std=Fn_std, Xi_poles_std=Xi_std, Phi_poles_std=Phi_std
        )
