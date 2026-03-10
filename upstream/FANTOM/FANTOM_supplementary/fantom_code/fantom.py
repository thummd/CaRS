# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader


from deci import process_adjacency_mats
from helper_functions import to_tensors
from nri_utils import convert_temporal_to_static_adjacency_matrix
from base_distributions import TemporalConditionalSplineFlow, TemporalSplineFlow
from deci import DECI
from generation_functions import TemporalContractiveInvertibleGNN
from variational_distributions import AdjMatrix, TemporalThreeWayGrahpDist


class FANTOM_stationary(DECI):
    

    def __init__(
        self,
        num_nodes,
        device: torch.device,
        lag: int,
        allow_instantaneous: bool,
        lambda_dag: float = 1.0,
        lambda_sparse: float = 1.0,
        lambda_sparse_l2: float = 0.0,
        l2_group_mode: str = "column",
        lambda_prior: float = 1.0,
        tau_gumbel: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        var_dist_A_mode: str = "temporal_three",
        norm_layers: bool = True,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        cate_rff_n_features: int = 3000,
        cate_rff_lengthscale: Union[float, List[float], Tuple[float, float]] = (
            0.1,
            1.0,
        ),
        prior_A_confidence: float = 0.5,
        graph_constraint_matrix: Optional[np.ndarray] = None,
        ICGNN_embedding_size: Optional[int] = None,
        init_logits: Optional[List[float]] = None,
        conditional_embedding_size: Optional[int] = None,
        conditional_encoder_layer_sizes: Optional[List[int]] = None,
        conditional_decoder_layer_sizes: Optional[List[int]] = None,
        conditional_spline_order: str = "quadratic",
        additional_spline_flow: int = 0,
        disable_diagonal_eval: bool = True,
        heteroscedastic: bool = True
    ):
        
        # Assertions: (1) lag>0, (2) imputation==False
        assert lag > 0, "The lag must be greater than 0."
        

    
        self.allow_instantaneous = allow_instantaneous
        self.init_logits = init_logits
        self.lag = lag
        self.num_nodes = num_nodes
        self.cts_node = [i for i in range (self.num_nodes)]
        self.continuous_range = [i for i in range(num_nodes)]
        self.cts_dim = len(self.cts_node)
        # conditional spline flow hyper-params.
        self.conditional_embedding_size = conditional_embedding_size
        self.conditional_encoder_layer_sizes = conditional_encoder_layer_sizes
        self.conditional_decoder_layer_sizes = conditional_decoder_layer_sizes
        self.conditional_spline_order = conditional_spline_order
        self.additional_spline_flow = additional_spline_flow
        self.group_mask  = np.eye(self.num_nodes, dtype=bool)
        self.heteroscedastic = heteroscedastic
        # For V0 AR-DECI, we only support mode_adjacency="learn", so hardcoded this argument.
        super().__init__(
            num_nodes=num_nodes,
            device=device,
            lambda_dag=lambda_dag,
            lambda_sparse=lambda_sparse,
            lambda_sparse_l2=lambda_sparse_l2,
            l2_group_mode=l2_group_mode,
            lambda_prior=lambda_prior,
            tau_gumbel=tau_gumbel,
            base_distribution_type=base_distribution_type,
            spline_bins=spline_bins,
            var_dist_A_mode=var_dist_A_mode,
            mode_adjacency="learn",
            norm_layers=norm_layers,
            res_connection=res_connection,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            cate_rff_n_features=cate_rff_n_features,
            cate_rff_lengthscale=cate_rff_lengthscale,
            prior_A=None,
            prior_A_confidence=prior_A_confidence,
            prior_mask=None,
            graph_constraint_matrix=graph_constraint_matrix,
            embedding_size=ICGNN_embedding_size,
            disable_diagonal_eval=disable_diagonal_eval,
        )

    def _generate_error_likelihoods(self, base_distribution_string: str) -> Dict[str, nn.Module]:
       
        error_likelihoods = super()._generate_error_likelihoods(
            base_distribution_string if base_distribution_string != "conditional_spline" and base_distribution_string != "coupling_spline" else "spline"
        )

        if base_distribution_string == "conditional_spline":
            error_likelihoods["continuous"] = TemporalConditionalSplineFlow(
                cts_node=self.cts_node,
                group_mask=torch.tensor(self.group_mask).to(self.device),
                device=self.device,
                lag=self.lag,
                num_bins=self.spline_bins,
                additional_flow=self.additional_spline_flow,
                layers_g=self.conditional_encoder_layer_sizes,
                layers_f=self.conditional_decoder_layer_sizes,
                embedding_size=self.conditional_embedding_size,
                order=self.conditional_spline_order,
            )
        if base_distribution_string == "coupling_spline":
            error_likelihoods["continuous"] = TemporalSplineFlow(
                                                        cts_node=self.cts_node,
                                                        group_mask=torch.tensor(self.group_mask).to(self.device),
                                                        device=self.device,
                                                        lag=self.lag,
                                                        num_bins=self.spline_bins,
                                                        additional_flow=self.additional_spline_flow,
                                                        layers_g=self.conditional_encoder_layer_sizes,
                                                        layers_f=self.conditional_decoder_layer_sizes,
                                                        embedding_size=self.conditional_embedding_size,
                                                        order=self.conditional_spline_order,
                                                    )
        
        return error_likelihoods

    def _create_var_dist_A_for_deci(self, var_dist_A_mode: str) -> Optional[AdjMatrix]:
        """
        This overwrites the original DECI one to generate a variational distribution supporting the temporal adj matrix.
        Args:
            var_dist_A_mode: the type of the variational distribution

        Returns:
            An instance of variational distribution.
        """
        assert (
            var_dist_A_mode == "temporal_three"
        ), f"Currently, var_dist_A only support type temporal_three, but {var_dist_A_mode} given"
        var_dist_A = TemporalThreeWayGrahpDist(
            device=self.device,
            input_dim=self.num_nodes,
            lag=self.lag,
            tau_gumbel=self.tau_gumbel,
            init_logits=self.init_logits,
        )
        return var_dist_A

    def _create_ICGNN_for_deci(self) -> nn.Module:
        """
        This overwrites the original one in DECI to generate an ICGNN that supports the auto-regressive formulation.

        Returns:
            An instance of the temporal ICGNN
        """

        return TemporalContractiveInvertibleGNN(
            group_mask=torch.tensor(self.group_mask),
            lag=self.lag,
            device=self.device,
            norm_layer=self.norm_layer,
            res_connection=self.res_connection,
            encoder_layer_sizes=self.encoder_layer_sizes,
            decoder_layer_sizes=self.decoder_layer_sizes,
            embedding_size=self.embedding_size,
            heteroscedastic=self.heteroscedastic,
        )

    def networkx_graph(self) -> nx.DiGraph:
        """
        This function converts the most probable graph to networkx graph. Due to the incompatibility of networkx and temporal
        adjacency matrix, we need to convert the temporal adj matrix to its static version before changing it to networkx graph.
        """
        adj_mat = self.get_adj_matrix(
            samples=1, most_likely_graph=True, squeeze=True
        )  # shape [lag+1, num_nodes, num_nodes]
        # Convert to static graph
        static_adj_mat = convert_temporal_to_static_adjacency_matrix(adj_mat, conversion_type="full_time", fill_value=0)
        # Check if non DAG adjacency matrix
        assert np.trace(scipy.linalg.expm(static_adj_mat)) == (self.lag + 1) * self.num_nodes, "Generate non DAG graph"
        return nx.convert_matrix.from_numpy_matrix(static_adj_mat, create_using=nx.DiGraph)

    def sample_graph_posterior(self, do_round: bool = True, samples: int = 100) -> Tuple[List[nx.DiGraph], np.ndarray]:
        """
        This function samples the graph from the variational posterior and convert them into networkx graph without duplicates.
        Due to the incompatibility of temporal adj matrix and networkx graph, they will be converted to its corresponding
        static adj before changing them to networkx graph.
        Args:
            do_round: If we round the probability during sampling.
            samples: The number of sampled graphs.

        Returns:
            A list of networkx digraph object.

        """

        adj_mats = self.get_adj_matrix(
            do_round=do_round, samples=samples, most_likely_graph=False
        )  # shape [samples, lag+1, num_nodes, num_nodes]
        # Convert to static graph
        static_adj_mats = convert_temporal_to_static_adjacency_matrix(
            adj_mats, conversion_type="full_time", fill_value=0
        )
        adj_mats, adj_weights = process_adjacency_mats(static_adj_mats, (self.lag + 1) * self.num_nodes)
        graph_list = [nx.convert_matrix.from_numpy_matrix(adj_mat, create_using=nx.DiGraph) for adj_mat in adj_mats]
        return graph_list, adj_weights

    

    def set_graph_constraint(self, graph_constraint_matrix: Optional[np.ndarray]):
        
        if graph_constraint_matrix is None:
            neg_constraint_matrix = np.ones((self.lag + 1, self.num_nodes, self.num_nodes))
            if self.allow_instantaneous:
                np.fill_diagonal(neg_constraint_matrix[0, ...], 0)
            else:
                neg_constraint_matrix[0, ...] = np.zeros((self.num_nodes, self.num_nodes))
            self.neg_constraint_matrix = torch.as_tensor(neg_constraint_matrix, device=self.device, dtype=torch.float32)
            self.pos_constraint_matrix = torch.zeros((self.lag + 1, self.num_nodes, self.num_nodes), device=self.device)
        else:
            negative_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=1.0)
            if not self.allow_instantaneous:
                negative_constraint_matrix[0, ...] = np.zeros((self.num_nodes, self.num_nodes))
            self.neg_constraint_matrix = torch.as_tensor(
                negative_constraint_matrix, device=self.device, dtype=torch.float32
            )
            # Disable diagonal elements in the instant graph constraint.
            torch.diagonal(self.neg_constraint_matrix[0, ...]).zero_()
            positive_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=0.0)
            self.pos_constraint_matrix = torch.as_tensor(
                positive_constraint_matrix, device=self.device, dtype=torch.float32
            )

    def dagness_factor(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the DAGness loss for a temporal adjacency matrix. Since the only possible violation of DAGness is the
        instantaneous adj A[0,...], we only need to check this dagness.
        Args:
            A: the temporal adjacency matrix with shape [lag+1, num_nodes, num_nodes].

        Returns: A DAGness loss tensor
        """
        # Check shape
        assert A.dim() == 3
        return super().dagness_factor(A[0, ...])

    def _log_prob(
        self,
        x: torch.Tensor,
        predict: torch.Tensor,
        var_pred: torch.Tensor,
        W: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        
        # The overall code structure should be similar to original DECI, but the key difference is the data are now in
        # temporal format with shape [N, lag+1, proc_dims], where data[:,-1,:] represents the data in current time step.
        # From the formulation of AR-DECI, we only care about the conditional log prob p(x_t|x_{<t}). So, the
        # predict from SEM should have shape [N, proc_dims] or [proc_dims]. Then, we can compute data[,-1,:] - predict to get the log probability.

        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, lag+1, proc_dim]
        batch_size, _, proc_dim = x.shape

        if predict.dim() == 1:
            predict = predict.unsqueeze(0)

        # Continuous
        cts_bin_log_prob = torch.zeros(batch_size, proc_dim).to(self.device)  # [batch, proc_dim]
        continuous_range = self.continuous_range
        if continuous_range:
            # History-dependent noise
            if self.base_distribution_type == "conditional_spline" or self.base_distribution_type == "coupling_spline":
                assert W is not None
                cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                    x[..., -1, continuous_range] - predict[..., continuous_range],
                    X_history=x, 
                    W=W,
                )#[..., 0:-1, :],
            else:
                if self.heteroscedastic:
                    cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                        (x[..., -1, continuous_range] - predict[..., continuous_range])/var_pred
                    )
                else:
                    cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                    x[..., -1, continuous_range] - predict[..., continuous_range]
                )

        log_prob = cts_bin_log_prob.sum(-1)  # [1] or [batch]

        return log_prob
    
    def _ELBO_terms(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes all terms involved in the ELBO.

        Args:
            X: Batched samples from the dataset, size (batch_size, input_dim).

        Returns:
            Dict[key, torch.Tensor] containing all the terms involved in the ELBO.
        """
        # Get adjacency matrix with weights
        A_sample = self.get_adj_matrix_tensor(do_round=False, samples=1, most_likely_graph=False).squeeze(0)
        if self.mode_adjacency == "learn":
            factor_q = 1.0
        elif self.mode_adjacency in ["upper", "lower"]:
            factor_q = 0.0
        else:
            raise NotImplementedError(f"Adjacency mode {self.mode_adjacency} not implemented")
        W_adj = A_sample * self.ICGNN.get_weighted_adjacency()
        if self.heteroscedastic:
            predict,var_est = self.ICGNN.predict(X, W_adj)
        else:
            predict = self.ICGNN.predict(X, W_adj)
            var_est = None
        log_p_A = self._log_prior_A(A_sample)  # A number
        penalty_dag = self.dagness_factor(A_sample)  # A number
        log_p_base = self._log_prob(
            X,
            predict,
            var_est,
            W=A_sample if self.base_distribution_type == "conditional_spline" or self.base_distribution_type == "coupling_spline" else None,
        )  # (B)
        log_q_A = -self.var_dist_A.entropy()  # A number
        cts_mse = self._icgnn_cts_mse(X, predict)  # (B)

        return {
            "penalty_dag": penalty_dag,
            "log_p_A": log_p_A,
            "log_p_base": log_p_base,
            "log_q_A": log_q_A * factor_q,
            "cts_mse": cts_mse,
        }

    def _icgnn_cts_mse(self, x: torch.Tensor, predict: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean-squared error (MSE) of the ICGNN on the continuous variables of the model.

        Args:
            x: a temporal data tensor with shape [N, lag+1, proc_dims] (proc_dims may not be equal to num_nodes with categorical variables)
            or [lag+1, proc_dims]
            predict: predictions from SEM with shape [N, proc_dims] or [proc_dims]

        Returns:
            MSE of ICGNN predictions on continuous variables. A number if x has shape (lag+1, proc_dims), or an array of
            shape (N) is X has shape (N, lag+1, proc_dim).
        """
    
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, lag+1, proc_dim]

        if predict.dim() == 1:
            predict = predict.unsqueeze(0)

        continuous_range = self.continuous_range
        return (x[..., -1, continuous_range] - predict[..., continuous_range]).pow(2).sum(-1)

    def _sample_base(self, Nsamples: int, time_span: int = 1) -> torch.Tensor:
        """
        This method draws noise samples from the base distribution with simulation time_span.
        Args:
            Nsamples: The batch size of samples.
            time_span: The simulation time span.

        Returns: A tensor with shape [Nsamples, time_span, proc_dims]
        """

        

        sample = torch.zeros((Nsamples, time_span, self.processed_dim_all), device=self.device)
        total_size = np.prod(sample.shape[:-1])

        # Continuous and binary
        for type_region in ["continuous", "binary"]:
            range_ = self.continuous_range
            if range_:
                if self.base_distribution_type == "conditional_spline" and type_region == "continuous":
                    sample[..., range_] = (
                        self.likelihoods[type_region].base_dist.sample([total_size]).view(*sample.shape[:-1], -1)
                    )  # shape[Nsamples, time_span, cts_dim]
                else:
                    sample[..., range_] = self.likelihoods[type_region].sample(total_size).view(*sample.shape[:-1], -1)


        return sample


    def log_prob(
        self,
        X: torch.Tensor,
        Nsamples_per_graph: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        This computes the log probability of the observations. For V0, does not support intervention.
        Most part is just a copy of parent method, the only difference is that for "conditional_spline", we need to pass
        W to self._log_prob.
        Args:
            X: The observation with shape [N_batch, lag+1, proc_dims]
            Nsamples_per_graph: The number of graph samples.
            most_likely_graph: whether to use the most likely graph. If true, Nsamples should be 1.
            intervention_idxs: Currently not support
            intervention_values: Currently not support

        Returns: a numpy with shape [N_batch]

        """
        # Assert for X shape
        assert X.dim() == 3, "X should be of shape [N_batch, lag+1, proc_dims]"
        # Assertions: intervention_idxs and intervention_values must be None for V0.
        assert intervention_idxs is None, "intervention_idxs is not supported for V0"
        assert intervention_values is None, "intervention_values is not supported for V0"
        # Assertions: Nsamples_per_graph must be 1 if most_likely_graph is true.
        if most_likely_graph:
            assert Nsamples_per_graph == 1, "Nsamples_per_graph should be 1 if most_likely_graph is true"
        (X,) = to_tensors(X, device=self.device, dtype=torch.float)
        var_est = None
        with torch.no_grad():

            log_prob_samples = []

            if most_likely_graph:
                Nsamples_per_graph = 1

            for _ in range(Nsamples_per_graph):

                A_sample = self.get_adj_matrix_tensor(do_round=False, samples=1, most_likely_graph=False).squeeze(0)
                W_adj = A_sample * self.ICGNN.get_weighted_adjacency()

                predict_result = self.ICGNN.predict(X, W_adj)
                # Handle heteroscedastic case where predict returns (prediction, variance)
                if self.heteroscedastic:
                    predict, var_est = predict_result
                else:
                    predict = predict_result
                # Note that the W input is for AR-DECI, DECI will not use W.
                W = A_sample if self.base_distribution_type == "conditional_spline" else None
                log_prob_samples.append(self._log_prob(X, predict, var_est, W=W))  # (B)

            log_prob = torch.logsumexp(torch.stack(log_prob_samples, dim=0), dim=0) - np.log(Nsamples_per_graph)
            return log_prob.detach().cpu().numpy().astype(np.float64)

        #return super().log_prob(
        #    X=X,
        #    Nsamples_per_graph=Nsamples_per_graph,
        #    most_likely_graph=most_likely_graph,
        #)

    def get_params_variational_distribution(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For V0, we do not support missing values, so there is no imputer. Raise NotImplementedError for now.

        """
        raise NotImplementedError


    def set_prior_A(
        self,
        prior_A: Optional[Union[np.ndarray, torch.Tensor]],
        prior_mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> None:
        """
        Overwrite its parent method.
        Set the soft priors for AR-DECI. The prior_A is a soft prior in the temporal format with shape [lag+1, num_nodes, num_nodes].
        prior_mask is a binary mask for soft prior, where mask_{ij}=1 means the value in prior_{ij} set to be the prior,
        and prior_{ij}=0 means we ignore the value in prior_{ij}.
        If prior_A = None, the default choice is zero matrix with shape [lag+1, num_nodes, num_nodes]. The mask is the same.
        Args:
            prior_A: a soft prior with shape [lag+1, num_nodes, num_nodes].
            prior_mask: a binary mask for soft prior with the same shape as prior_A

        Returns: No return

        """
        self.exist_prior = False
        # Default prior_A and mask
        self.prior_A = nn.Parameter(
                torch.zeros((self.lag + 1, self.num_nodes, self.num_nodes), device=self.device),
                requires_grad=False,
            )
        self.prior_mask = nn.Parameter(
                torch.zeros((self.lag + 1, self.num_nodes, self.num_nodes), device=self.device),
                requires_grad=False,
            )

    

    def run_train(
        self,
        dataloader,
        num_samples,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        This method implements the training scripts of AR-DECI. This also setup a soft prior (if exists) for the AR-DECI.
        We should also change the termination condition rather than the using num_below_tol in DECI,
        since if we set allow_instantaneous=False, AR-DECI will always be a DAG. By default,
        only 5 training epochs (num_below_tol>=5, it will break the training loop in DECI).
        The evaluation for loss tracker should also be adapted so that it supports temporal adjacency matrix.
        This can be done by calling convert_to_static in FT-DECI to transform the temporal adj to static adj, and re-using the evaluation for DECI.
        Compared to DECI training, the differences are (1) setup a soft prior for AR-DECI, (2) different stopping creterion, (3) evaluation of temporal adjacency matrix for loss tracker.
        Args:
            dataset: the training temporal dataset.
            train_config_dict: the training config dict.
            report_progress_callback: the callback function to report progress.
            run_context: the run context.

        Returns: No return
        """
        # Load the soft prior from dataset if exists and update the prior for AR-DECI by calling self.set_prior_A(...).
        
        # Setup the logging machinery (similar to DECI training).
        # Setup the optimizer (similar to DECI training).
        # Outer optimization loop. Note the termination condition should be changed based on the value we set for allow_instantaneous.
        # Inner loop by calling self.optimize_inner_auglag(...). No change is needed.
        # Update rho, alpha, loss tracker (similar to DECI).
        super().run_train(
            dataloader,
            num_samples,
            train_config_dict=train_config_dict,
            report_progress_callback=report_progress_callback,
        )
        # Save the sampled adjacency matrix
        sampled_probable_adjacency = self.get_adj_matrix(do_round=True, samples=1, most_likely_graph=True, squeeze=True)
