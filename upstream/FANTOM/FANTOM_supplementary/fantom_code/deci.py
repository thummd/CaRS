import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


import networkx as nx
import numpy as np
import scipy
import torch
import torch.distributions as td
from torch import nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter


from helper_functions import to_tensors
from base_distributions import (
    DiagonalFlowBase,
    GaussianBase,
    TemporalSplineFlow,
    TemporalConditionalSplineFlow,
)
from generation_functions import ContractiveInvertibleGNN
from variational_distributions import (
    AdjMatrix,
    CategoricalAdjacency,
    DeterministicAdjacency,
    ThreeWayGraphDist,
    VarDistA_Simple,
)


def process_adjacency_mats(adj_mats: np.ndarray, num_nodes: int):
    """
    This processes the adjacency matrix in the format [num, variable, variable]. It will remove the duplicates and non DAG adjacency matrix.
    Args:
        adj_mats (np.ndarry): A group of adjacency matrix
        num_nodes (int): The number of variables (dimensions of the adjacency matrix)

    Returns:
        A list of adjacency matrix without duplicates and non DAG
        A np.ndarray storing the weights of each adjacency matrix.
    """

    # This method will get rid of the non DAG and duplicated ones. It also returns a proper weight for each of the adjacency matrix
    if len(adj_mats.shape) == 2:
        # Single adjacency matrix
        assert (np.trace(scipy.linalg.expm(adj_mats)) - num_nodes) == 0, "Generate non DAG graph"
        return adj_mats, np.ones(1)
    else:
        # Multiple adjacency matrix samples
        # Remove non DAG adjacency matrix
        adj_mats = np.array(
            [adj_mat for adj_mat in adj_mats if (np.trace(scipy.linalg.expm(adj_mat)) - num_nodes) == 0]
        )
        assert np.any(adj_mats), "Generate non DAG graph"
        # Remove duplicated adjacency and aggregate the weights
        adj_mats_unique, dup_counts = np.unique(adj_mats, axis=0, return_counts=True)
        # Normalize the weights
        adj_weights = dup_counts / np.sum(dup_counts)
        return adj_mats_unique, adj_weights


class DECI(torch.nn.Module
):
    """
    Flow-based end-to-end causal model, which does causal discovery using a contractive and
    invertible GNN. The adjacency is a random variable over which we do inference.

    In the DECI model, the causal graph size is determined by the variables *groups* in the dataset. Arrows in the
    graph connect different variable groups. In the current implementation, different variables in the same group
    are assumed to be *independent* conditional upon their parents.
    """

    def __init__(
        self,
        num_nodes,
        device: torch.device,
        lambda_dag: float = 1.0,
        lambda_sparse: float = 1.0,
        lambda_sparse_l2: float = 0.0,
        l2_group_mode: str = "column",
        lambda_prior: float = 1.0,
        tau_gumbel: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        var_dist_A_mode: str = "enco",
        mode_adjacency: str = "learn",
        norm_layers: bool = True,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        cate_rff_n_features: int = 3000,
        cate_rff_lengthscale: Union[int, float, List[float], Tuple[float, float]] = (
            0.1,
            1.0,
        ),
        prior_A: Union[torch.Tensor, np.ndarray] = None,
        prior_A_confidence: float = 0.5,
        prior_mask: Union[torch.Tensor, np.ndarray] = None,
        graph_constraint_matrix: Optional[np.ndarray] = None,
        dense_init: bool = False,
        embedding_size: Optional[int] = None,
        log_scale_init: float = 0,
        disable_diagonal_eval: bool = True,
    ):
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device to load model to.
            imputation: Whether to train an imputation network simultaneously with the DECI network.
            lambda_dag: Coefficient for the prior term that enforces DAG.
            lambda_sparse: Coefficient for the L1 prior term that enforces element-wise sparsity.
            lambda_sparse_l2: Coefficient for the L2 group sparsity penalty (0 = disabled).
            l2_group_mode: Grouping mode for L2 penalty. Options: "column" (incoming edges per node),
                          "row" (outgoing edges per node), "lag" (edges per temporal lag),
                          "frobenius" (full matrix L2 norm).
            lambda_prior: Coefficient for the prior term that enforces prior.
            tau_gumbel: Temperature for the gumbel softmax trick.
            base_distribution_type: which type of base distribution to use for the additive noise SEM
            Options:
                fixed_gaussian: Gaussian with non-leanrable mean of 0 and variance of 1
                gaussian: Gaussian with fixed mean of 0 and learnable variance
                spline: learnable flow transformation which composes an afine layer a spline and another affine layer
            spline_bins: How many bins to use for spline flow base distribution if the 'spline' choice is made
            var_dist_A_mode: Variational distribution for adjacency matrix. Admits {"simple", "enco", "true", "three"}.
                             "simple" parameterizes each edge (including orientation) separately. "enco" parameterizes
                             existence of an edge and orientation separately. "true" uses the true graph.
                             "three" uses a 3-way categorical sample for each (unordered) pair of nodes.
            imputer_layer_sizes: Number and size of hidden layers for imputer NN for variational distribution.
            mode_adjacency: In {"upper", "lower", "learn"}. If "learn", do our method as usual. If
                            "upper"/"lower" fix adjacency matrix to strictly upper/lower triangular.
            norm_layers: bool indicating whether all MLPs should use layer norm
            res_connection:  bool indicating whether all MLPs should use layer norm
            encoder_layer_sizes: Optional list indicating width of layers in GNN encoder MLP
            decoder_layer_sizes: Optional list indicating width of layers in GNN decoder MLP,
            cate_rff_n_features: number of random features to use in functiona pproximation when estimating CATE,
            cate_rff_lengthscale: lengthscale of RBF kernel used when estimating CATE,
            prior_A: prior adjacency matrix,
            prior_A_confidence: degree of confidence in prior adjacency matrix enabled edges between 0 and 1,
            graph_constraint_matrix: a matrix imposing constraints on the graph. If the entry (i, j) of the constraint
                            matrix is 0, then there can be no edge i -> j, if the entry is 1, then an edge i -> j
                            must exist, if the entry is `nan`, then an edge i -> j may be learned.
                            By default, only self-edges are constrained to not exist.
            dense_init: Whether we initialize the initial variational distribution to give dense adjacency matrix.
                        Default is False.
            log_scale_init: initial log scale of Gaussian likelihood
            disable_diagonal_eval: for evaluating aggregated temporal adj matrix only, whether to ignore the diagonal connections after aggregation.

        """
        super().__init__()
        self.disable_diagonal_eval = disable_diagonal_eval

        self.base_distribution_type = base_distribution_type
        self.dense_init = dense_init
        self.embedding_size = embedding_size
        self.device = device
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.lambda_sparse_l2 = lambda_sparse_l2
        self.l2_group_mode = l2_group_mode
        self.lambda_prior = lambda_prior
        self.log_scale_init = log_scale_init

        self.cate_rff_n_features = cate_rff_n_features
        self.cate_rff_lengthscale = cate_rff_lengthscale
        self.tau_gumbel = tau_gumbel
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes

        # DECI treats *groups* as distinct nodes in the graph
        self.num_nodes = num_nodes #variables.num_groups
        self.processed_dim_all = num_nodes #variables.num_processed_non_aux_cols
        self.continuous_range = [i for i in range(num_nodes)]

        # set up soft prior over graphs
        self.set_prior_A(prior_A, prior_mask)
        assert 0 <= prior_A_confidence <= 1
        self.prior_A_confidence = prior_A_confidence

        # Set up the Neural Nets
        self.res_connection = res_connection
        self.norm_layer = nn.LayerNorm if norm_layers else None
        self.ICGNN = self._create_ICGNN_for_deci()

        self.spline_bins = spline_bins
        self.likelihoods = nn.ModuleDict(self._generate_error_likelihoods(self.base_distribution_type))
        self.group_mask  = np.eye(self.num_nodes, dtype=bool)
        

        self.mode_adjacency = mode_adjacency
        self.var_dist_A = self._create_var_dist_A_for_deci(var_dist_A_mode)

        self.set_graph_constraint(graph_constraint_matrix)
        # Adding a buffer to hold the log likelihood. This will be saved with the state dict.
        self.register_buffer("log_p_x", torch.tensor(-np.inf))
        self.log_p_x: torch.Tensor  # This is simply a scalar.
        self.register_buffer("spline_mean_ewma", torch.tensor(0.0))
        self.spline_mean_ewma: torch.Tensor

    def set_prior_A(
        self,
        prior_A: Optional[Union[np.ndarray, torch.Tensor]],
        prior_mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> None:

        """
        This method setup the soft prior for deci. Since this is also responsible for setting up the default prior (prior_A=None),
        this may be overwritten by the subclass. For example, temporal deci model need to overwrite it s.t. the default
        prior is temporal soft prior.
        Args:
            prior_A: The soft prior
            prior_mask: The corresponding mask for prior_A
        """
        self.exist_prior = False
        self.prior_A = nn.Parameter(
                torch.zeros((self.num_nodes, self.num_nodes), device=self.device),
                requires_grad=False,
            )
        self.prior_mask = nn.Parameter(
                torch.zeros((self.num_nodes, self.num_nodes), device=self.device),
                requires_grad=False,
            )

    def _create_var_dist_A_for_deci(self, var_dist_A_mode: str) -> AdjMatrix:
        """
        This method creates the variational distribution for DECI. For any models that inherited from DECI, this can be
        overwritten if the model needs to use a different variational distribution.
        Args:
            var_dist_A_mode: The type of variational distribution

        Returns:
            An instance of the chosen variational distribution
        """
        if var_dist_A_mode == "simple":
            var_dist_A: AdjMatrix = VarDistA_Simple(
                device=self.device, input_dim=self.num_nodes, tau_gumbel=self.tau_gumbel
            )
        elif var_dist_A_mode == "true":
            var_dist_A = DeterministicAdjacency(device=self.device)
        elif var_dist_A_mode == "three":
            var_dist_A = ThreeWayGraphDist(device=self.device, input_dim=self.num_nodes, tau_gumbel=self.tau_gumbel)
        elif var_dist_A_mode == "categorical":
            var_dist_A = CategoricalAdjacency(device=self.device)
        else:
            raise NotImplementedError()

        return var_dist_A

    def _create_ICGNN_for_deci(self) -> ContractiveInvertibleGNN:
        """
        This creates the SEM used for DECI. For models that inherited from DECI, this function may need to be overwritten
        to generate different types of SEM.
        Returns:
            An instance of the ICGNN network
        """
        return ContractiveInvertibleGNN(
            torch.tensor(self.group_mask),
            self.device,
            norm_layer=self.norm_layer,
            res_connection=self.res_connection,
            encoder_layer_sizes=self.encoder_layer_sizes,
            decoder_layer_sizes=self.decoder_layer_sizes,
            embedding_size=self.embedding_size,
        )

    def set_graph_constraint(self, graph_constraint_matrix: Optional[np.ndarray]):
        if graph_constraint_matrix is None:
            self.neg_constraint_matrix = 1.0 - torch.eye(self.num_nodes, device=self.device)
            self.pos_constraint_matrix = torch.zeros((self.num_nodes, self.num_nodes), device=self.device)
        else:
            negative_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=1.0)
            self.neg_constraint_matrix = torch.tensor(negative_constraint_matrix, device=self.device)
            self.neg_constraint_matrix *= 1.0 - torch.eye(self.num_nodes, device=self.device)
            positive_constraint_matrix = np.nan_to_num(graph_constraint_matrix, nan=0.0)
            self.pos_constraint_matrix = torch.tensor(positive_constraint_matrix, device=self.device)

    def networkx_graph(self):

        """
        This samples the most probable graphs and conver to networkx digraph
        Returns:
            A networkx digraph object
        """

        # Return the most probable graph from a fitted DECI model in networkx.digraph form
        adj_mat = self.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
        # Check if non DAG adjacency matrix
        assert (np.trace(scipy.linalg.expm(adj_mat)) - self.num_nodes) == 0, "Generate non DAG graph"
        return nx.convert_matrix.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)

    def sample_graph_posterior(self, do_round: bool = True, samples: int = 100):

        """
        This sample the adjacency matrix from the posterior and convert them to networkx format without duplicates
        Args:
            do_round (bool):  If we round the probability during sampling.
            samples (int): The number of adjacency matrix sampled from the posterior

        Returns:
            A list of networkx digraph object.
        """

        # Return a list of DAG networkx digraph without duplicates
        adj_mats = self.get_adj_matrix(do_round=do_round, samples=samples)
        # Remove the duplicates and non DAG adjacency matrix
        adj_mats, adj_weights = process_adjacency_mats(adj_mats, self.num_nodes)
        graph_list = [nx.convert_matrix.from_numpy_matrix(adj_mat, create_using=nx.DiGraph) for adj_mat in adj_mats]
        return graph_list, adj_weights

    def _generate_error_likelihoods(self, base_distribution_string: str) -> Dict[str, nn.Module]:
        """
        Instantiate error likelihood models for each variable in the SEM.
        For continuous variables, the likelihood is for an additive noise model whose exact form is determined by
        the `base_distribution_string` argument (see below for exact options). For vector-valued continuous variables,
        we assume the error model factorises over dimenions.
        For discrete variables, the likelihood directly parametrises the per-class probabilities.
        For sampling these variables, we rely on the Gumbel max trick. For this to be implemented correctly, this method
        also returns a list of subarrays which should be treated as single categorical variables and processed with a
        `max` operation during sampling.

        Args:
            base_distribution_string: which type of base distribution to use for the additive noise SEM
            Options:
                fixed_gaussian: Gaussian with non-leanrable mean of 0 and variance of 1
                gaussian: Gaussian with fixed mean of 0 and learnable variance
                spline: learnable flow transformation which composes an afine layer a spline and another affine layer
        Returns:
            A dictionary from variable type to the likelihood distribution(s) for variables of that type.
        """

        conditional_dists: Dict[str, nn.Module] = {}
        
        # Continuous
        continuous_range = self.continuous_range
        if continuous_range:
            dist: nn.Module
            if base_distribution_string == "fixed_gaussian":
                dist = GaussianBase(
                    len(continuous_range),
                    device=self.device,
                    train_base=False,
                    log_scale_init=self.log_scale_init,
                )
            elif base_distribution_string == "gaussian":
                dist = GaussianBase(
                    len(continuous_range),
                    device=self.device,
                    train_base=True,
                    log_scale_init=self.log_scale_init,
                )
            elif base_distribution_string == "spline":
                dist = DiagonalFlowBase(
                    len(continuous_range),
                    device=self.device,
                    num_bins=self.spline_bins,
                    flow_steps=1,
                )
            
            else:
                raise NotImplementedError("Base distribution type not recognised")
            conditional_dists["continuous"] = dist

        return conditional_dists

    @classmethod
    def name(cls) -> str:
        return "deci"

    def get_adj_matrix_tensor(
        self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False
    ) -> torch.Tensor:
        if self.mode_adjacency == "learn":
            if most_likely_graph:
                assert samples == 1, "When passing most_likely_graph, only 1 sample can be returned."
                A_samples = [self.var_dist_A.get_adj_matrix(do_round=do_round)]
            else:
                A_samples = [self.var_dist_A.sample_A() for _ in range(samples)]
                if do_round:
                    A_samples = [A.round() for A in A_samples]
            adj = torch.stack(A_samples, dim=0)
        elif self.mode_adjacency == "upper":
            adj = (
                torch.triu(torch.ones(self.num_nodes, self.num_nodes), diagonal=1)
                .to(self.device)
                .expand(samples, -1, -1)
            )
        elif self.mode_adjacency == "lower":
            adj = (
                torch.tril(torch.ones(self.num_nodes, self.num_nodes), diagonal=-1)
                .to(self.device)
                .expand(samples, -1, -1)
            )
        else:
            raise NotImplementedError(f"Adjacency mode {self.mode_adjacency} not implemented")
        return self._apply_constraints(adj)

    def _apply_constraints(self, G: torch.Tensor) -> torch.Tensor:
        """
        Set all entries where self.neg_contraint_matrix=0 to 0, leave elements where self.neg_constraint_matrix=1 unchanged
        Set all entries where self.pos_contraint_matrix=1 to 1, leave elements where self.pos_constraint_matrix=0 unchanged
        """
        return 1.0 - (1.0 - G * self.neg_constraint_matrix) * (1.0 - self.pos_constraint_matrix)

    def get_adj_matrix(
        self,
        do_round: bool = True,
        samples: int = 100,
        most_likely_graph: bool = False,
        squeeze: bool = False,
    ) -> np.ndarray:
        """
        Returns the adjacency matrix (or several) as a numpy array.
        """
        adj_matrix = self.get_adj_matrix_tensor(do_round, samples, most_likely_graph)

        if squeeze and samples == 1:
            adj_matrix = adj_matrix.squeeze(0)
        # Here we have the cast to np.float64 because the original type
        # np.float32 has some issues with json, when saving causality results
        # to a file.
        return adj_matrix.detach().cpu().numpy().astype(np.float64)

    def get_weighted_adj_matrix(
        self,
        do_round: bool = True,
        samples: int = 100,
        most_likely_graph: bool = False,
        squeeze: bool = False,
    ) -> torch.Tensor:
        """
        Returns the weighted adjacency matrix (or several) as a numpy array.
        """
        A_samples = self.get_adj_matrix_tensor(do_round, samples, most_likely_graph)

        W_adjs = A_samples * self.ICGNN.get_weighted_adjacency().unsqueeze(0)

        if squeeze and samples == 1:
            W_adjs = W_adjs.squeeze(0)

        return W_adjs

    def sample_parameters(self, samples: int = 100):
        # For compatibility with `BayesianContextActionModel`
        return self.get_weighted_adj_matrix(do_round=False, samples=samples, most_likely_graph=False, squeeze=False)

    def dagness_factor(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the dag penalty for matrix A as trace(expm(A)) - dim.

        Args:
            A: Binary adjacency matrix, size (input_dim, input_dim).
        """
        return torch.trace(torch.matrix_exp(A)) - self.num_nodes

    def _log_prior_A(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the prior for adjacency matrix A, which consists of a term encouraging DAGness
        and another encouraging sparsity (see https://arxiv.org/pdf/2106.07635.pdf).

        Supports hierarchical (2-layer) sparsity penalty:
        - Layer 1 (L1): Element-wise sparsity penalty on individual edges
        - Layer 2 (L2): Group sparsity penalty based on l2_group_mode

        Args:
            A: Adjacency matrix of shape (input_dim, input_dim) or (lag+1, input_dim, input_dim).

        Returns:
            Log probability of A for prior distribution, a number.
        """
        eps = 1e-8

        # Layer 1: L1 sparsity (element-wise)
        l1_term = -self.lambda_sparse * A.abs().sum()

        # Layer 2: Group L2 sparsity
        if self.lambda_sparse_l2 > 0:
            if self.l2_group_mode == "column":
                # Incoming edges per node: sum over source dimension
                if A.dim() == 3:  # Temporal: [lag+1, n, n]
                    group_norms = torch.sqrt((A ** 2).sum(dim=-2) + eps)  # [lag+1, n]
                else:  # Static: [n, n]
                    group_norms = torch.sqrt((A ** 2).sum(dim=0) + eps)  # [n]

            elif self.l2_group_mode == "row":
                # Outgoing edges per node: sum over target dimension
                if A.dim() == 3:
                    group_norms = torch.sqrt((A ** 2).sum(dim=-1) + eps)  # [lag+1, n]
                else:
                    group_norms = torch.sqrt((A ** 2).sum(dim=1) + eps)  # [n]

            elif self.l2_group_mode == "lag":
                # Per-lag grouping (only for temporal)
                if A.dim() == 3:
                    group_norms = torch.sqrt((A ** 2).sum(dim=(-2, -1)) + eps)  # [lag+1]
                else:
                    # Fallback to Frobenius for non-temporal
                    group_norms = torch.sqrt((A ** 2).sum() + eps).unsqueeze(0)

            elif self.l2_group_mode == "frobenius":
                # Full matrix L2 norm
                group_norms = torch.sqrt((A ** 2).sum() + eps).unsqueeze(0)

            else:
                raise ValueError(f"Unknown l2_group_mode: {self.l2_group_mode}. "
                                f"Options: column, row, lag, frobenius")

            l2_group_term = -self.lambda_sparse_l2 * group_norms.sum()
        else:
            l2_group_term = 0.0

        return l1_term + l2_group_term

    def get_auglag_penalty(self, tracker_dag_penalty: List) -> float:
        """
        Computes DAG penalty for augmented Lagrangian update step as the average of dag factors of binary
        adjacencies sampled during this inner optimization step.
        """
        return torch.mean(torch.Tensor(tracker_dag_penalty)).item()

    def _log_prob(
        self,
        x: torch.Tensor,
        predict: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """
        Computes the log probability of the observed data given the predictions from the SEM.

        Args:
            x: Array of size (processed_dim_all) or (batch_size, processed_dim_all), works both ways (i.e. single sample
            or batched).
            predict: tensor of the same shape as x.
            intervention_mask (num_nodes): optional array containing indicators of variables that have been intervened upon.
            These will not be considered for log probability computation.

        Returns:
            Log probability of non intervened samples. A number if x has shape (input_dim), or an array of
            shape (batch_size) is x has shape (batch_size, input_dim).
        """
        

        # Continuous
        cts_bin_log_prob = torch.zeros_like(x)
        continuous_range = self.continuous_range
        if continuous_range:
            cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                x[..., continuous_range] - predict[..., continuous_range]
            )

        log_prob = cts_bin_log_prob.sum(-1)
        return log_prob

    def _icgnn_cts_mse(self, x: torch.Tensor, predict: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared error (SE) of the ICGNN on the continuous variables of the model.

        Args:
            x: Array of size (processed_dim_all) or (batch_size, processed_dim_all), works both ways (i.e. single sample
            or batched).
            predict: tensor of the same shape as x.

        Returns:
            SE of ICGNN predictions on continuous variables. A number if x has shape (input_dim), or an array of
            shape (batch_size) is X has shape (batch_size, input_dim).
        """
        continuous_range = self.continuous_range
        if isinstance(self.likelihoods["continuous"], GaussianBase):
            return (x[..., continuous_range] - predict[..., continuous_range]).pow(2).sum(-1)
        else:
            # Updates to `self.spline_mean_ewma` are made inside `optimize_inner_auglag`
            return (x[..., continuous_range] - predict[..., continuous_range] - self.spline_mean_ewma).pow(2).sum(-1)

    def _sample_base(self, Nsamples: int) -> torch.Tensor:
        """
        Draw samples from the base distribution.

        Args:
            Nsamples: Number of samples to draw

        Returns:
            torch.Tensor z of shape (batch_size, input_dim).
        """
        sample = torch.zeros((Nsamples, self.processed_dim_all), device=self.device)
        

        # Continuous
        continuous_range = self.continuous_range
        if continuous_range:
            sample[:, continuous_range] = self.likelihoods["continuous"].sample(Nsamples)

        return sample

    def log_prob(
        self,
        X: Union[torch.Tensor, np.ndarray],
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
    ):

        """
        Evaluate log-probability of observations X. Optionally this evaluation can be subject to an intervention on our causal model.
        Then, the log probability of non-intervened variables is computed.

        Args:
            X: torch.Tensor of shape (Nsamples, input_dim) containing the observations we want to evaluate
            Nsamples: int containing number of graph samples to draw.
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior instead of sampling graphs
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.
            conditioning_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that we condition on.
            conditioning_values: torch.Tensor of shape (input_dim) optional array containing values for variables that we condition on.
            Nsamples_per_graph: int containing number of samples to draw
            Ngraphs: Number of different graphs to sample for graph posterior marginalisation. If None, defaults to Nsamples
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample
            fixed_seed: The integer to seed the random number generator (unused)

        Returns:
            log_prob: torch.tensor  (Nsamples)
        """

        if fixed_seed is not None:
            raise NotImplementedError("Fixed seed not supported by DECI")

        # TODO: move these lines into .log_prob(), add dimmension check to sampling / ate code as well

        (X,) = to_tensors(X, device=self.device, dtype=torch.float)

        with torch.no_grad():

            log_prob_samples = []

            if most_likely_graph:
                Nsamples_per_graph = 1

            for _ in range(Nsamples_per_graph):

                A_sample = self.get_adj_matrix_tensor(do_round=False, samples=1, most_likely_graph=False).squeeze(0)
                W_adj = A_sample * self.ICGNN.get_weighted_adjacency()

                predict = self.ICGNN.predict(X, W_adj)
                # Note that the W input is for AR-DECI, DECI will not use W.
                W = A_sample if self.base_distribution_type == "conditional_spline" else None
                log_prob_samples.append(self._log_prob(X, predict, W=W))  # (B)

            log_prob = torch.logsumexp(torch.stack(log_prob_samples, dim=0), dim=0) - np.log(Nsamples_per_graph)
            return log_prob.detach().cpu().numpy().astype(np.float64)

    
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
        predict = self.ICGNN.predict(X, W_adj)
        log_p_A = self._log_prior_A(A_sample)  # A number
        penalty_dag = self.dagness_factor(A_sample)  # A number
        log_p_base = self._log_prob(
            X,
            predict,
            W=A_sample if self.base_distribution_type == "conditional_spline" else None,
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

    def compute_loss(
        self,
        step: int,
        x: torch.Tensor,
        num_samples: int,
        tracker: Dict,
        train_config_dict: Dict[str, Any],
        alpha: float = None,
        rho: float = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Computes the loss and updates trackers of different terms.

        Args:
            step: Inner auglag step.
            x: Input data of shape (batch_size, input_dim).
            mask_train_batch: Mask indicating which values are missing in the dataset of shape (batch_size, input_dim).
            input_mask: Mask indicating which additional values are aritificially masked.
            num_samples: Number of samples used to compute a stochastic estimate of the loss.
            tracker: Tracks terms in the loss during the inner auglag optimisation.
            train_config_dict: Contains training configuration.
            alpha: Parameter used to scale the penalty_dag term in the prior. Defaults to None.
            rho: Parameter used to scale the penalty_dag term in the prior. Defaults to None.
            adj_true: ground truth adj matrix for tracking causal discovery performance in inner loops
            compute_cd_fscore: whether to compute `cd_fscore` metric at each step. Warning: this may have negative
                side-effects on speeed and GPU utilization.

        Returns:
            Tuple containing the loss and the tracker.
        """
        _ = kwargs

        x_fill = x
        imputation_entropy = avg_reconstruction_err = torch.tensor(0.0, device=self.device)

        #  Compute remaining terms
        elbo_terms = self._ELBO_terms(x_fill)
        log_p_term = elbo_terms["log_p_base"].mean(dim=0)
        log_p_A_term = elbo_terms["log_p_A"] / num_samples
        log_q_A_term = elbo_terms["log_q_A"] / num_samples
        cts_mse = elbo_terms["cts_mse"].mean(dim=0)
        cts_medse, _ = torch.median(elbo_terms["cts_mse"], dim=0)

        penalty_dag_term = elbo_terms["penalty_dag"] * alpha / num_samples
        penalty_dag_term += elbo_terms["penalty_dag"] * elbo_terms["penalty_dag"] * rho / (2 * num_samples)

        if train_config_dict["anneal_entropy"] == "linear":
            ELBO = log_p_term + imputation_entropy + log_p_A_term - log_q_A_term / max(step - 5, 1) - penalty_dag_term
        elif train_config_dict["anneal_entropy"] == "noanneal":
            ELBO = log_p_term + imputation_entropy + log_p_A_term - log_q_A_term - penalty_dag_term
        loss = -ELBO + avg_reconstruction_err * train_config_dict["reconstruction_loss_factor"]


        tracker["loss"].append(loss.item())
        tracker["penalty_dag"].append(elbo_terms["penalty_dag"].item())
        tracker["penalty_dag_weighed"].append(penalty_dag_term.item())
        tracker["log_p_A_sparse"].append(log_p_A_term.item())
        tracker["log_p_x"].append(log_p_term.item())
        tracker["imputation_entropy"].append(imputation_entropy.item())
        tracker["log_q_A"].append(log_q_A_term.item())
        tracker["reconstruction_mse"].append(avg_reconstruction_err.item())
        tracker["cts_mse_icgnn"].append(cts_mse.item())
        tracker["cts_medse_icgnn"].append(cts_medse.item())
        return loss, tracker

    def print_tracker(self, inner_step: int, tracker: dict) -> None:
        """Prints formatted contents of loss terms that are being tracked."""
        tracker_copy = tracker.copy()

        loss = np.mean(tracker_copy.pop("loss")[-100:])
        log_p_x = np.mean(tracker_copy.pop("log_p_x")[-100:])
        penalty_dag = np.mean(tracker_copy.pop("penalty_dag")[-100:])
        log_p_A_sparse = np.mean(tracker_copy.pop("log_p_A_sparse")[-100:])
        log_q_A = np.mean(tracker_copy.pop("log_q_A")[-100:])
        h_filled = np.mean(tracker_copy.pop("imputation_entropy")[-100:])
        reconstr = np.mean(tracker_copy.pop("reconstruction_mse")[-100:])

        out = (
            f"Inner Step: {inner_step}, loss: {loss:.2f}, log p(x|A): {log_p_x:.2f}, dag: {penalty_dag:.8f}, "
            f"log p(A)_sp: {log_p_A_sparse:.2f}, log q(A): {log_q_A:.3f}, H filled: {h_filled:.3f}, rec: {reconstr:.3f}"
        )

        for k, v in tracker_copy.items():
            out += f", {k}: {np.mean(v[-100:]):.3g}"

        print(out)

    def run_train(
        self,
        dataloader,
        num_samples,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Runs training.
        """
        if train_config_dict is None:
            train_config_dict = {}

        # initialise logging machinery
       
        #writer = SummaryWriter(log_path, flush_secs=1)
        

        rho = train_config_dict["rho"]
        alpha = train_config_dict["alpha"]
        progress_rate = train_config_dict["progress_rate"]
        base_beta = train_config_dict["beta"] if "beta" in train_config_dict else 1.0
        base_lr = train_config_dict["learning_rate"]
        anneal_beta = train_config_dict["anneal_beta"] if "anneal_beta" in train_config_dict else None
        anneal_beta_max_steps = (
            train_config_dict["anneal_beta_max_steps"]
            if "anneal_beta_max_steps" in train_config_dict
            else int(train_config_dict["max_steps_auglag"] / 2)
        )

        # This allows the setting of the starting learning rate of each of the different submodules in the config, e.g. "likelihoods_learning_rate".
        parameter_list = [
            {
                "params": module.parameters(),
                "lr": train_config_dict.get(f"{name}_learning_rate", base_lr),
                "name": name,
            }
            for name, module in self.named_children()
        ]

        self.opt = torch.optim.Adam(parameter_list)

        # Outer optimization loop
        
        dag_penalty_prev = float("inf")
        num_below_tol = 0
        num_max_rho = 0
        num_not_done = 0
        for step in range(train_config_dict["max_steps_auglag"]):

            # Stopping if DAG conditions satisfied
            patience_dag_reached = train_config_dict.get("patience_dag_reached", 5)
            patience_max_rho = train_config_dict.get("patience_max_rho", 3)
            if num_below_tol >= patience_dag_reached:
                print(f"DAG penalty below tolerance for more than {patience_dag_reached} steps")
                break
            elif num_max_rho >= patience_max_rho:
                print(f"Above max rho for more than {patience_max_rho} steps")
                break

            if rho >= train_config_dict["safety_rho"]:
                num_max_rho += 1

            # Anneal beta.
            if anneal_beta == "linear":
                beta = base_beta * min((step + 1) / anneal_beta_max_steps, 1.0)
            elif anneal_beta == "reverse":
                beta = base_beta * max((anneal_beta_max_steps - step) / anneal_beta_max_steps, 0.2)
            else:
                beta = base_beta

            # Logging outer progress and adjacency matrix
            

            # Inner loop
            print(f"Auglag Step: {step}")

            print(f"Beta Value: {beta}")

            # Optimize adjacency for fixed rho and alpha
            outer_step_start_time = time.time()
            done_inner, tracker_loss_terms = self.optimize_inner_auglag(
                rho, alpha, beta, step, num_samples, dataloader, train_config_dict
            )
            outer_step_time = time.time() - outer_step_start_time
            dag_penalty = np.mean(tracker_loss_terms["penalty_dag"])

            # Print some stats about the DAG distribution
            #print(f"Dag penalty after inner: {dag_penalty:.10f}")
            #print("Time taken for this step", outer_step_time)

            # Update alpha (and possibly rho) if inner optimization done or if 2 consecutive not-done
            if done_inner or num_not_done == 1:
                num_not_done = 0
                if dag_penalty < train_config_dict["tol_dag"]:
                    num_below_tol += 1
                if report_progress_callback is not None:
                    report_progress_callback(self.model_id, step + 1, train_config_dict["max_steps_auglag"])

                with torch.no_grad():
                    if dag_penalty > dag_penalty_prev * progress_rate:
                        print(f"Updating rho, dag penalty prev: {dag_penalty_prev: .10f}")
                        rho *= 10.0
                    else:
                        print("Updating alpha.")
                        dag_penalty_prev = dag_penalty
                        alpha += rho * dag_penalty
                        if dag_penalty == 0.0:
                            alpha *= 5
                    if rho >= train_config_dict["safety_rho"]:
                        alpha *= 5
                    rho = min([rho, train_config_dict["safety_rho"]])
                    alpha = min([alpha, train_config_dict["safety_alpha"]])

            else:
                num_not_done += 1
                #print("Not done inner optimization.")

            # Print the current values of the auglag parameters rho, alpha
            if dag_penalty_prev is not None:
                print(f"Dag penalty: {dag_penalty:.15f}")
                print(f"Rho: {rho:.2f}, alpha: {alpha:.2f}")

    def optimize_inner_auglag(
        self,
        rho: float,
        alpha: float,
        beta: float,
        step: int,
        num_samples: int,
        dataloader,
        train_config_dict: Optional[Dict[str, Any]] = None,
        n_spline_sample: int = 32,
        spline_ewma_alpha: float = 0.05,
    ) -> Tuple[bool, Dict]:
        """
        Optimize for a given alpha and rho
        Args:
            rho: Parameter used to scale the penalty_dag term in the prior. Defaults to None.
            alpha: Parameter used to scale the penalty_dag term in the prior. Defaults to None.
            beta: KL term annealing coefficient
            step: Auglag step
            num_samples: Number of samples in the dataset
            dataloader: Dataset to generate a dataloader for.
            train_config_dict: Dictionary with training hyperparameters.
            adj_true: ground truth adj matrix
            bidirected_adj_true: ground truth bidirected adj matrix
            n_spline_samples: to estimate the non-zero mean of the spline noise distributions by sampling. These are
                aggregated using an exponentially weighted moving average.
            alpha: exponentially weighted moving average parameter for spline means.

        Returns:
            done_opt: boolean indicating if optimization is done.
            tracker_loss_terms: Dictionary for tracking loss terms
        """
        if train_config_dict is None:
            train_config_dict = {}

        def get_lr():
            for param_group in self.opt.param_groups:
                return param_group["lr"]

        def set_lr(factor):
            for param_group in self.opt.param_groups:
                param_group["lr"] = param_group["lr"] * factor

        def initialize_lr():
            base_lr = train_config_dict["learning_rate"]
            for param_group in self.opt.param_groups:
                name = param_group["name"]
                param_group["lr"] = train_config_dict.get(f"{name}_learning_rate", base_lr)

        lim_updates_down = 3
        num_updates_lr_down = 0
        auglag_inner_early_stopping_lag = train_config_dict.get("auglag_inner_early_stopping_lag", 1500)
        auglag_inner_reduce_lr_lag = train_config_dict.get("auglag_inner_reduce_lr_lag", 500)
        initialize_lr()
        print("LR:", get_lr())
        best_loss = np.nan
        last_updated = -1
        done_opt = False

        tracker_loss_terms: Dict = defaultdict(list)
        inner_step = 0

        while inner_step < train_config_dict["max_auglag_inner_epochs"]:  # and not done_steps:

            for x in dataloader:
                x = x.to(self.device)
                loss, tracker_loss_terms = self.compute_loss(
                    step,
                    x.float(),
                    num_samples,
                    tracker_loss_terms,
                    train_config_dict,
                    alpha,
                    rho,
                    beta=beta,
                )
                self.opt.zero_grad()
                loss.backward()

                # Gradient clipping for training stability
                if train_config_dict.get('gradient_clip_norm', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        train_config_dict['gradient_clip_norm']
                    )

                self.opt.step()

                # For MSE metric, update an estimate of the spline means
                if not isinstance(self.likelihoods["continuous"], (GaussianBase,TemporalSplineFlow, TemporalConditionalSplineFlow)):
                    error_dist_mean = self.likelihoods["continuous"].sample(n_spline_sample).mean(0)
                    self.spline_mean_ewma = (
                        spline_ewma_alpha * error_dist_mean + (1 - spline_ewma_alpha) * self.spline_mean_ewma
                    )

                inner_step += 1

                if int(inner_step) % 1000 == 0:
                    self.print_tracker(inner_step, tracker_loss_terms)
                if int(inner_step) % 500 == 0:
                    break
                elif inner_step >= train_config_dict["max_auglag_inner_epochs"]:
                    break

            # Save if loss improved
            if np.isnan(best_loss) or np.mean(tracker_loss_terms["loss"][-10:]) < best_loss:
                best_loss = np.mean(tracker_loss_terms["loss"][-10:])
                best_inner_step = inner_step
            # Check if has to reduce step size
            if (
                inner_step >= best_inner_step + auglag_inner_reduce_lr_lag
                and inner_step >= last_updated + auglag_inner_reduce_lr_lag
            ):
                last_updated = inner_step
                num_updates_lr_down += 1
                set_lr(0.1)
                print(f"Reducing lr to {get_lr():.5f}")
                if num_updates_lr_down >= 2:
                    done_opt = True
                if num_updates_lr_down >= lim_updates_down:
                    done_opt = True
                    print(f"Exiting at inner step {inner_step}.")
                    # done_steps = True
                    break
            if inner_step >= best_inner_step + auglag_inner_early_stopping_lag:
                done_opt = True
                print(f"Exiting at inner step {inner_step}.")
                # done_steps = True
                break
            if np.any(np.isnan(tracker_loss_terms["loss"])):
                print(tracker_loss_terms)
                print("Loss is nan, I'm done.", flush=True)
                # done_steps = True
                break
        self.print_tracker(inner_step, tracker_loss_terms)
        print(f"Best model found at innner step {best_inner_step}, with Loss {best_loss:.2f}")
        return done_opt, tracker_loss_terms

    