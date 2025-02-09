import itertools
import warnings

import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample
from statsmodels.tsa.vector_ar.var_model import VAR
from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC, LinearRegression
from scipy.stats import gamma
from statsmodels.nonparametric import bandwidths


def hsic_test_gamma(X, Y, bw_method="mdbs"):
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    if bw_method == "scott":
        width_x = bandwidths.bw_scott(X)
        width_y = bandwidths.bw_scott(Y)
    elif bw_method == "silverman":
        width_x = bandwidths.bw_silverman(X)
        width_y = bandwidths.bw_silverman(Y)
    else:
        width_x = get_kernel_width(X)
        width_y = get_kernel_width(Y)

    K, Kc = get_gram_matrix(X, width_x)
    L, Lc = get_gram_matrix(Y, width_y)

    n = X.shape[0]
    test_stat = hsic_teststat(Kc, Lc, n)

    var = (1 / 6 * Kc * Lc) ** 2
    var = 1 / n / (n - 1) * (np.sum(var) - np.trace(var))
    var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

    K[np.diag_indices(n)] = 0
    L[np.diag_indices(n)] = 0
    mu_X = 1 / n / (n - 1) * K.sum()
    mu_Y = 1 / n / (n - 1) * L.sum()
    mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

    alpha = mean**2 / var
    beta = var * n / mean
    p = gamma.sf(test_stat, alpha, scale=beta)

    return test_stat, p


class BootstrapMixin:
    """Mixin class for all LiNGAM algorithms that implement the method of bootstrapping."""

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : BootstrapResult
            Returns the result of bootstrapping.
        """
        # Check parameters
        X = check_array(X)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError("n_sampling must be an integer greater than 0.")
        else:
            raise ValueError("n_sampling must be an integer greater than 0.")

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        index = np.arange(X.shape[0])
        resampled_indices = []
        for i in range(n_sampling):
            resampled_X, resampled_index = resample(X, index)
            self.fit(resampled_X)
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c + 1 :]:
                    total_effects[i, to, from_] = calculate_total_effect(
                        self._adjacency_matrix, from_, to
                    )

            resampled_indices.append(resampled_index)

        return BootstrapResult(adjacency_matrices, total_effects, resampled_indices=resampled_indices)


class BootstrapResult(object):
    """The result of bootstrapping."""

    def __init__(self, adjacency_matrices, total_effects, resampled_indices=None):
        """Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        total_effects : array-like, shape (n_sampling)
            The total effects list by bootstrapping.
        resampled_indices :  array-like, shape (n_sampling, resample_size), optional (default=None)
            The list of original index of resampled samples.
        """
        self._adjacency_matrices = adjacency_matrices
        self._total_effects = total_effects
        self._resampled_indices = resampled_indices

    @property
    def adjacency_matrices_(self):
        """The adjacency matrix list by bootstrapping.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (n_sampling)
            The adjacency matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._adjacency_matrices

    @property
    def total_effects_(self):
        """The total effect list by bootstrapping.

        Returns
        -------
        total_effects_ : array-like, shape (n_sampling)
            The total effect list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._total_effects

    @property
    def resampled_indices_(self):
        """The list of original index of resampled samples.

        Returns
        -------
        resampled_indices_ : array-like, shape (n_sampling, resample_size)
            The list of original index of resampled samples,
            where ``n_sampling`` is the number of bootstrap sampling
            and ``resample_size`` is the size of each subsample set.
        """
        return self._resampled_indices

    def get_causal_direction_counts(
        self,
        n_directions=None,
        min_causal_effect=None,
        split_by_causal_effect_sign=False,
    ):
        """Get causal direction count as a result of bootstrapping.

        Parameters
        ----------
        n_directions : int, optional (default=None)
            If int, then The top ``n_directions`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects
            less than ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

        Returns
        -------
        causal_direction_counts : dict
            List of causal directions sorted by count in descending order.
            The dictionary has the following format::

            {'from': [n_directions], 'to': [n_directions], 'count': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if isinstance(n_directions, (numbers.Integral, np.integer)):
            if not 0 < n_directions:
                raise ValueError("n_directions must be an integer greater than 0")
        elif n_directions is None:
            pass
        else:
            raise ValueError("n_directions must be an integer greater than 0")

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # Count causal directions
        directions = []
        for am in np.nan_to_num(self._adjacency_matrices):
            direction = np.array(np.where(np.abs(am) > min_causal_effect))
            if split_by_causal_effect_sign:
                signs = (
                    np.array([np.sign(am[i][j]) for i, j in direction.T])
                    .astype("int64")
                    .T
                )
                direction = np.vstack([direction, signs])
            directions.append(direction.T)
        directions = np.concatenate(directions)

        if len(directions) == 0:
            cdc = {"from": [], "to": [], "count": []}
            if split_by_causal_effect_sign:
                cdc["sign"] = []
            return cdc

        directions, counts = np.unique(directions, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = (
            sort_order[:n_directions] if n_directions is not None else sort_order
        )
        counts = counts[sort_order]
        directions = directions[sort_order]

        cdc = {
            "from": directions[:, 1].tolist(),
            "to": directions[:, 0].tolist(),
            "count": counts.tolist(),
        }
        if split_by_causal_effect_sign:
            cdc["sign"] = directions[:, 2].tolist()

        return cdc

    def get_directed_acyclic_graph_counts(
        self, n_dags=None, min_causal_effect=None, split_by_causal_effect_sign=False
    ):
        """Get DAGs count as a result of bootstrapping.

        Parameters
        ----------
        n_dags : int, optional (default=None)
            If int, then The top ``n_dags`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than
            ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

        Returns
        -------
        directed_acyclic_graph_counts : dict
            List of directed acyclic graphs sorted by count in descending order.
            The dictionary has the following format::

            {'dag': [n_dags], 'count': [n_dags]}.

            where ``n_dags`` is the number of directed acyclic graphs.
        """
        # Check parameters
        if isinstance(n_dags, (numbers.Integral, np.integer)):
            if not 0 < n_dags:
                raise ValueError("n_dags must be an integer greater than 0")
        elif n_dags is None:
            pass
        else:
            raise ValueError("n_dags must be an integer greater than 0")

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # Count directed acyclic graphs
        dags = []
        for am in np.nan_to_num(self._adjacency_matrices):
            dag = np.abs(am) > min_causal_effect
            if split_by_causal_effect_sign:
                direction = np.array(np.where(dag))
                signs = np.zeros_like(dag).astype("int64")
                for i, j in direction.T:
                    signs[i][j] = np.sign(am[i][j]).astype("int64")
                dag = signs
            dags.append(dag)

        dags, counts = np.unique(dags, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_dags] if n_dags is not None else sort_order
        counts = counts[sort_order]
        dags = dags[sort_order]

        if split_by_causal_effect_sign:
            dags = [
                {
                    "from": np.where(dag)[1].tolist(),
                    "to": np.where(dag)[0].tolist(),
                    "sign": [dag[i][j] for i, j in np.array(np.where(dag)).T],
                }
                for dag in dags
            ]
        else:
            dags = [
                {"from": np.where(dag)[1].tolist(), "to": np.where(dag)[0].tolist()}
                for dag in dags
            ]

        return {"dag": dags, "count": counts.tolist()}

    def get_probabilities(self, min_causal_effect=None):
        """Get bootstrap probability.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than
            ``min_causal_effect`` are excluded.

        Returns
        -------
        probabilities : array-like
            List of bootstrap probability matrix.
        """
        # check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        adjacency_matrices = np.nan_to_num(self._adjacency_matrices)
        shape = adjacency_matrices[0].shape
        bp = np.zeros(shape)
        for B in adjacency_matrices:
            bp += np.where(np.abs(B) > min_causal_effect, 1, 0)
        bp = bp / len(adjacency_matrices)

        if int(shape[1] / shape[0]) == 1:
            return bp
        else:
            return np.hsplit(bp, int(shape[1] / shape[0]))

    def get_total_causal_effects(self, min_causal_effect=None):
        """Get total effects list.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than
            ``min_causal_effect`` are excluded.

        Returns
        -------
        total_causal_effects : dict
            List of bootstrap total causal effect sorted by probability in descending order.
            The dictionary has the following format::

            {'from': [n_directions], 'to': [n_directions], 'effect': [n_directions], 'probability': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # Calculate probability
        probs = np.sum(
            np.where(np.abs(self._total_effects) > min_causal_effect, 1, 0),
            axis=0,
            keepdims=True,
        )[0]
        probs = probs / len(self._total_effects)

        # Causal directions
        dirs = np.array(np.where(np.abs(probs) > 0))
        probs = probs[dirs[0], dirs[1]]

        # Calculate median effect without zero
        effects = np.zeros(dirs.shape[1])
        for i, (to, from_) in enumerate(dirs.T):
            idx = np.where(np.abs(self._total_effects[:, to, from_]) > 0)
            effects[i] = np.median(self._total_effects[:, to, from_][idx])

        # Sort by probability
        order = np.argsort(-probs)
        dirs = dirs.T[order]
        effects = effects[order]
        probs = probs[order]

        ce = {
            "from": dirs[:, 1].tolist(),
            "to": dirs[:, 0].tolist(),
            "effect": effects.tolist(),
            "probability": probs.tolist(),
        }

        return ce

    def get_paths(self, from_index, to_index, min_causal_effect=None):
        """Get all paths from the start variable to the end variable and their bootstrap probabilities.

        Parameters
        ----------
        from_index : int
            Index of the variable at the start of the path.
        to_index : int
            Index of the variable at the end of the path.
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        paths : dict
            List of path and bootstrap probability.
            The dictionary has the following format::

            {'path': [n_paths], 'effect': [n_paths], 'probability': [n_paths]}

            where ``n_paths`` is the number of paths.
        """
        # check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # Find all paths from from_index to to_index
        paths_list = []
        effects_list = []
        for am in self._adjacency_matrices:
            paths, effects = find_all_paths(am, from_index, to_index, min_causal_effect)
            # Convert path to string to make them easier to handle.
            paths_list.extend(["_".join(map(str, p)) for p in paths])
            effects_list.extend(effects)

        paths_list = np.array(paths_list)
        effects_list = np.array(effects_list)

        # Count paths
        paths_str, counts = np.unique(paths_list, axis=0, return_counts=True)

        # Sort by count
        order = np.argsort(-counts)
        probs = counts[order] / len(self._adjacency_matrices)
        paths_str = paths_str[order]

        # Calculate median of causal effect for each path
        effects = [
            np.median(effects_list[np.where(paths_list == p)]) for p in paths_str
        ]

        result = {
            "path": [[int(i) for i in p.split("_")] for p in paths_str],
            "effect": effects,
            "probability": probs.tolist(),
        }
        return result


def predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    # Standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Pruning with Adaptive Lasso
    lr = LinearRegression()
    lr.fit(X_std[:, predictors], X_std[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X_std[:, predictors] * weight, X_std[:, target])
    pruned_idx = np.abs(reg.coef_ * weight) > 0.0

    # Calculate coefficients of the original scale
    coef = np.zeros(reg.coef_.shape)
    if pruned_idx.sum() > 0:
        lr = LinearRegression()
        pred = np.array(predictors)
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_

    return coef


def find_all_paths(dag, from_index, to_index, min_causal_effect=0.0):
    """Find all paths from point to point in DAG.

    Parameters
    ----------
    dag : array-like, shape (n_features, n_features)
        The adjacency matrix to fine all paths, where n_features is the number of features.
    from_index : int
        Index of the variable at the start of the path.
    to_index : int
        Index of the variable at the end of the path.
    min_causal_effect : float, optional (default=0.0)
        Threshold for detecting causal direction.
        Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

    Returns
    -------
    paths : array-like, shape (n_paths)
        List of found path, where n_paths is the number of paths.
    effects : array-like, shape (n_paths)
        List of causal effect, where n_paths is the number of paths.
    """
    # Extract all edges
    edges = np.array(np.where(np.abs(np.nan_to_num(dag)) > min_causal_effect)).T

    # Aggregate edges by start point
    to_indices = []
    for i in range(dag.shape[0]):
        adj_list = edges[edges[:, 1] == i][:, 0].tolist()
        if len(adj_list) != 0:
            to_indices.append(adj_list)
        else:
            to_indices.append([])

    # DFS
    paths = []
    stack = [from_index]
    stack_to_indice = [to_indices[from_index]]
    while stack:
        if len(stack) > dag.shape[0]:
            raise ValueError(
                "Unable to find the path because a cyclic graph has been specified."
            )

        cur_index = stack[-1]
        to_indice = stack_to_indice[-1]

        if cur_index == to_index:
            paths.append(stack.copy())
            stack.pop()
            stack_to_indice.pop()
        else:
            if len(to_indice) > 0:
                next_index = to_indice.pop(0)
                stack.append(next_index)
                stack_to_indice.append(to_indices[next_index].copy())
            else:
                stack.pop()
                stack_to_indice.pop()

    # Calculate the causal effect for each path
    effects = []
    for p in paths:
        coefs = [dag[p[i + 1], p[i]] for i in range(len(p) - 1)]
        effects.append(np.cumprod(coefs)[-1])

    return paths, effects

def calculate_total_effect(adjacency_matrix, from_index, to_index, is_continuous=None):
    """Calculate total effect.

    Parameters
    ----------
    adjacency_matrix : array_like
        The adjacency matrix.
    from_index : int
        The index of the cause variable.
    to_index : int
        The index of the effect variable.
    is_continuous : list
        The list of boolean. is_continuous indicates whether each variable
        is continuous or discrete.

    Returns
    -------
    total_effect : float
    """

    # check inputs
    adjacency_matrix = check_array(
        adjacency_matrix,
        ensure_min_samples=2,
        ensure_min_features=2,
        force_all_finite="allow-nan",
    )
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(
            "adjacency_matrix must be an square matrix.", adjacency_matrix.shape
        )

    from_index = check_scalar(
        from_index,
        "from_index",
        (numbers.Integral, np.integer),
        min_val=0,
        max_val=len(adjacency_matrix) - 1,
    )

    to_index = check_scalar(
        to_index,
        "to_index",
        (numbers.Integral, np.integer),
        min_val=0,
        max_val=len(adjacency_matrix) - 1,
    )

    if from_index == to_index:
        raise ValueError("from_index and to_index mustn't be the same.")

    if is_continuous is None:
        is_continuous = [True for _ in range(len(adjacency_matrix))]
    else:
        is_continuous = check_array(
            is_continuous, ensure_2d=False, ensure_min_samples=len(adjacency_matrix)
        )

    # find all paths
    path_list, effects = find_all_paths(adjacency_matrix, from_index, to_index)

    # check all nodes on the path are continuous
    for path in path_list:
        for node in path[1:]:
            if not is_continuous[node]:
                raise ValueError("Variables on the path must be continuous variables.")

    total_effect = sum(effects)

    return total_effect


class _BaseLiNGAM(metaclass=ABCMeta):
    """Base class for all LiNGAM algorithms."""

    def __init__(self, random_state=None):
        """Construct a _BaseLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            random_state is the seed used by the random number generator.
        """
        self._random_state = random_state
        self._causal_order = None
        self._adjacency_matrix = None

    @abstractmethod
    def fit(self, X):
        """Subclasses should implement this method!
        Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def estimate_total_effect(self, X, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check parameters
        X = check_array(X)

        # Check from/to causal order
        from_order = self._causal_order.index(from_index)
        to_order = self._causal_order.index(to_index)
        if from_order > to_order:
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the causal order of the destination variable (to_index={to_index}) "
                f"is earlier than the source variable (from_index={from_index})."
            )

        # from_index + parents indices
        parents = np.where(np.abs(self._adjacency_matrix[from_index]) > 0)[0]
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, to_index])

        return lr.coef_[0]

    def get_error_independence_p_values(self, X):
        """Calculate the p-value matrix of independence between error variables.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        # Check parameters
        X = check_array(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        E = X - np.dot(self._adjacency_matrix, X.T).T
        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            _, p_value = hsic_test_gamma(np.reshape(E[:, i], [n_samples, 1]), np.reshape(E[:, j], [n_samples, 1]))
            p_values[i, j] = p_value
            p_values[j, i] = p_value

        return p_values

    def _estimate_adjacency_matrix(self, X, prior_knowledge=None):
        """Estimate adjacency matrix by causal order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
            Prior knowledge matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if prior_knowledge is not None:
            pk = prior_knowledge.copy()
            np.fill_diagonal(pk, 0)

        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(self._causal_order)):
            target = self._causal_order[i]
            predictors = self._causal_order[:i]

            # Exclude variables specified in no_path with prior knowledge
            if prior_knowledge is not None:
                predictors = [p for p in predictors if pk[target, p] != 0]

            # target is exogenous variables if predictors are empty
            if len(predictors) == 0:
                continue

            B[target, predictors] = predict_adaptive_lasso(X, predictors, target)

        self._adjacency_matrix = B
        return self

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where
            n_features is the number of features.
        """
        return self._causal_order

    @property
    def adjacency_matrix_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_features, n_features)
            The adjacency matrix B of fitted model, where
            n_features is the number of features.
        """
        return self._adjacency_matrix



class DirectLiNGAM(_BaseLiNGAM):
    """Implementation of DirectLiNGAM Algorithm [1]_ [2]_

    References
    ----------
    .. [1] S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen.
       DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model.
       Journal of Machine Learning Research, 12(Apr): 1225--1248, 2011.
    .. [2] A. Hyvärinen and S. M. Smith. Pairwise likelihood ratios for estimation of non-Gaussian structural eauation models.
       Journal of Machine Learning Research 14:111-152, 2013.
    """

    def __init__(
        self,
        random_state=None,
        prior_knowledge=None,
        apply_prior_knowledge_softly=False,
        measure="pwling",
    ):
        """Construct a DirectLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        apply_prior_knowledge_softly : boolean, optional (default=False)
            If True, apply prior knowledge softly.
        measure : {'pwling', 'kernel', 'pwling_v2'}, optional (default='pwling')
        """
        super().__init__(random_state)
        self._Aknw = prior_knowledge
        self._apply_prior_knowledge_softly = apply_prior_knowledge_softly
        self._measure = measure

        if self._Aknw is not None:
            self._Aknw = check_array(self._Aknw)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X)
        n_features = X.shape[1]

        if self._Aknw is not None:
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError("The shape of prior knowledge must be (n_features, n_features)")
            else:
                if not self._apply_prior_knowledge_softly:
                    self._partial_orders = self._extract_partial_orders(self._Aknw)

        U = np.arange(n_features)
        K = []
        X_ = np.copy(X)
        if self._measure == "pwling_v2" or self._measure == "pwling_v3":
            X_ = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            start_time = time.time()
            X_entropies = self._precompute_entropies(X_)
            X_residuals_entropies = self._precompute_residual_entropies(X_)
            end_time = time.time()
            execution_time = end_time - start_time
            print("precomputation Execution time:", execution_time, "seconds")

        print("Measure method:", self._measure)
        

        start_time = time.time()
        if self._measure != "pwling_v3":
            for _ in range(n_features):
                if self._measure == "pwling_v2":
                    m = self._search_causal_order_v2(U, X_entropies, X_residuals_entropies)
                else:
                    m = self._search_causal_order(X_, U)
                    
                if self._measure != "pwling_v2":
                    for i in U:
                        if i != m:
                            X_[:, i] = self._residual(X_[:, i], X_[:, m])
                K.append(m)
                U = U[U != m]
                if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                    self._partial_orders = self._partial_orders[self._partial_orders[:, 0] != m]
        else:
            M_list = []
            for i in U:
                M = 0
                for j in U:
                    if i != j:
                        M += np.min([0, self._diff_mutual_info_v2(X_entropies, X_residuals_entropies, i, j)]) ** 2
                M_list.append(M)
                K = list(np.argsort(M_list))

        end_time = time.time()
        execution_time = end_time - start_time
        print("search causal order Execution time:", execution_time, "seconds")

        self._causal_order = K

        start_time = time.time()
        self._estimate_adjacency_matrix(X, prior_knowledge=self._Aknw)
        end_time = time.time()
        execution_time = end_time - start_time
        print("estimate adjacency matrix Execution time:", execution_time, "seconds")
        return self

    def _extract_partial_orders(self, pk):
        path_pairs = np.array(np.where(pk == 1)).transpose()
        no_path_pairs = np.array(np.where(pk == 0)).transpose()

        check_pairs = np.concatenate([path_pairs, path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            if len(pairs[counts > 1]) > 0:
                raise ValueError(f"The prior knowledge contains inconsistencies at the following indices: {pairs[counts>1].tolist()}")

        check_pairs = np.concatenate([no_path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            check_pairs = np.concatenate([no_path_pairs, pairs[counts > 1]])
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            no_path_pairs = pairs[counts < 2]

        check_pairs = np.concatenate([path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) == 0:
            return check_pairs

        pairs = np.unique(check_pairs, axis=0)
        return pairs[:, [1, 0]]

    def _precompute_entropies(self, X_std):
        n_features = X_std.shape[1]
        X_entropy = np.zeros(n_features) 
        
        u = X_std
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        
        log_cosh = np.log(np.cosh(u))
        exp_term = u * np.exp(-(u ** 2) / 2)
        
        X_entropy = (1 + np.log(2 * np.pi)) / 2 - \
                    k1 * (np.mean(log_cosh, axis=0) - gamma) ** 2 - \
                    k2 * (np.mean(exp_term, axis=0)) ** 2
        
        return X_entropy

    def _precompute_residual_entropies(self, X_):
        n_features = X_.shape[1]
        n_samples = X_.shape[0]
        X_entropy = np.zeros((n_features, n_features))
        
        variances = np.var(X_, axis=0)
        covariances = np.cov(X_.T, bias=True)
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    beta = covariances[i, j] / variances[j]
                    residual = X_[:, i] - beta * X_[:, j]
                    residual_std = residual / np.std(residual)
                    
                    u = residual_std
                    k1 = 79.047
                    k2 = 7.4129
                    gamma = 0.37457
                    
                    log_cosh_mean = np.mean(np.log(np.cosh(u)))
                    exp_term_mean = np.mean(u * np.exp(-(u ** 2) / 2))
                    
                    X_entropy[i, j] = (1 + np.log(2 * np.pi)) / 2 - \
                                     k1 * (log_cosh_mean - gamma) ** 2 - \
                                     k2 * exp_term_mean ** 2
        
        return X_entropy

    def _residual(self, xi, xj):
        return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

    def _entropy(self, u):
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) - (self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i)))
    
    def _diff_mutual_info_v2(self, X_entropies, X_residual_entropies, i, j):
        return (X_entropies[j] + X_residual_entropies[i][j]) - (X_entropies[i] + X_residual_entropies[j][i])

    def _search_candidate(self, U):
        if self._Aknw is None:
            return U, []

        if not self._apply_prior_knowledge_softly:
            if len(self._partial_orders) != 0:
                Uc = [i for i in U if i not in self._partial_orders[:, 1]]
                return Uc, []
            else:
                return U, []

        Uc = []
        for j in U:
            index = U[U != j]
            if self._Aknw[j][index].sum() == 0:
                Uc.append(j)

        if len(Uc) == 0:
            U_end = []
            for j in U:
                index = U[U != j]
                if np.nansum(self._Aknw[j][index]) > 0:
                    U_end.append(j)

            for i in U:
                index = U[U != i]
                if self._Aknw[index, i].sum() == 0:
                    U_end.append(i)
            Uc = [i for i in U if i not in set(U_end)]

        Vj = []
        for i in U:
            if i in Uc:
                continue
            if self._Aknw[i][Uc].sum() == 0:
                Vj.append(i)
        return Uc, Vj

    def _search_causal_order(self, X, U):
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        M_list = []
        for i in Uc:
            M = 0
            for j in U:
                if i != j:
                    xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                    xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                    ri_j = xi_std if i in Vj and j in Uc else self._residual(xi_std, xj_std)
                    rj_i = xj_std if j in Vj and i in Uc else self._residual(xj_std, xi_std)
                    M += np.min([0, self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return Uc[np.argmax(M_list)]
    
    def _search_causal_order_v2(self, U, X_entropies, X_residuals_entropies):
        M_list = []
        for i in U:
            M = 0
            for j in U:
                if i != j:
                    M += np.min([0, self._diff_mutual_info_v2(X_entropies, X_residuals_entropies, i, j)]) ** 2
            M_list.append(-1.0 * M)
        return U[np.argmax(M_list)]



class VARLiNGAM:
    def __init__(
        self,
        lags=1,
        criterion="bic",
        prune=True,
        ar_coefs=None,
        lingam_model=None,
        random_state=None,
        lingam_measure='pwling',
    ):
        """Construct a VARLiNGAM model.

        Parameters
        ----------
        lags : int, optional (default=1)
            Number of lags.
        criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
            Criterion to decide the best lags within ``lags``.
            Searching the best lags is disabled if ``criterion`` is ``None``.
        prune : boolean, optional (default=True)
            Whether to prune the adjacency matrix of lags.
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR model. Estimating AR model is skipped if specified ``ar_coefs``.
            Shape must be (``lags``, n_features, n_features).
        lingam_model : lingam object inherits 'lingam._BaseLiNGAM', optional (default=None)
            LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._lags = lags
        self._criterion = criterion
        self._prune = prune
        self._ar_coefs = (
            check_array(ar_coefs, allow_nd=True) if ar_coefs is not None else None
        )
        self._lingam_model = lingam_model
        self._random_state = random_state
        self._lingam_measure = lingam_measure

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        returns
        -------
        self : object
            Returns the instance itself.
        """
        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)

        lingam_model = self._lingam_model
        if lingam_model is None:
            lingam_model = DirectLiNGAM(measure=self._lingam_measure)
        elif not isinstance(lingam_model, _BaseLiNGAM):
            raise ValueError("lingam_model must be a subclass of _BaseLiNGAM")

        M_taus = self._ar_coefs

        if M_taus is None:
            start_time = time.time()
            M_taus, lags, residuals = self._estimate_var_coefs(X)
            end_time = time.time()
            execution_time = end_time - start_time
            print("Estimate VAR coefficients time:", execution_time, "seconds")
        else:
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)

        model = lingam_model
        model.fit(residuals)

        B_taus = self._calc_b(X, model.adjacency_matrix_, M_taus)

        if self._prune:
            B_taus = self._pruning(X, B_taus, model.causal_order_)

        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals

        self._causal_order = model.causal_order_
        self._adjacency_matrices = B_taus

        return self

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : TimeseriesBootstrapResult
            Returns the result of bootstrapping.
        """
        X = check_array(X)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # store initial settings
        ar_coefs = self._ar_coefs
        lags = self._lags

        criterion = self._criterion
        self._criterion = None

        self.fit(X)

        fitted_ar_coefs = self._ar_coefs

        total_effects = np.zeros(
            [n_sampling, n_features, n_features * (1 + self._lags)]
        )

        adjacency_matrices = []
        for i in range(n_sampling):
            sampled_residuals = resample(self._residuals, n_samples=n_samples)

            resampled_X = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                if j < lags:
                    resampled_X[j, :] = sampled_residuals[j]
                    continue

                ar = np.zeros((1, n_features))
                for t, M in enumerate(fitted_ar_coefs):
                    ar += np.dot(M, resampled_X[j - t - 1, :].T).T

                resampled_X[j, :] = ar + sampled_residuals[j]

            # restore initial settings
            self._ar_coefs = ar_coefs
            self._lags = lags

            self.fit(resampled_X)
            am = np.concatenate([*self._adjacency_matrices], axis=1)
            adjacency_matrices.append(am)

            # total effects
            for c, to in enumerate(reversed(self._causal_order)):
                # time t
                for from_ in self._causal_order[: n_features - (c + 1)]:
                    total_effects[i, to, from_] = self.estimate_total_effect2(
                        n_features, from_, to
                    )

                # time t-tau
                for lag in range(self._lags):
                    for from_ in range(n_features):
                        total_effects[
                            i, to, from_ + n_features * (lag + 1)
                        ] = self.estimate_total_effect2(n_features, from_, to, lag + 1)

        self._criterion = criterion

        return VARBootstrapResult(adjacency_matrices, total_effects)

    def estimate_total_effect(self, X, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        X = check_array(X)
        n_features = X.shape[1]

        # Check from/to causal order
        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_index={to_index}) "
                    f"is earlier than the source variable (from_index={from_index})."
                )

        # X + lagged X
        X_joined = np.zeros((X.shape[0], X.shape[1] * (1 + self._lags + from_lag)))
        for p in range(1 + self._lags + from_lag):
            pos = n_features * p
            X_joined[:, pos : pos + n_features] = np.roll(X[:, 0:n_features], p, axis=0)

        # from_index + parents indices
        am = np.concatenate([*self._adjacency_matrices], axis=1)
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        from_index = (
            from_index if from_lag == 0 else from_index + (n_features * from_lag)
        )
        parents = parents if from_lag == 0 else parents + (n_features * from_lag)
        predictors = [from_index]
        predictors.extend(parents)

        # estimate total effect
        lr = LinearRegression()
        lr.fit(X_joined[:, predictors], X_joined[:, to_index])

        return lr.coef_[0]

    def estimate_total_effect2(self, n_features, from_index, to_index, from_lag=0):
        """Estimate total effect using causal model.

        Parameters
        ----------
        n_features :
            The number of features.
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check from/to causal order
        if from_lag == 0:
            from_order = self._causal_order.index(from_index)
            to_order = self._causal_order.index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_index={to_index}) "
                    f"is earlier than the source variable (from_index={from_index})."
                )

        # from_index + parents indices
        am = np.concatenate([*self._adjacency_matrices], axis=1)
        am = np.pad(am, [(0, am.shape[1] - am.shape[0]), (0, 0)])
        from_index = (
            from_index if from_lag == 0 else from_index + (n_features * from_lag)
        )

        effect = calculate_total_effect(am, from_index, to_index)

        return effect

    def get_error_independence_p_values(self):
        """Calculate the p-value matrix of independence between error variables.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        nn = self.residuals_
        B0 = self._adjacency_matrices[0]
        E = np.dot(np.eye(B0.shape[0]) - B0, nn.T).T
        n_samples = E.shape[0]
        n_features = E.shape[1]

        p_values = np.zeros([n_features, n_features])
        for i, j in itertools.combinations(range(n_features), 2):
            _, p_value = hsic_test_gamma(
                np.reshape(E[:, i], [n_samples, 1]), np.reshape(E[:, j], [n_samples, 1])
            )
            p_values[i, j] = p_value
            p_values[j, i] = p_value

        return p_values

    def _estimate_var_coefs(self, X):
        """Estimate coefficients of VAR"""
        # XXX: VAR.fit() is not searching lags correctly
        if self._criterion not in ["aic", "fpe", "hqic", "bic"]:
            var = VAR(X)
            result = var.fit(maxlags=self._lags, trend="n")
        else:
            min_value = float("Inf")
            result = None

            for lag in range(1, self._lags + 1):
                var = VAR(X)
                fitted = var.fit(maxlags=lag, ic=None, trend="n")

                value = getattr(fitted, self._criterion)
                if value < min_value:
                    min_value = value
                    result = fitted

        return result.coefs, result.k_ar, result.resid

    def _calc_residuals(self, X, M_taus, lags):
        """Calculate residulas"""
        X = X.T
        n_features = X.shape[0]
        n_samples = X.shape[1]

        residuals = np.zeros((n_features, n_samples))
        for t in range(n_samples):
            if t - lags < 0:
                continue

            estimated = np.zeros((X.shape[0], 1))
            for tau in range(1, lags + 1):
                estimated += np.dot(M_taus[tau - 1], X[:, t - tau].reshape((-1, 1)))

            residuals[:, t] = X[:, t] - estimated.reshape((-1,))

        residuals = residuals[:, lags:].T

        return residuals

    def _calc_b(self, X, B0, M_taus):
        """Calculate B"""
        n_features = X.shape[1]

        B_taus = np.array([B0])

        for M in M_taus:
            B_t = np.dot((np.eye(n_features) - B0), M)
            B_taus = np.append(B_taus, [B_t], axis=0)

        return B_taus

    def _pruning(self, X, B_taus, causal_order):
        """Prune edges"""
        n_features = X.shape[1]

        stacked = [np.flip(X, axis=0)]
        for i in range(self._lags):
            stacked.append(np.roll(stacked[-1], -1, axis=0))
        blocks = np.array(list(zip(*stacked)))[: -self._lags]

        for i in range(n_features):
            causal_order_no = causal_order.index(i)
            ancestor_indexes = causal_order[:causal_order_no]

            obj = np.zeros((len(blocks)))
            exp = np.zeros((len(blocks), causal_order_no + n_features * self._lags))
            for j, block in enumerate(blocks):
                obj[j] = block[0][i]
                exp[j:] = np.concatenate(
                    [block[0][ancestor_indexes].flatten(), block[1:][:].flatten()],
                    axis=0,
                )

            # adaptive lasso
            predictors = [i for i in range(exp.shape[1])]
            target = len(predictors)
            X_con = np.concatenate([exp, obj.reshape(-1, 1)], axis=1)
            coef = predict_adaptive_lasso(X_con, predictors, target)

            B_taus[0][i, ancestor_indexes] = coef[:causal_order_no]
            for j in range(len(B_taus[1:])):
                B_taus[j + 1][i, :] = coef[
                    causal_order_no + n_features * j :
                    causal_order_no + n_features * j + n_features
                ]

        return B_taus

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where
            n_features is the number of features.
        """
        return self._causal_order

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (lags, n_features, n_features)
            The adjacency matrix of fitted model, where
            n_features is the number of features.
        """
        return self._adjacency_matrices

    @property
    def residuals_(self):
        """Residuals of regression.

        Returns
        -------
        residuals_ : array-like, shape (n_samples)
            Residuals of regression, where n_samples is the number of samples.
        """
        return self._residuals
