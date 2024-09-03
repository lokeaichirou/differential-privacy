
from collections import namedtuple, defaultdict
from typing import Optional, Dict, Any
import operator
import math
import numpy as np

from sklearn.tree._tree import NODE_DTYPE, Tree
from sklearn.tree._classes import BaseDecisionTree, DecisionTreeRegressor

from accountant import BudgetAccountant
from bounds import DiffprivlibMixin
from mechanisms.exponential import Exponential
from mechanisms.laplace import Laplace

import logger as logging

logging.SetUpLogger(__name__)
logger = logging.GetLogger(__name__)


class DifferentiallyPrivateDecisionTreeRegressor(DecisionTreeRegressor, DiffprivlibMixin):
    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(BaseDecisionTree,
                                                                          "max_depth",
                                                                          "random_state")

    def __init__(self,
                 tree_index,
                 epsilon,
                 max_depth,
                 max_leaves,
                 min_samples_split,
                 leaf_clipping,
                 learning_rate,
                 l2_threshold,
                 l2_lambda,
                 delta_g,
                 delta_v,
                 cat_idx      = None,
                 random_state = None,
                 accountant   = None,
                 **unused_args):
        try:
            super().__init__(criterion                = None,
                             splitter                 = None,
                             max_depth                = max_depth,
                             min_samples_split        = None,
                             min_samples_leaf         = None,
                             min_weight_fraction_leaf = None,
                             max_features             = None,
                             random_state             = random_state,
                             max_leaf_nodes           = None,
                             min_impurity_decrease    = None,
                             min_impurity_split       = None)

        except TypeError:
            super().__init__(criterion                = None,
                             splitter                 = None,
                             max_depth                = max_depth,
                             min_samples_split        = None,
                             min_samples_leaf         = None,
                             min_weight_fraction_leaf = None,
                             max_features             = None,
                             random_state             = random_state,
                             max_leaf_nodes           = None,
                             min_impurity_decrease    = None)

        self.tree_index        = tree_index
        self.epsilon           = epsilon
        self.max_depth         = max_depth
        self.max_leaves        = max_leaves
        self.min_samples_split = min_samples_split
        self.leaf_clipping     = leaf_clipping
        self.learning_rate     = learning_rate
        self.l2_threshold      = l2_threshold
        self.l2_lambda         = l2_lambda
        self.delta_g           = delta_g
        self.delta_v           = delta_v
        self.cat_idx           = cat_idx

        self.accountant = BudgetAccountant.load_default(accountant)
        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight = None, check_input = False, X_idx_sorted="deprecated"):

        self.accountant.check(self.epsilon, 0)

        self.n_features_in_ = X.shape[1]

        y_dimension = len(y.shape)
        if y_dimension == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[1]

        fitting_dp_tree = FittingDifferentiallyPrivateTree(K                 = 1,
                                                           tree_index        = self.tree_index,
                                                           epsilon           = self.epsilon,
                                                           max_depth         = self.max_depth,
                                                           max_leaves        = self.max_leaves,
                                                           min_samples_split = self.min_samples_split,
                                                           leaf_clipping     = self.leaf_clipping,
                                                           learning_rate     = self.learning_rate,
                                                           l2_threshold      = self.l2_threshold,
                                                           l2_lambda         = self.l2_lambda,
                                                           delta_g           = self.delta_g,
                                                           delta_v           = self.delta_v,
                                                           cat_idx           = self.cat_idx)

        fitting_dp_tree.fit(X, y)

        # Load params from _FittingTree into sklearn.Tree
        # -------------------------------------------------------------------------------------------------------------
        d    = fitting_dp_tree.__getstate__()
        tree = Tree(self.n_features_in_,
                    np.array([1], dtype=np.intp),
                    self.n_outputs_)
        tree.__setstate__(d)
        self.tree_ = tree

        # Take privacy cost into account
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.spend(self.epsilon, 0)

        return self


class FittingDifferentiallyPrivateTree(DiffprivlibMixin):

    _TREE_LEAF      = -1
    _TREE_UNDEFINED = -2
    StackNode       = namedtuple("StackNode",
                                 ["parent",
                                  "is_left",
                                  "depth",
                                  "X",
                                  "y",
                                  "gradients"]
                                 )

    def __init__(self,
                 K,
                 tree_index,
                 epsilon,
                 max_depth,
                 max_leaves,
                 min_samples_split,
                 leaf_clipping,
                 learning_rate,
                 l2_threshold,
                 l2_lambda,
                 delta_g,
                 delta_v,
                 cat_idx):
        """
        Array-based representation of a binary decision tree, trained with differential privacy.

        This tree mimics the architecture of the corresponding Tree from sklearn.tree.tree_, but without many methods
        given in Tree. The purpose of _FittingTree is to fit the parameters of the model, and have those parameters
        passed to Tree (using _FittingTree.__getstate__() and Tree.__setstate__()), to be used for prediction.

        :param K:                   The number of classes.
        :param tree_index           The index of the tree.
        :param epsilon              The privacy parameter.
        :param max_depth            The maximum depth of the tree.
        :param max_leaves           The maximum number of leaves in the tree.
        :param min_samples_split    The minimum number of samples required to split an internal node.
        :param leaf_clipping        The clipping threshold for leaf node values.
        :param learning_rate        The learning rate for updating the tree.
        :param l2_threshold         The threshold for applying L2 regularization.
        :param l2_lambda            The regularization parameter for L2 regularization.
        :param delta_g              The sensitivity of the gradients.
        :param delta_v              The sensitivity of the values.
        :param cat_idx              The indices of categorical features.

        """

        self.node_count        = 0
        self.nodes             = []
        self.K                 = K
        self.tree_index        = tree_index
        self.epsilon           = epsilon
        self.max_depth         = max_depth
        self.max_leaves        = max_leaves
        self.min_samples_split = min_samples_split
        self.leaf_clipping     = leaf_clipping
        self.learning_rate     = learning_rate
        self.l2_threshold      = l2_threshold
        self.l2_lambda         = l2_lambda
        self.delta_g           = delta_g
        self.delta_v           = delta_v
        self.cat_idx           = cat_idx

        # This handles attribute comparison depending on the attribute's nature
        self.feature_to_op = defaultdict(lambda: (operator.lt, operator.ge))  # type: Dict[int, Any]
        if self.cat_idx:
            for feature_index in self.cat_idx:
                self.feature_to_op[feature_index] = (operator.eq, operator.ne)

        self.privacy_budget_for_internal_node = np.around(np.divide(self.epsilon/2, self.max_depth), decimals=7)
        self.privacy_budget_for_leaf_node     = self.epsilon/2

    def __getstate__(self):
        """Get state of _FittingTree to feed into __setstate__ of sklearn.Tree"""
        d = {"max_depth":  self.max_depth,
             "node_count": self.node_count,
             "nodes":      np.array([tuple(node) for node in self.nodes], dtype=NODE_DTYPE),
             "values":     self.values_}
        return d

    def compute_gain(self,
                    index: int,
                    value: Any,
                    X: np.array,
                    gradients: np.array):
        """

        Compute the gain for a given split.

        See https://dl.acm.org/doi/pdf/10.1145/2939672.2939785

        Args:
            index (int):                            The index for the feature to split on.
            value (Any):                            The feature's value to split on.
            X (np.array):                           The dataset.
            gradients (np.array):                   The gradients for the dataset instances.

        Returns:
          float: The gain for the split.
        """
        lhs_op, rhs_op = self.feature_to_op[index]
        lhs            = np.where(lhs_op(X[:, index], value))[0]
        rhs            = np.where(rhs_op(X[:, index], value))[0]

        if len(lhs) == 0 or len(rhs) == 0:
            # Can't split on this feature as all instances share the same value
            return -1

        lhs_grad, rhs_grad = gradients[lhs], gradients[rhs]
        lhs_gain           = np.square(np.sum(lhs_grad)) / (len(lhs) + self.l2_lambda)  # type: float
        rhs_gain           = np.square(np.sum(rhs_grad)) / (len(rhs) + self.l2_lambda)  # type: float
        total_gain         = lhs_gain + rhs_gain

        if total_gain >= 0.:
            result = (total_gain, lhs, rhs)
        else:
            result = (0., lhs, rhs)

        return result

    def find_best_split(self,
                      parent,
                      is_left,
                      X: np.array,
                      y: np.array,
                      gradients: np.array,
                      current_depth: Optional[int] = None):
        """
        Find best split of data using the exponential mechanism.

        :param X (np.array):                           The dataset.
        :param gradients (np.array):                   The gradients for the dataset instances.
        :param current_depth (int): Optional.          The current depth of the tree. If specified, the privacy budget
                                                       decays with the depth growing.

        :return
        """

        # privacy budget for internal node
        logger.debug('Using {0:f} budget for internal leaf nodes.'.format(self.privacy_budget_for_internal_node))

        possible_split_infos, gains = [], []
        # Iterate over features
        for feature_index in range(X.shape[1]):

            # Iterate over unique values for this feature
            for idx, value in enumerate(np.unique(X[:, feature_index])):

                # Find gain for that split
                result = self.compute_gain(index             = feature_index,
                                           value             = value,
                                           X                 = X,
                                           gradients         = gradients)

                if result == -1:
                     # Feature's value cannot be chosen, skipping
                     continue
                else:
                    gain, lhs, rhs = result[0], result[1], result[2]

                gains.append(gain)
                possible_split_info = {"parent":  parent,
                                       "is_left": is_left,
                                       "depth":   current_depth+1,
                                       "index":   feature_index,
                                       "value":   value,
                                       "gain":    gain,
                                       "lhs":     lhs,
                                       "rhs":     rhs}

                possible_split_infos.append(possible_split_info)

        if len(gains) < 1:
            return None
        else:
            mech = Exponential(epsilon     = self.privacy_budget_for_internal_node,
                               sensitivity = self.delta_g,
                               utility     = gains,
                               candidates  = possible_split_infos)

            selected_split_info = mech.randomise()

            lhs, rhs     = selected_split_info["lhs"], selected_split_info["rhs"]
            lhs_X, lhs_y = X[lhs], y[lhs]
            rhs_X, rhs_y = X[rhs], y[rhs]

            lhs_gradients = gradients[lhs]
            rhs_gradients = gradients[rhs]

            return {"parent":          selected_split_info["parent"],
                    "is_left":         selected_split_info["is_left"],
                    "depth":           selected_split_info["depth"] + 1,
                    "index":           selected_split_info["index"],
                    "threshold_value": selected_split_info["value"],
                    "gain":            selected_split_info["gain"],
                    "lhs_X":           lhs_X,
                    "rhs_X":           rhs_X,
                    "lhs_y":           lhs_y,
                    "rhs_y":           rhs_y,
                    "lhs_gradients":   lhs_gradients,
                    "rhs_gradients":   rhs_gradients}

    def update_parent_node_info(self, parent, is_left, node_id):
        """
        Update the parent node information.

        :param parent: The parent node.
        :param is_left: Whether the current node is the left child of the parent.
        :param node_id: The ID of the current node.
        :return: None
        """

        if is_left:
            self.nodes[parent].left_child = node_id
        else:
            self.nodes[parent].right_child = node_id

    def create_intermediate_parent_nodes(self, node_id, feature_index, threshold_value):
        """
        Function being used in build function to create intermediate parent nodes.

        1. Create the decision node

        2. Create the intermediate_parent_node by instantiating the _Node class and give input parameters node_id,
        splitting feature and threshold value for splitting feature

        3. Append the newly created _Node object to the self.nodes list

        4. count up the self.node_count node counter

        :param node_id:         The ID of the current node.
        :param feature_index:   The index of the feature used for splitting.
        :param threshold_value: The threshold value for splitting the feature.
        :return: None
        """

        # Create the intermediate_parent_node by instantiating the _Node class and append the newly created _Node object
        # to the self.nodes list
        # --------------------------------------------------------------------------------------------------------------
        self.nodes.append(_Node(node_id, feature_index, threshold_value))

        # count up the self.node_count node counter
        # --------------------------------------------------------------------------------------------------------------
        self.node_count += 1

    def compute_predictions(self,
                           gradients: np.ndarray,
                           y: np.ndarray) -> float:
        """
        Computes the predictions of a leaf.

        Used in the `DifferentiallyPrivateTree` as well as in `SplitNode` for the 3-tree version.

        Ref:
            Friedman 01. "Greedy function approximation: A gradient boosting machine."
              (https://projecteuclid.org/euclid.aos/1013203451)

        Args:
            gradients (np.ndarray): The positive gradients y˜ for the dataset instances.
            y (np.ndarray): The dataset labels y.
            l2_lambda (float): Regularization parameter for l2 loss function.

        Returns:
            Prediction γ of a leaf
        """

        if len(gradients) == 0:
            prediction = 0.  # type: ignore

        if self.K > 1:
            # sum of neg. gradients divided by sum of 2nd derivatives aka one Newton-Raphson step for details ref.
            # (eq 33+34) in Friedman 01.
            prediction = -1 * np.sum(gradients) * (self.K - 1) / self.K
            denom      = np.sum((y + gradients) * (1 - y - gradients))
            prediction = 0 if abs(denom) < 1e-150 else prediction / (denom + self.l2_lambda)
        else:
            # Equation 4 in the paper
            prediction = (-1 * np.sum(gradients) / (len(gradients) + self.l2_lambda))

        return prediction

    def geometric_leaf_clipping(self, prediction):

        threshold = self.l2_threshold * math.pow((1 - self.learning_rate), self.tree_index)

        if np.abs(prediction) > threshold:
            if prediction > 0:
                prediction = threshold
            else:
                prediction = -1 * threshold

        return prediction

    def create_leaf_nodes(self, node_id, gradients, y):
        """
        Create leaf nodes

        :param node_id:
        :param gradients:
        :param y:

        :return: Nothing
        """

        # Create the leaf_node by instantiating the _Node class
        node             = _Node(node_id, self._TREE_UNDEFINED, self._TREE_UNDEFINED)
        node.left_child  = self._TREE_LEAF
        node.right_child = self._TREE_LEAF

        # Append the newly created _Node object to the self.nodes list
        # --------------------------------------------------------------------------------------------------------------
        self.nodes.append(node)

        # count up the self.node_count node counter
        # --------------------------------------------------------------------------------------------------------------
        self.node_count += 1

        prediction = self.compute_predictions(gradients = gradients,
                                              y         = y)

        if self.leaf_clipping:
            prediction = self.geometric_leaf_clipping(prediction=prediction)

        mech = Laplace(epsilon     = self.privacy_budget_for_leaf_node,
                       sensitivity = self.delta_v)

        noisy_prediction = mech.randomise(prediction)

        return noisy_prediction

    def fit(self, X, y):
        """["parent", "is_left", "depth", "X", "y", "gradients"]"""

        leaf_node_ids_predictions = []

        stack = [self.StackNode(parent          = self._TREE_UNDEFINED,
                                is_left         = False,
                                depth           = 0,
                                X               = X,
                                y               = y,
                                gradients       = -y)]

        # Use iterative method to build the binary decision tree in pre-order traversing manner
        while stack:

            # Step 1: Pop up the topmost node in stack, generate node_id for this node, take its info for further
            # processing
            # ---------------------------------------------------------------------------------------------------------

            parent, is_left, depth, X, y, gradients = stack.pop()
            node_id                                 = self.node_count

            # Step 2: Create leaf nodes or parental node
            # ---------------------------------------------------------------------------------------------------------
            if (len(X) < self.min_samples_split) or (depth == self.max_depth):
                # Create leaf nodes
                noisy_prediction = self.create_leaf_nodes(node_id, gradients, y)
                leaf_node_ids_predictions.append((node_id, noisy_prediction))

            else:
                best_split_info = self.find_best_split(parent        = parent,
                                                       is_left       = is_left,
                                                       X             = X,
                                                       y             = y,
                                                       gradients     = gradients,
                                                       current_depth = depth)
                if best_split_info is not None:

                    # Create parental node and find best split
                    self.create_intermediate_parent_nodes(node_id         = node_id,
                                                          feature_index   = best_split_info["index"],
                                                          threshold_value = best_split_info["threshold_value"])

                    stack.append(self.StackNode(parent        = node_id,
                                                is_left       = False,
                                                depth         = depth+1,
                                                X             = best_split_info["rhs_X"],
                                                y             = best_split_info["rhs_y"],
                                                gradients     = best_split_info["rhs_gradients"]))

                    stack.append(self.StackNode(parent        = node_id,
                                                is_left       = True,
                                                depth         = depth + 1,
                                                X             = best_split_info["lhs_X"],
                                                y             = best_split_info["lhs_y"],
                                                gradients     = best_split_info["lhs_gradients"]))

                else:
                    # Create leaf nodes
                    noisy_prediction = self.create_leaf_nodes(node_id, gradients, y)
                    leaf_node_ids_predictions.append((node_id, noisy_prediction))

            # Step 3: Update parent node's children node information
            # ---------------------------------------------------------------------------------------------------------

            # if parent = self._TREE_UNDEFINED, then the node is the most root node, the most root node does not have
            # parent node, so no need to update root node's parent node information; Otherwise,
            if parent != self._TREE_UNDEFINED:
                self.update_parent_node_info(parent  = parent,
                                             is_left = is_left,
                                             node_id = node_id)

        # Step 4 Assign prediction values to corresponding leaf nodes
        # ---------------------------------------------------------------------------------------------------------
        values = np.zeros(shape=(self.node_count, 1, self.K))

        for leaf_node_id, prediction in leaf_node_ids_predictions:
            values[leaf_node_id, 0, 0] = prediction

        self.values_ = values

        return self


class _Node:
    """
    Base storage structure for the nodes in a _FittingTree object.
    """

    def __init__(self, node_id, feature, threshold):
        self.node_id = node_id
        self.feature = feature
        self.threshold = threshold
        self.left_child = -1
        self.right_child = -1

    def __iter__(self):
        """
        Defines parameters needed to populate NODE_DTYPE for Tree.__setstate__ using tuple(_Node).
        """
        yield self.left_child
        yield self.right_child
        yield self.feature
        yield self.threshold
        yield 0.0  # Impurity
        yield 0  # n_node_samples
        yield 0.0  # weighted_n_node_samples
