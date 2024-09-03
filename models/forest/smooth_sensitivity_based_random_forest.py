"""
Random Forest Classifier with Differential Privacy
"""
from collections import namedtuple
import warnings
from joblib import Parallel, delayed
import math

import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import Tree, DOUBLE, DTYPE, NODE_DTYPE
from sklearn.ensemble._forest import RandomForestClassifier as skRandomForestClassifier, _parallel_build_trees
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier

from accountant import BudgetAccountant
from utils import PrivacyLeakWarning, check_random_state
from mechanisms.exponential import Exponential
from bounds import DiffprivlibMixin

MAX_INT = np.iinfo(np.int32).max


class SmoothSensitivityBasedRandomForestClassifier(skRandomForestClassifier, DiffprivlibMixin):

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(skRandomForestClassifier,
                                                                          "n_estimators",
                                                                          "n_jobs",
                                                                          "verbose",
                                                                          "random_state",
                                                                          "warm_start")

    def __init__(self,
                 n_estimators = 10,
                 *,
                 epsilon      = 1.0,
                 bounds       = None,
                 classes      = None,
                 n_jobs       = 1,
                 verbose      = 0,
                 accountant   = None,
                 random_state = None,
                 max_depth    = 5,
                 warm_start   = False,
                 shuffle      = False,
                 **unused_args):
        """
        Random Forest Classifier with differential privacy.
        This class implements Differentially Private Random Decision Forests using [1].
        ${epsilon}-Differential privacy is achieved by constructing decision trees via random splitting criterion and
        applying the Exponential Mechanism to determine a noisy label.

        :param n_estimators: int type, default: 10.     The number of trees in the forest.
        :param epsilon:      float type, default: 1.0.  Privacy parameter ${epsilon}.

        :param bounds:       tuple type, optional.      Bounds of the data, provided as a tuple of the form (min, max).
                                                        min and max can either be scalars, covering the min/max of the
                                                        entire data, or vectors with one entry per feature.  If not
                                                        provided, the bounds are computed on the data when ``.fit()`` is
                                                        first called, resulting in a :class:`.PrivacyLeakWarning`.

        :param classes:      array-like of shape (n_classes,). Array of unique class values to be trained on.

        :param n_jobs:       int type, default: 1.      Number of CPU cores used when parallelising over classes. ``-1``
                                                        means using all processors.

        :param verbose:      int type, default: 0.      Set to any positive number for verbosity.
        :param accountant:   BudgetAccountant, optional Accountant to keep track of privacy budget.

        :param random_state: int or RandomState, optional. Controls both the randomness of the shuffling of the samples
                                                           used when building trees (if shuffle=True) and training of
                                                           the differentially-private class: DecisionTreeClassifier to
                                                           construct the forest. To obtain a deterministic behaviour
                                                           during randomisation, random_state has to be fixed to an
                                                           integer.

        :param max_depth:    int type, default: 5.      The maximum depth of the tree.  The depth translates to an
                                                        exponential increase in memory usage.

        :param warm_start:   bool type, default=False.  When set to ``True``, reuse the solution of the previous call to
                                                        fit and add more estimators to the ensemble, otherwise, just fit
                                                        a whole new forest.

        :param shuffle:      bool type, default=False.  When set to ``True``, shuffles the datapoints to be trained on
                                                        trees at random.

        :param unused_args:  base_estimator_ : DecisionTreeClassifier. The child estimator template used to create the
                                                                       collection of fitted sub-estimators.

        :param n_outputs_ : int type.                   The number of outputs when ``fit`` is performed.

        unused_args
        ----------
        :param estimators_ : list of DecisionTreeClassifier. The collection of fitted sub-estimators.
        :param classes_ :    ndarray of shape (n_classes,) or a list of such arrays. The classes labels.
        :param n_classes_ :  int type or list. The number of classes.
        :param n_features_in_ : int type.               Number of features seen during :term:`fit`.
        :param feature_names_in_ : ndarray of shape (`n_features_in_`,).  Names of features seen during :term:`fit`.
                                                                          Defined only when `X` has feature names that
                                                                          are all strings.

        References
        ----------
        [1] Sam Fletcher, Md Zahidul Islam. "Differentially Private Random Decision Forests using Smooth Sensitivity"
        https://arxiv.org/abs/1606.03572
        """
        super().__init__(n_estimators = n_estimators,
                         criterion    = None,
                         bootstrap    = False,
                         n_jobs       = n_jobs,
                         random_state = random_state,
                         verbose      = verbose,
                         warm_start   = warm_start)

        self.epsilon             = epsilon
        self.bounds              = bounds
        self.classes             = classes
        self.max_depth           = max_depth
        self.shuffle             = shuffle
        self.accountant          = BudgetAccountant.load_default(accountant)

        if hasattr(self, "estimator"):
            self.estimator      = DecisionTreeClassifier()
        else:
            self.base_estimator = DecisionTreeClassifier()

        self.estimator_params = ("max_depth", "epsilon", "bounds", "classes")
        self._warn_unused_args(unused_args)

    def check_bounds(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))

        self.bounds = self._check_bounds(self.bounds, shape=X.shape[1])
        X           = self._clip_to_bounds(X, self.bounds)
        y           = np.atleast_1d(y)

        return X, y

    def process_output(self, y):
        """
        Convert continuous output y (if y is continuous) into categorical data like 0, 1, 2, ... , num_unique_val_y -1

        :param y:
        :return:
        """
        # Process output y
        # -------------------------------------------------------------------------------------------------------------
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was expected. Please change the shape of y to "
                          "(n_samples,), for example using ravel().", DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if self.classes is None:
            warnings.warn("Classes have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify the prediction classes for model.", PrivacyLeakWarning)
            self.classes = np.unique(y)

        # Return a contiguously flattened sorted classes array that contain all unique values in the output y.
        # self.classes_   = np.ravel(self.classes.sort())
        self.classes_   = np.ravel(self.classes)
        self.n_classes_ = len(self.classes_)

        # Even if original y is categorical data, it can be converted into integer typed class index, and convert into
        # float type
        y = np.searchsorted(self.classes_, y)
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        return y

    def check_parameters(self):
        """
        Check parameters

        :return:
        """
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        if not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(f"n_estimators={self.n_estimators} must be larger or equal to len(estimators_)="
                             f"{len(self.estimators_)} when warm_start==True")
        if n_more_estimators == 0:
            warnings.warn("Warm-start fitting without increasing n_estimators does not fit new trees.")
            return self

        if len(self.estimators_) > 0:
            # We draw from the random state to get the random state we would have got if we hadn't used a warm_start.
            random_state.randint(MAX_INT, size=len(self.estimators_))

        return n_more_estimators, random_state

    def compute_tree_depth(self, num_numerical_features):
        """
        Compute tree depth based on the idea proposed in 4.3 in [1], the proposed depth is within [m/2, m], the exact
        value of depth is computed by minimizing the expected number X of continuous features m not tested, which is
        m * [(m - 1)/m]^d, and [(m - 1)/m]^d is the probability of not being chosen for certain continuous feature.

        In sklean, it only supports the continuous features and does not support categorical features at all.
        So, num_numerical_features = total_num_features


        :param num_numers: number of continuous typed attributes
        :param num_categs: number of discrete typed attributes
        :return:
        """
        ''' Designed using balls-in-bins probability. See the paper for details. '''
        m              = float(num_numerical_features)
        depth          = 0
        # the number of unique attributes not selected so far
        expected_empty = m

        # repeat until we have less than half the attributes being empty
        while expected_empty > m / 2.:
            expected_empty = m * ((m-1.)/m)**depth
            depth         += 1

        final_depth = math.floor(depth)  # the above was only for half the numerical attributes now add half the
                                         # categorical attributes
        ''' WARNING: The depth translates to an exponential increase in memory usage. Do not go above ~15 unless you
         have 50+ GB of RAM. '''

        return min(15, final_depth)

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        :param X:             array-like of shape (n_samples, n_features). The training input samples. Internally,
                              its dtype will be converted to dtype=np.float32.

        :param y:             array-like of shape (n_samples,). The target values (class labels in classification,
                                                                real numbers in regression).
        :param sample_weight:
        :return:              self : object. Fitted estimator.
        """

        num_samples, num_features = X.shape

        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        # Validate or convert input data
        X, y = self._validate_data(X, y, multi_output=False, dtype=DTYPE)

        # Check bounds
        # -------------------------------------------------------------------------------------------------------------
        X, y = self.check_bounds(X, y)

        # Process output y
        # -------------------------------------------------------------------------------------------------------------
        y = self.process_output(y)

        # Check parameters
        # -------------------------------------------------------------------------------------------------------------
        n_more_estimators, random_state = self.check_parameters()

        # compute_tree_depth
        self.max_depth = self.compute_tree_depth(num_features)

        # Defining estimator
        # -------------------------------------------------------------------------------------------------------------
        #if hasattr(self, "estimator"):
        #    self.estimator      = DecisionTreeClassifier(max_depth = self.max_depth,
        #                                                 epsilon   = self.epsilon)
        #else:
        #    self.base_estimator = DecisionTreeClassifier(max_depth = self.max_depth,
        #                                                 epsilon   = self.epsilon)

        # Defining random forest model which is composed of trees model and train model based on data
        trees = [self._make_estimator(append=False, random_state=random_state) for _ in range(n_more_estimators)]

        # Split samples between trees as evenly as possible (randomly if shuffle==True), Shuffle samples first if set
        # shuffle = True, tree_idxs contains assigned tree's index to each sample
        tree_idxs = random_state.permutation(num_samples) if self.shuffle else np.arange(num_samples)
        tree_idxs = (tree_idxs // (num_samples / n_more_estimators)).astype(int)

        try:
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")\
                    (delayed(_parallel_build_trees)(tree          = t,
                                                    bootstrap     = False,
                                                    X             = X[tree_idxs == i],
                                                    y             = y[tree_idxs == i],
                                                    sample_weight = None,
                                                    tree_idx      = i,
                                                    n_trees       = len(trees),
                                                    verbose       = self.verbose,
                                                    )
                     for i, t in enumerate(trees)
                    )

        except TypeError:
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")\
                    (delayed(_parallel_build_trees)(tree          = t,
                                                    forest        = self,
                                                    X             = X[tree_idxs == i],
                                                    y             = y[tree_idxs == i],
                                                    sample_weight = None,
                                                    tree_idx      = i,
                                                    n_trees       = len(trees),
                                                    verbose       = self.verbose,)
                    for i, t in enumerate(trees)
                    )

        # Collect newly grown trees and take privacy cost into account
        # -------------------------------------------------------------------------------------------------------------
        self.estimators_.extend(trees)
        self.accountant.spend(self.epsilon, 0)

        return self


class DecisionTreeClassifier(skDecisionTreeClassifier, DiffprivlibMixin):

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(skDecisionTreeClassifier,
                                                                          "max_depth",
                                                                          "random_state")

    def __init__(self,
                 max_depth    = None,
                 *,
                 epsilon      = 1.0,
                 bounds       = None,
                 classes      = None,
                 random_state = None,
                 accountant   = None,
                 **unused_args):

        """
        Decision Tree Classifier with differential privacy.
        This class implements the base differentially private decision tree classifier for the Random Forest classifier
        algorithm. Not meant to be used separately.

        :param max_depth:    int type, default: 5.     The maximum depth of the tree.
        :param epsilon:      float type, default: 1.0. Privacy parameter ${epsilon}.
        :param bounds:       tuple type, optional.     Bounds of the data, provided as a tuple of the form (min, max).
                                                       min and max can either be scalars, covering the min/max of the
                                                       entire data, or vectors with one entry per feature.  If not
                                                       provided, the bounds are computed on the data when .fit() is
                                                       first called, resulting in a class: PrivacyLeakWarning.

        :param classes:      array-like of shape (n_classes,), optional. Array of class labels. If not provided, the
                                                                         classes will be read from the data when .fit()
                                                                         is first called, resulting in a class
                                                                         PrivacyLeakWarning.

        :param random_state: int or RandomState, optional. Controls the randomness of the estimator.  At each split,
                                                           the feature to split on is chosen randomly, as is the
                                                           threshold at which to split.  The classification label at
                                                           each leaf is then randomised, subject to differential privacy
                                                           constraints. To obtain a deterministic behaviour during
                                                           randomisation, random_state has to be fixed to an integer.
        :param accountant:   BudgetAccountant, optional.   Accountant to keep track of privacy budget.

        unused_args
        ----------
        :param n_features_in_: int type.                   The number of features when fit is performed.
        :param n_classes_:     int type.                   The number of classes.
        :param classes_:      array of shape (n_classes, ) The class labels.
        """

        # Todo: Remove when scikit-learn v1.0 is a min requirement
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

        self.epsilon    = epsilon
        self.bounds     = bounds
        self.classes    = classes
        self.accountant = BudgetAccountant.load_default(accountant)
        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Build a differentially-private decision tree classifier from the training set (X, y).

        :param X:             array-like of shape (n_samples, n_features). The training input samples. Internally, it
                                                                           will be converted to ``dtype=np.float32``.
        :param y:             array-like of shape (n_samples,).            The target values (class labels) as integers
                                                                           or strings.
        :param sample_weight:
        :param check_input:   bool type, default=True.                     Allow to bypass several input checking.
                                                                           Don't use this parameter unless you know what
                                                                           you do.
        :return:              DecisionTreeClassifier.                      Fitted estimator.
        """

        random_state = check_random_state(self.random_state)
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        if check_input:
            X, y = self._validate_data(X, y, multi_output=False)

        self.n_outputs_ = 1

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))

        self.bounds = self._check_bounds(self.bounds, shape=X.shape[1])
        X           = self._clip_to_bounds(X, self.bounds)

        if self.classes is None:
            warnings.warn("Classes have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify the prediction classes for model.", PrivacyLeakWarning)
            self.classes = np.unique(y)

        self.classes_       = np.ravel(self.classes)
        self.n_classes_     = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        # Build and fit the _FittingTree
        # -------------------------------------------------------------------------------------------------------------
        fitting_tree = _FittingTree(self.max_depth,
                                    self.n_features_in_,
                                    self.classes_,
                                    self.epsilon,
                                    self.bounds,
                                    random_state)
        fitting_tree.build()
        fitting_tree.fit(X, y)

        # Load params from _FittingTree into sklearn.Tree
        # -------------------------------------------------------------------------------------------------------------
        d    = fitting_tree.__getstate__()
        tree = Tree(self.n_features_in_,
                    np.array([self.n_classes_]),
                    self.n_outputs_)
        tree.__setstate__(d)
        self.tree_ = tree

        # Take privacy cost into account
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.spend(self.epsilon, 0)

        return self

    @property
    def n_features_(self):
        return self.n_features_in_

    def _more_tags(self):
        return {}


class _FittingTree(DiffprivlibMixin):

    _TREE_LEAF = -1
    _TREE_UNDEFINED = -2
    StackNode = namedtuple("StackNode", ["parent", "is_left", "depth", "bounds"])

    def __init__(self,
                 max_depth,
                 n_features,
                 classes,
                 epsilon,
                 bounds,
                 random_state):
        """
        Array-based representation of a binary decision tree, trained with differential privacy.

        This tree mimics the architecture of the corresponding Tree from sklearn.tree.tree_, but without many methods
        given in Tree. The purpose of _FittingTree is to fit the parameters of the model, and have those parameters
        passed to Tree (using _FittingTree.__getstate__() and Tree.__setstate__()), to be used for prediction.

        :param max_depth:    int type; The maximum depth of the tree.
        :param n_features:   int type; The number of features of the training dataset.
        :param classes:      array-like of shape (n_classes,); The classes of the training dataset.
        :param epsilon:      float type; Privacy parameter ${epsilon}
        :param bounds:       tuple type; Bounds of the data, provided as a tuple of the form (min, max).
                                        min and max can either be scalars, covering the min/max of the entire data.
        :param random_state: Controls the randomness of the building and training process: the feature to split at each
                             node, the threshold to split at and the randomisation of the label at each leaf.
        """

        self.node_count   = 0
        self.nodes        = []
        self.max_depth    = max_depth
        self.n_features   = n_features
        self.classes      = classes
        self.epsilon      = epsilon
        self.bounds       = bounds
        self.random_state = random_state

    def __getstate__(self):
        """Get state of _FittingTree to feed into __setstate__ of sklearn.Tree"""
        d = {"max_depth":  self.max_depth,
             "node_count": self.node_count,
             "nodes":      np.array([tuple(node) for node in self.nodes], dtype=NODE_DTYPE),
             "values":     self.values_}
        return d

    def build(self):
        """
        Build the decision tree using random feature selection and random thresholding.

        We use iterative method to build the binary decision tree in pre-order traversing manner.
        Fill up the self.nodes list with instantiated _Node object in pre-order traversing manner. self.nodes is one of
        parameters to be used in Sklearn Tree. _Node class has a. node_id  b. feature  c. threshold d. left_child
        e. right_child attributes. The node_id, feature and threshold attributes are given in instantiation step,
        left_child and right_child are to be filled up when iterating to its children node.

        We use the stack (which we use list's pop up function to achieve) to save the children nodes to be created
        (by instantiating _Node). Once it is being iterated, it is created. Before it is created by instantiating _Node
        class, update its parent node's children node information, and then we create it by instantiating _Node class.

        :return:
        """
        stack = [self.StackNode(parent  = self._TREE_UNDEFINED,
                                is_left = False,
                                depth   = 0,
                                bounds  = self.bounds)]

        # Use iterative method to build the binary decision tree in pre-order traversing manner
        while stack:

            # Step 1: Pop up the topmost node in stack, generate node_id for this node, take its bounds info for further
            # processing
            # ---------------------------------------------------------------------------------------------------------

            parent, is_left, depth, bounds = stack.pop()
            node_id                        = self.node_count
            lower_bounds, upper_bounds     = self._check_bounds(bounds, shape=self.n_features)

            # Step 2: Update parent node's children node information
            # ---------------------------------------------------------------------------------------------------------

            # if parent = self._TREE_UNDEFINED, then the node is the most root node, the most root node does not have
            # parent node, so no need to update root node's parent node information;
            # Otherwise,
            if parent != self._TREE_UNDEFINED:
                self.update_parent_node_info(parent,
                                             is_left,
                                             node_id)

            # Step 3: Create leaf nodes and intermediate decision nodes
            # ---------------------------------------------------------------------------------------------------------

            # Leaf nodes case: if the node reaches the max depth, then it's the leaf node, no need to pick feature and
            # threshold and no need to create children nodes for leaf nodes, use continus command to skip the step 3
            if depth >= self.max_depth:
                self.create_leaf_nodes(node_id)
                continue
            else:
                # Intermediate decision nodes case: Otherwise, the node is intermediate parent node or let's say the
                # decision node in between the most root node and leaf nodes, so we need to randomly pick feature and
                # threshold for each of them when creating them
                right_bounds_lower, left_bounds_upper = self.create_intermediate_parent_nodes(node_id,
                                                                                              lower_bounds,
                                                                                              upper_bounds)

                # Step 4 (Optional, only for Intermediate decision nodes case): Add left and right children nodes
                # ---------------------------------------------------------------------------------------------------------
                stack.append(self.StackNode(parent  = node_id,
                                            is_left = False,
                                            depth   = depth+1,
                                            bounds  = (right_bounds_lower, upper_bounds)
                                            )
                             )

                stack.append(self.StackNode(parent  = node_id,
                                            is_left = True,
                                            depth   = depth + 1,
                                            bounds  = (lower_bounds, left_bounds_upper)
                                            )
                             )

        return self

    def update_parent_node_info(self, parent, is_left, node_id):
        """

        :param parent:
        :param is_left:
        :param node_id:
        :return: None
        """

        if is_left:
            self.nodes[parent].left_child = node_id
        else:
            self.nodes[parent].right_child = node_id

    def create_intermediate_parent_nodes(self, node_id, lower_bounds, upper_bounds):
        """
        Function being used in build function

        1. Create the decision node in between the most root node and leaf nodes by randomly picking feature and
        threshold value for each of them, return the updated bounds at the end.

        E.g. lower_bounds: {feature_A: A_min, feature_B: B_min, feature_C: C_min}
             upper_bounds: {feature_A: A_max, feature_B: B_max, feature_C: C_max}

             if choosing feature_A:

             --->  new_lower_bounds for right child node: {feature_A: A_threshold, feature_B: B_min, feature_C: C_min}

                   new_upper_bounds for left child node:  {feature_A: A_threshold, feature_B: B_max, feature_C: C_max}

        2. Create the intermediate_parent_node by instantiating the _Node class and give input parameters node_id,
        splitting feature and threshold value for splitting feature

        3. Append the newly created _Node object to the self.nodes list

        4. count up the self.node_count node counter

        :param node_id:
        :param lower_bounds:
        :param upper_bounds:
        :return:
        """

        # Randomly picking feature and threshold value for the node, and update the bounds
        # --------------------------------------------------------------------------------------------------------------
        feature   = self.random_state.randint(self.n_features)
        threshold = self.random_state.uniform(lower_bounds[feature], upper_bounds[feature])

        lower_bounds[feature] = threshold
        upper_bounds[feature] = threshold

        # Create the intermediate_parent_node by instantiating the _Node class and append the newly created _Node object 
        # to the self.nodes list
        # --------------------------------------------------------------------------------------------------------------
        self.nodes.append(_Node(node_id, feature, threshold))

        # count up the self.node_count node counter
        # --------------------------------------------------------------------------------------------------------------
        self.node_count += 1

        return lower_bounds, upper_bounds

    def create_leaf_nodes(self, node_id):
        """
        Create leaf nodes

        :param node_id:
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

    def fit(self, X, y):
        """
        Fit the tree to the given training data.

        :param X: array-like, shape (n_samples, n_features). Training vector, where n_samples is the number of samples
                                                             and n_features is the number of features.
        :param y: array-like, shape (n_samples,).            Target vector relative to X.
        :return:
        """

        if not self.nodes:
            raise ValueError("Fitting Tree must be built before calling fit().")

        # leaves: vector of (n_samples, 1) indicating the leaf region(node_id) in each data entry
        leaves        = self.apply(X)

        # unique node_id(s)
        unique_leaves = np.unique(leaves)

        #
        values        = np.zeros(shape=(self.node_count, 1, len(self.classes)))

        # Populate value of real leaves
        for leaf in unique_leaves:
            # leaf is node_id

            # idxs: indexes for the data sample who drop in certain leaf(node_id) region
            idxs   = (leaves == leaf)

            # Get their values
            leaf_y = y[idxs]

            # Count the amount of each unique value for the certain leaf(node_id) region
            counts           = [np.sum(leaf_y == cls) for cls in self.classes]

            # compute smooth sensitivity
            count_difference = counts[0] - counts[1]
            sensitivity      = math.exp(-1 * count_difference * self.epsilon)

            # instantiate Exponential mechanism noiser
            mech             = Exponential(epsilon      = self.epsilon,
                                           sensitivity  = sensitivity,
                                           utility      = counts,
                                           random_state = self.random_state)

            values[leaf, 0, mech.randomise()] = 1

        # Populate value of empty leaves
        for node in self.nodes:
            if values[node.node_id].sum() or node.left_child != self._TREE_LEAF:
                continue

            values[node.node_id, 0, self.random_state.randint(len(self.classes))] = 1

        self.values_ = values

        return self

    def apply(self, X):
        """
        Finds the terminal region (=leaf node) for each sample in X.

        :param X:
        :return: out: vector of (n_samples, 1) indicating the leaf region(node_id) in each data entry
        """

        n_samples = X.shape[0]
        out       = np.zeros((n_samples,), dtype=int)
        out_ptr   = out.data

        for i in range(n_samples):
            node = self.nodes[0]

            while node.left_child != self._TREE_LEAF:
                if X[i, node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            out_ptr[i] = node.node_id

        return out


class _Node:
    """
    Base storage structure for the nodes in a _FittingTree object.
    """
    def __init__(self, node_id, feature, threshold):
        self.node_id     = node_id
        self.feature     = feature
        self.threshold   = threshold
        self.left_child  = -1
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
        yield 0    # n_node_samples
        yield 0.0  # weighted_n_node_samples
