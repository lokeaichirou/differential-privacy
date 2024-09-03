
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow_privacy

import pandas as pd

from joblib import delayed, Parallel

from sklearn.utils import check_array, check_consistent_length

from accountant import BudgetAccountant
from models.logistic_regression.dp_logistic_regression import DifferentiallyPrivateLogisticRegression
from tailed_notions import *
from utils import warn_unused_args


def dpsgd(features, labels, batch_size, fit_intercept, clip_norm, epsilon, delta, learning_rate, num_epochs):
    """
    Returns linear regression model computed by DPSGD with given params.

    :param features: Matrix of feature vectors. Assumed to have intercept feature.
    :param labels:   Vector of labels.
    :return:
    """

    n, _        = features.shape
    max_samples = int((n // batch_size) * batch_size)
    print('max_samples: ', max_samples)

    if isinstance(features, pd.DataFrame):
        features = features.reset_index(drop=True)
        labels   = labels.reset_index(drop=True)

        features_processed = features.iloc[:max_samples, :].values
        labels_processed   = labels.iloc[:max_samples].values

    else:
        features_processed = features[:max_samples, :]
        labels_processed   = labels[:max_samples]

    if fit_intercept:
        model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, use_bias=True, activation='sigmoid'))
    else:
        model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, use_bias=False, activation='sigmoid'))

    noise_multiplier = np.sqrt(2*np.log(1.25/delta))/epsilon/clip_norm

    model.compile(optimizer = tensorflow_privacy.DPKerasSGDOptimizer(l2_norm_clip     = clip_norm,
                                                                     noise_multiplier = noise_multiplier,
                                                                     num_microbatches = batch_size,
                                                                     learning_rate    = learning_rate),

                  loss      = 'binary_crossentropy')  # metrics   = ['binary_accuracy']

    model.fit(features_processed,
              labels_processed,
              epochs     = num_epochs,
              batch_size = batch_size,
              verbose    = 0)

    if fit_intercept:
        return np.squeeze(model.layers[0].get_weights()[0]), model.layers[0].get_weights()[1]
    else:
        return np.squeeze(model.layers[0].get_weights()[0])


class DPsgdBasedDPLogisticRegressionOVR(DifferentiallyPrivateLogisticRegression):

    def __init__(self,
                 privacy_notion,
                 fit_intercept,
                 accountant,
                 **unused_args):

        """
        Logistic Regression with differential privacy algorithm class based on dpsgd algorithm [1].

        Reference:
        [1] Abadi, M., Chu, A., Goodfellow, I.J., McMahan, H.B., Mironov, I., Talwar, K., & Zhang, L. (2016).
        Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and
        Communications Security.
        """

        super().__init__(privacy_notion = privacy_notion,
                         fit_intercept  = fit_intercept,
                         multi_class    = 'ovr')

        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight = None):

        num_classes               = len(self.classes_)
        classes_                  = self.classes_
        num_samples, num_features = X.shape

        if len(self.classes_) == 2:
            num_classes = 1
            classes_    = classes_[1:]

        # Perform logistic regression by calling dp_logistic_regression_handler
        # -------------------------------------------------------------------------------------------------------------

        path_func                    = delayed(dp_logistic_regression_handler)
        set_of_coefs_and_intercepts_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes') \
                                       (path_func(X,
                                                  y,
                                                  epsilon       = self.epsilon / num_classes,
                                                  delta         = self.delta,
                                                  pos_class     = class_,
                                                  num_classes   = num_classes,
                                                  fit_intercept = self.fit_intercept,
                                                  verbose       = self.verbose,
                                                  check_input   = True) for class_ in classes_)

        set_of_coefs_, set_of_intercepts = zip(*set_of_coefs_and_intercepts_)

        self.coef_      = np.asarray(set_of_coefs_).reshape(num_classes, num_features)
        self.intercept_ = np.asarray(set_of_intercepts).reshape(num_classes, 1)

        self.accountant.spend(self.epsilon, self.delta)

        return self


def dp_logistic_regression_handler(X,
                                   y,
                                   epsilon,
                                   delta,
                                   pos_class     = None,
                                   fit_intercept = True,
                                   check_input   = True,
                                   **unused_args):
    warn_unused_args(unused_args)

    # Pre-processing.
    # -------------------------------------------------------------------------------------------------------------
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64, accept_large_sparse=True)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)

    classes       = np.unique(y)

    if pos_class is None:
        if classes.size > 2:
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # Mark the non-positive classes if there are more than 2 classes to be classified, so that the binary classification
    # is performed for each class
    masked_y = get_intermediate_data_info(y,
                                          pos_class,
                                          data_type=X.dtype)

    coef, intercept = dpsgd(features      = X,
                            labels        = masked_y,
                            batch_size    = 1,
                            fit_intercept = fit_intercept,
                            clip_norm     = 1,
                            epsilon       = epsilon,
                            delta         = delta,
                            learning_rate = 1,
                            num_epochs    = 20)

    return coef, intercept


def get_intermediate_data_info(y, pos_class, data_type):
    """
    Get some intermediate data info masked y

    :param y:
    :param pos_class:
    :param fit_intercept:
    :return:
    """
    mask         = (y == pos_class)
    y_bin        = np.ones(y.shape, dtype=data_type)
    y_bin[~mask] = -1.0

    return y_bin
