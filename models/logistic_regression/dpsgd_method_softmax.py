
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow_privacy

import pandas as pd

from sklearn.utils import check_array, check_consistent_length

from accountant import BudgetAccountant
from models.logistic_regression.dp_logistic_regression import DifferentiallyPrivateLogisticRegression
from tailed_notions import *
from utils import warn_unused_args


def dpsgd(features, labels, output_dimension, batch_size, fit_intercept, clip_norm, epsilon, delta, learning_rate, num_epochs):
    """
    Returns linear regression model computed by DPSGD with given params.

    :param features: Matrix of feature vectors. Assumed to have intercept feature.
    :param labels:   Vector of labels.
    :return:
    """

    n, _        = features.shape
    max_samples = int((n // batch_size) * batch_size)

    if isinstance(features, pd.DataFrame):
        features = features.reset_index(drop=True)
        labels   = labels.reset_index(drop=True)

        features_processed = features.iloc[:max_samples, :].values
        labels_processed   = labels.iloc[:max_samples].values

    else:
        features_processed = features[:max_samples, :]
        labels_processed   = labels[:max_samples]

    if fit_intercept:
        model = tf.keras.models.Sequential(tf.keras.layers.Dense(units=output_dimension, use_bias=True, activation=tf.nn.softmax))
    else:
        model = tf.keras.models.Sequential(tf.keras.layers.Dense(units=output_dimension, use_bias=False, activation=tf.nn.softmax))

    noise_multiplier = np.sqrt(2*np.log(1.25/delta))/epsilon/clip_norm

    model.compile(optimizer = tensorflow_privacy.DPKerasSGDOptimizer(l2_norm_clip     = clip_norm,
                                                                     noise_multiplier = noise_multiplier,
                                                                     num_microbatches = batch_size,
                                                                     learning_rate    = learning_rate),

                  loss      = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)) # metrics   = ['accuracy']

    model.fit(features_processed,
              labels_processed,
              epochs     = num_epochs,
              batch_size = batch_size,
              verbose    = 0)

    if fit_intercept:
        return np.squeeze(model.layers[0].get_weights()[0]), model.layers[0].get_weights()[1]
    else:
        return np.squeeze(model.layers[0].get_weights()[0])


class DPsgdBasedDPLogisticRegressionSoftmax(DifferentiallyPrivateLogisticRegression):

    def __init__(self,
                 privacy_notion,
                 fit_intercept,
                 output_dimension,
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
                         multi_class    = 'multinomial')

        self.output_dimension = output_dimension

        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):

        self.coef_, self.intercept_ = dp_logistic_regression_handler(X,
                                                                     y,
                                                                     self.epsilon,
                                                                     self.delta,
                                                                     self.output_dimension,
                                                                     fit_intercept = self.fit_intercept,
                                                                     num_epochs    = 20,
                                                                     check_input   = True)

        self.accountant.spend(self.epsilon, self.delta)

        return self


def dp_logistic_regression_handler(X,
                                   y,
                                   epsilon,
                                   delta,
                                   output_dimension,
                                   fit_intercept,
                                   num_epochs,
                                   check_input=True,
                                   **unused_args):
    warn_unused_args(unused_args)

    # Pre-processing.
    # -------------------------------------------------------------------------------------------------------------
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64, accept_large_sparse=True)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)

    if fit_intercept:
        coef, intercept = dpsgd(features      = X,
                                labels        = y,
                                output_dimension = output_dimension,
                                batch_size    = 128,
                                fit_intercept = fit_intercept,
                                clip_norm     = 1,
                                epsilon       = epsilon,
                                delta         = delta,
                                learning_rate = 1,
                                num_epochs    = num_epochs)

    else:
        coef = dpsgd(features      = X,
                     labels        = y,
                     output_dimension = output_dimension,
                     batch_size    = 128,
                     fit_intercept = fit_intercept,
                     clip_norm     = 1,
                     epsilon       = epsilon,
                     delta         = delta,
                     learning_rate = 1,
                     num_epochs    = num_epochs)

        intercept = 0

    return coef.T, intercept
