
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow_privacy

from models.linear_regression.dp_linear_regression import DifferentiallyPrivateLinearRegression
from tailed_notions import *


def dpsgd(features, labels, batch_size, fit_intercept, clip_norm, epsilon, delta, learning_rate, num_epochs):
    """
    Returns linear regression model computed by DPSGD with given params.

    :param features: Matrix of feature vectors. Assumed to have intercept feature.
    :param labels:   Vector of labels.
    :return:
    """

    n, _  = features.shape

    features = features.reset_index(drop=True)
    labels   = labels.reset_index(drop=True)

    max_samples        = int((n//batch_size)*batch_size)
    features_processed = features.iloc[:max_samples, :].values
    labels_processed   = labels.iloc[:max_samples].values

    if fit_intercept:
        model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, use_bias=True))
    else:
        model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, use_bias=False))

    noise_multiplier = np.sqrt(2*np.log(1.25/delta))/epsilon/clip_norm

    model.compile(optimizer = tensorflow_privacy.DPKerasSGDOptimizer(l2_norm_clip     = clip_norm,
                                                                     noise_multiplier = noise_multiplier,
                                                                     num_microbatches = batch_size,
                                                                     learning_rate    = learning_rate),

                  loss      = tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE))

    model.fit(features_processed,
              labels_processed,
              epochs     = num_epochs,
              batch_size = batch_size,
              verbose    = 0)

    if fit_intercept:
        return np.squeeze(model.layers[0].get_weights()[0]), model.layers[0].get_weights()[1]
    else:
        return np.squeeze(model.layers[0].get_weights()[0])


class DPsgdBasedDPLinearRegression(DifferentiallyPrivateLinearRegression):
    def __init__(self,
                 privacy_notion,
                 fit_intercept,
                 accountant,
                 **unused_args):

        """
        Linear Regression with differential privacy algorithm class based on dpsgd algorithm [1].

        Reference:
        [1] Abadi, M., Chu, A., Goodfellow, I.J., McMahan, H.B., Mironov, I., Talwar, K., & Zhang, L. (2016).
        Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and
        Communications Security.
        """

        super().__init__(privacy_notion = privacy_notion,
                         fit_intercept  = fit_intercept)

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        :param X:                   array-like or sparse matrix, shape (n_samples, n_features). Training data
        :param y:                   array_like, shape (n_samples, n_targets).                   Target values.
        :param scale_training_data: whether strictly follow the assumption on training data distribution in [1]
        :param sample_weight:
        :return:                    self : returns an instance of self.
        """

        random_state               = self.necessary_steps(X, y, sample_weight)

        self.coef_, self.intercept_ = dpsgd(features      = X,
                                            labels        = y,
                                            batch_size    = 128,
                                            fit_intercept = self.fit_intercept,
                                            clip_norm     = 1,
                                            epsilon       = self.epsilon,
                                            delta         = self.delta,
                                            learning_rate = 1,
                                            num_epochs    = 20)

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        # Deduct the privacy cost from the privacy budget
        # -------------------------------------------------------------------------------------------------------------
        self.accountant.spend(self.epsilon, self.delta)

        return self

