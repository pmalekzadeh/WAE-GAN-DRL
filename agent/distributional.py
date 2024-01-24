from absl import flags
from typing import Optional, Sequence, Union
import enum
from acme.tf import networks
from acme import types
from acme.tf import utils as tf2_utils

from acme.tf.networks import distributions as ad
from acme.tf.networks import DiscreteValuedHead, CriticMultiplexer, LayerNormMLP
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
import numpy as np
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization

from scipy.stats import norm

tfd = tfp.distributions


uniform_initializer = tf.initializers.VarianceScaling(
    distribution='uniform', mode='fan_out', scale=0.333)


FLAGS = flags.FLAGS


class RiskDiscreteValuedDistribution(ad.DiscreteValuedDistribution):
    def __init__(self,
                 values: tf.Tensor,
                 logits: Optional[tf.Tensor] = None,
                 probs: Optional[tf.Tensor] = None,
                 name: str = 'RiskDiscreteValuedDistribution'):
        super().__init__(values, logits, probs, name)

    def _normal_dist_volc(self, quantile):
        prob_density = round(norm.ppf(quantile), 4)
        return prob_density

    def meanstd(self) -> tf.Tensor:
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc*self.stddev()

    def var(self, th) -> tf.Tensor:
        """Implements mean-volc*std for VaR estimation"""
        volc = self._normal_dist_volc(th)
        return self.mean() - volc*self.stddev()

    def cvar(self, th) -> tf.Tensor:
        quantile = 1 - th
        cdf = tf.cumsum(self.probs_parameter(), axis=-1)
        exclude_logits = cdf > quantile
        zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
        clogits = tf.where(exclude_logits, zero, self.probs_parameter())
        return tf.reduce_sum(clogits * self.values, axis=-1)

    def gain_loss_tradeoff(self) -> tf.Tensor:
        """Implements gain_loss tradeoff objective function"""
        zero = tf.constant(0, dtype=tf.float32)
        return tf.reduce_mean(FLAGS.k1 * tf.pow(tf.maximum(zero, self._values), FLAGS.alpha) - FLAGS.k2 * tf.pow(tf.maximum(zero, -self._values), FLAGS.alpha), axis=-1)


class RiskDiscreteValuedHead(DiscreteValuedHead):
    def __init__(self,
                 vmin: Union[float, np.ndarray, tf.Tensor],
                 vmax: Union[float, np.ndarray, tf.Tensor],
                 num_atoms: int,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
        super().__init__(vmin, vmax, num_atoms, w_init, b_init)

    def __call__(self, inputs: tf.Tensor) -> RiskDiscreteValuedDistribution:
        logits = self._distributional_layer(inputs)
        logits = tf.reshape(logits,
                            tf.concat([tf.shape(logits)[:1],  # batch size
                                       tf.shape(self._values)],
                                      axis=0))
        values = tf.cast(self._values, logits.dtype)

        return RiskDiscreteValuedDistribution(values=values, logits=logits)


def quantile_project(  # pylint: disable=invalid-name
    q: tf.Tensor,
    v: tf.Tensor,
    q_grid: tf.Tensor,
) -> tf.Tensor:
    """Project quantile distribution (quantile_grid, values) onto quantile under the L2-metric over CDFs.

    This projection works for any support q.
    Let Kq be len(q_grid)

    Args:
    q: () quantile
    v: (batch_size, Kq) values to project onto
    q_grid:  (Kq,) Quantiles for P(Zp[i])

    Returns:
    Quantile projection of (q_grid, v) onto q.
    """

    # Asserts that Zq has no leading dimension of size 1.
    if q_grid.get_shape().ndims > 1:
        q_grid = tf.squeeze(q_grid, axis=0)
    q = q[None]
    # Extracts vmin and vmax and construct helper tensors from Zq.
    vmin, vmax = q_grid[0], q_grid[-1]
    d_pos = tf.concat([q_grid, vmin[None]], 0)[1:]
    d_neg = tf.concat([vmax[None], q_grid], 0)[:-1]

    # Clips Zp to be in new support range (vmin, vmax).
    clipped_q = tf.clip_by_value(q, vmin, vmax)  # (1,)
    eq_mask = tf.cast(tf.equal(q_grid, q), q_grid.dtype)
    if tf.equal(tf.reduce_sum(eq_mask), 1.0):
        # (batch_size, )
        return tf.squeeze(tf.boolean_mask(v, eq_mask, axis=1), axis=-1)

    # need interpolation
    pos_neg_mask = tf.cast(tf.roll(q_grid <= q, 1, axis=0), q_grid.dtype) \
        * tf.cast(tf.roll(q_grid >= q, -1, axis=0), q_grid.dtype)
    pos_neg_v = tf.boolean_mask(v, pos_neg_mask, axis=1)    # (batch_size, 2)

    # Gets the distance between atom values in support.
    d_pos = (d_pos - q_grid)[None, :]  # (1, Kq)
    d_neg = (q_grid - d_neg)[None, :]  # (1, Kq)

    clipped_q_grid = q_grid[None, :]  # (1, Kq)
    delta_qp = clipped_q - clipped_q_grid  # (1, Kq)

    d_sign = tf.cast(delta_qp >= 0., dtype=v.dtype)
    delta_hat = (d_sign * delta_qp / d_pos) - \
        ((1. - d_sign) * delta_qp / d_neg)  # (1, Kq)
    # (batch_size, )
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * v, 1)


@tfp.experimental.register_composite
class QuantileDistribution(tfd.Categorical):
    def __init__(self,
                 values: tf.Tensor,
                 quantiles: tf.Tensor,
                 probs: tf.Tensor,
                 name: str = 'QuantileDistribution'):
        """Quantile Distribution
        values: (batch_size, Kq)
        quantiles: (Kq,) or (batch_size, Kq)
        probs: (Kq,)
        """
        self._quantiles = tf.convert_to_tensor(quantiles)
        self._shape_strings = [f'D{i}' for i, _ in enumerate(quantiles.shape)]
        self._values = tf.convert_to_tensor(values)
        self._probs = tf.convert_to_tensor(probs)

        super().__init__(probs=probs, name=name)
        self._parameters = dict(values=values,
                                quantiles=quantiles,
                                probs=probs,
                                name=name)

    @property
    def quantiles(self) -> tf.Tensor:
        return self._quantiles

    @property
    def values(self) -> tf.Tensor:
        return self._values

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            values=tfp.util.ParameterProperties(
                event_ndims=lambda self: self.quantiles.shape.rank),
            quantiles=tfp.util.ParameterProperties(
                event_ndims=None),
            probs=tfp.util.ParameterProperties(
                event_ndims=None))

    def _sample_n(self, n, seed=None) -> tf.Tensor:
        indices = super()._sample_n(n, seed=seed)
        return tf.gather(self.values, indices, axis=-1)

    def _mean(self) -> tf.Tensor:
        # assume values are always with equal prob
        return tf.reduce_mean(self.values, axis=-1)

    def _variance(self) -> tf.Tensor:
        dist_squared = tf.square(tf.expand_dims(self.mean(), -1) - self.values)
        return tf.reduce_sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self):
        # Omit the atoms axis, to return just the shape of a single (i.e. unbatched)
        # sample value.
        return self._quantiles.shape[:-1]

    def _event_shape_tensor(self):
        return tf.shape(self._quantiles)[:-1]

    def meanstd(self) -> tf.Tensor:
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc*self.stddev()

    def var(self, th) -> tf.Tensor:
        quantile = tf.convert_to_tensor(1 - th)
        return quantile_project(quantile, self._values, self.quantiles)

    def cvar(self, th) -> tf.Tensor:
        quantile = 1 - th
        cdf = tf.cumsum(self.probs_parameter(), axis=-1)
        exclude_probs = cdf > quantile
        zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
        cprobs = tf.where(exclude_probs, zero, self.probs_parameter())
        return tf.reduce_sum(cprobs * self.values, axis=-1)

    def gain_loss_tradeoff(self) -> tf.Tensor:
        """Implements gain_loss tradeoff objective function"""
        zero = tf.constant(0, dtype=tf.float32)
        return tf.reduce_mean(FLAGS.k1 * tf.pow(tf.maximum(zero, self._values), FLAGS.alpha) - FLAGS.k2 * tf.pow(tf.maximum(zero, -self._values), FLAGS.alpha), axis=-1)


class QuantileDistProbType(enum.Enum):
    LEFT = 1
    MID = 2
    RIGHT = 3


class QuantileDiscreteValuedHead(snt.Module):
    def __init__(self,
                 quantiles: np.ndarray,
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
        super().__init__(name='QuantileDiscreteValuedHead')
        self._quantiles = tf.convert_to_tensor(quantiles)
        assert quantiles[0] > 0
        assert quantiles[-1] < 1.0
        left_probs = quantiles - np.insert(quantiles[:-1], 0, 0.0)
        right_probs = np.insert(
            quantiles[1:], len(quantiles)-1, 1.0) - quantiles
        if prob_type == QuantileDistProbType.LEFT:
            probs = left_probs
        elif prob_type == QuantileDistProbType.MID:
            probs = (left_probs + right_probs) / 2
        elif prob_type == QuantileDistProbType.RIGHT:
            probs = right_probs
        self._probs = tf.convert_to_tensor(probs)
        self._distributional_layer = snt.Linear(tf.size(self._quantiles),
                                                w_init=w_init,
                                                b_init=b_init)

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        quantile_values = self._distributional_layer(inputs)
        quantile_values = tf.reshape(quantile_values,
                                     tf.concat([tf.shape(quantile_values)[:1],
                                                tf.shape(self._quantiles)],
                                               axis=0))
        quantiles = tf.cast(self._quantiles, quantile_values.dtype)
        probs = tf.cast(self._probs, quantile_values.dtype)
        return QuantileDistribution(values=quantile_values, quantiles=quantiles,
                                    probs=probs)

################ GAN Networks
class GeneratorHead(snt.Module):
    """Module that outputs samples of Generator."""

    def __init__(self, n_samples, w_init=snt.initializers.VarianceScaling(1e-4), b_init=snt.initializers.Zeros()):
        super().__init__(name='GeneratorHead')
        self.n_samples = n_samples
        self._distributional_layer = snt.Linear(self.n_samples, w_init=w_init, b_init=b_init)

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        # Generate returns for each sample in the batch.
        generated_returns = self._distributional_layer(inputs)

        # Reshape to [batch_size, n_samples] to represent an empirical distribution for each batch element.
        generated_returns = tf.reshape(generated_returns, [tf.shape(inputs)[0], self.n_samples])

        return tfd.Empirical(samples=generated_returns, event_ndims=1)


class EmpiricalDistribution(tfd.Empirical):
    """Module that outputs cvar of Empirical distribution."""

    def __init__(self, values, validate_args=False, allow_nan_stats=True, name='EmpiricalDistribution'):
        self._values = tf.convert_to_tensor(values)
        self._num_values = tf.shape(self._values)[-1]
        self._probs = tf.fill(self._num_values, 1.0 / tf.cast(self._num_values, dtype=tf.float32))
        self._sorted_values = tf.sort(self._values)

        parameters = dict(locals())
        super().__init__(
            dtype=self._values.dtype,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            values=tfp.util.ParameterProperties(event_ndims=1))

    def _event_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return tf.shape([])

    def _mean(self):
        return tf.reduce_mean(self._values, axis=-1)

    def _sample_n(self, n, seed=None):
        indices = tf.random.uniform(shape=[n], maxval=self._num_values, dtype=tf.int32, seed=seed)
        return tf.gather(self._values, indices, axis=-1)

    def _variance(self):
        mean = self._mean()
        squared_diff = tf.square(self._values - mean[..., tf.newaxis])
        return tf.reduce_mean(squared_diff, axis=-1)

    def _stddev(self):
        return tf.sqrt(self._variance())

    def meanstd(self):
        """Implements mean - volc * stddev."""
        volc = FLAGS.std_coef
        return self._mean() - volc * self._stddev()

    def var(self, th):
        quantile = 1 - th
        index = tf.cast(quantile * tf.cast(tf.size(self._values), tf.float32), tf.int32)
        return tf.gather(self._sorted_values, index)

    def cvar(self, th):
        quantile = 1 - th
        threshold_value = self.var(th)
        eligible_values = tf.boolean_mask(self._values, self._values <= threshold_value)
        return tf.reduce_mean(eligible_values)

    def values(self):
        return self._values

    def gain_loss_tradeoff(self, k1, k2, alpha):
        zero = tf.constant(0, dtype=tf.float32)
        gains = tf.pow(tf.maximum(zero, self._values), alpha)
        losses = tf.pow(tf.maximum(zero, -self._values), alpha)
        return tf.reduce_mean(k1 * gains - k2 * losses, axis=-1)


class EncoderHead(snt.Module):
    """Module that outputs mean and variance of Encoder distribution."""

    def __init__(self,  latent_dim: int, loc_layer_sizes: Sequence[int] = (512, 512, 256), scale_layer_sizes: Sequence[int] = (512, 512, 256),
                 init_scale=0.3, min_scale=1e-6,
                 w_init=snt.initializers.VarianceScaling(1e-4),
                 b_init=snt.initializers.Zeros()):

        super().__init__(name='EncoderMultivariateNormalDiagHead')

        self._init_scale = init_scale
        self._min_scale = min_scale
        self.latent_dim=latent_dim

        # Define location network
        self._loc_layer = snt.Sequential([
            DynamicMultiplexer(),
            networks.LayerNormMLP(loc_layer_sizes, activate_final=True),
            snt.Linear(latent_dim, w_init=w_init, b_init=b_init)
        ])

        # Define scale network
        def scale_transformation(inputs):
            # Apply softplus and ensure the scale is non-zero
            scale = tf.nn.softplus(inputs)
            zero = tf.zeros_like(scale)
            scale *= self._init_scale / tf.nn.softplus(zero)
            scale += self._min_scale
            return scale

        self._scale_layer = snt.Sequential([
            DynamicMultiplexer(),
            networks.LayerNormMLP(scale_layer_sizes, activate_final=True),
            snt.Linear(latent_dim, w_init=w_init, b_init=b_init),
            scale_transformation  # Add this layer for the scale transformation
        ])

        @property
        def loc_network(self):
            return self._loc_layer

        @property
        def scale_network(self):
            return self._scale_layer


class DynamicMultiplexer(snt.Module):
    """Multiplexer module for dynamically concatenating multiple inputs.

    This module can take any number of inputs and concatenates them along the batch dimension.
    """

    def __init__(self):
        super().__init__(name='dynamic_multiplexer')

    def __call__(self, *inputs: types.NestedTensor) -> tf.Tensor:
        # Ensure all inputs are of the same type for concat to work
        dtype = inputs[0].dtype
        casted_inputs = [tf.cast(input_tensor, dtype) if input_tensor.dtype != dtype else input_tensor for input_tensor in inputs]

        # Concatenate all inputs, with one batch dimension.
        concatenated_outputs = tf2_utils.batch_concat(casted_inputs)
        return concatenated_outputs

####

class IQNHead(snt.Module):
    def __init__(self, th, n_cos=64, n_tau=8, n_k=32, layer_size: int = 256,
                 quantiles: np.ndarray = np.arange(0.01, 1.0, 0.01),
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = uniform_initializer):
        super().__init__(name='IQNHead')
        self._th = th
        self._n_cos = n_cos
        self._n_tau = n_tau
        self._n_k = n_k
        self._pis = tf.convert_to_tensor(
            [np.pi*i for i in range(self._n_cos)])[None, None, :]
        self._phi = snt.nets.MLP(
            (layer_size,),
            w_init=w_init,
            activation=tf.nn.relu,
            activate_final=True)
        self._f = snt.nets.MLP(
            (layer_size, 1),
            w_init=w_init,
            activation=tf.nn.elu,
            activate_final=False
        )
        self._quantiles = tf.convert_to_tensor(quantiles)
        assert quantiles[0] > 0
        assert quantiles[-1] < 1.0
        left_probs = quantiles - np.insert(quantiles[:-1], 0, 0.0)
        right_probs = np.insert(
            quantiles[1:], len(quantiles)-1, 1.0) - quantiles
        if prob_type == QuantileDistProbType.LEFT:
            probs = left_probs
        elif prob_type == QuantileDistProbType.MID:
            probs = (left_probs + right_probs) / 2
        elif prob_type == QuantileDistProbType.RIGHT:
            probs = right_probs
        self._probs = tf.convert_to_tensor(probs)

    def __call__(self, inputs: tf.Tensor, policy=False) -> tfd.Distribution:
        if not policy:
            # (batch, n_tau, 1)
            taus = tf.random.uniform(
                (inputs.shape[0], self._n_tau, 1), 0, 1, dtype=inputs.dtype)
        else:
            # (batch, n_k, 1)
            taus = tf.random.uniform(
                (inputs.shape[0], self._n_k, 1), 0, 1, dtype=inputs.dtype)*(1-self._th)
        cos = tf.cos(taus*self._pis)        # (batch, n_tau, n_cos)
        cos_x = self._phi(cos)              # (batch, n_tau, n_layer)
        x = tf.expand_dims(inputs, axis=1)  # (batch, 1, n_layer)
        icdf = self._f(x*cos_x)             # (batch, n_tau, 1)
        if not policy:
            taus = tf.transpose(taus, [2, 0, 1])  # (1, batch, n_tau)
        probs = tf.cast(self._probs, inputs.dtype)
        return QuantileDistribution(values=tf.squeeze(icdf, axis=-1), quantiles=taus, probs=probs)


class IQNCritic(snt.Module):
    def __init__(self, th, n_cos=64, n_tau=8, n_k=32,
                 critic_layer_sizes: Sequence[int] = (512, 512, 256),
                 quantiles: np.ndarray = np.arange(0.01, 1.0, 0.01),
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = uniform_initializer):
        super().__init__(name='IQNCritic')
        self._head = snt.Sequential([
            # The multiplexer concatenates the observations/actions.
            CriticMultiplexer(),
            LayerNormMLP(critic_layer_sizes, activate_final=True)])
        self._iqn = IQNHead(th, n_cos, n_tau, n_k,
                            critic_layer_sizes[-1], quantiles, prob_type, w_init)

    def __call__(self, observation: tf.Tensor, action: tf.Tensor, policy=False) -> tfd.Distribution:
        return self._iqn(self._head(observation, action), policy)


def huber(x: tf.Tensor, k=1.0):
    return tf.where(tf.abs(x) < k, 0.5 * tf.pow(x, 2), k * (tf.abs(x) - 0.5 * k))


def quantile_regression(q_tm1: QuantileDistribution, r_t: tf.Tensor,
                        d_t: tf.Tensor,
                        q_t: QuantileDistribution):
    """Implements Quantile Regression Loss
    q_tm1: critic distribution of t-1
    r_t:   reward
    d_t:   discount
    q_t:   target critic distribution of t
    """

    z_t = tf.reshape(r_t, (-1, 1)) + tf.reshape(d_t, (-1, 1)) * q_t.values
    z_tm1 = q_tm1.values
    diff = tf.expand_dims(tf.transpose(z_t), -1) - \
        z_tm1    # (n_tau_p, n_batch, n_tau)
    k = 1
    loss = huber(diff, k) * tf.abs(q_tm1.quantiles -
                                   tf.cast(diff < 0, diff.dtype)) / k

    return tf.reduce_mean(loss, (0, -1))

# def discriminator_loss(x_tilde_t: tf.Tensor, x_t: tf.Tensor,
#                         z_t: tf.Tensor,
#                         z_tilde_t: tf.Tensor,
#                         dis_network: snt.Module):
#     """Implements Quantile Regression Loss
#     q_tm1: critic distribution of t-1
#     r_t:   reward
#     d_t:   discount
#     q_t:   target critic distribution of t
#     """
#
#     loss = dis_network(x_tilde_t,z_t,  )
#
#     return tf.reduce_mean(loss, (0, -1))
