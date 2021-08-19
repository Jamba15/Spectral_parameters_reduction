from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops.gen_math_ops import mul, mat_mul
from tensorflow.python.ops.linalg.linalg import svd, transpose
"""

Implementation of a Spectral + SVD layer that can learn eigenvalues of the associated adjacency matrix and of its SVD.

@authors: Lorenzo Chicchi, Lorenzo Buffoni, Lorenzo Giambagli

"""


@keras_export('keras.layers.SSVD')
class SSVD(Layer):

    def __init__(self,
                 units,
                 activation=None,
                 is_eig_in_trainable=True,
                 is_eig_out_trainable=True,
                 is_svd_trainable=True,
                 use_bias=False,
                 eig_in_initializer='optimized_uniform',
                 eig_out_initializer='optimized_uniform',
                 svd_initializer='optimized_uniform',
                 bias_initializer='zeros',
                 eig_in_regularizer=None,
                 eig_out_regularizer=None,
                 svd_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 eig_in_constraint=None,
                 eig_out_constraint=None,
                 svd_constraint=None,
                 bias_constraint=None,
                 phi_mean=0,
                 phi_stddev=0.036,
                 eig_in=0.97,
                 eig_out=0.13,
                 primo=True,
                 **kwargs):

        super(SSVD, self).__init__(
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.is_eig_in_trainable = is_eig_in_trainable
        self.is_eig_out_trainable = is_eig_out_trainable
        self.is_svd_trainable = is_svd_trainable
        self.use_bias = use_bias

        # 'optimized_uniform' initializers optmized by Buffoni and Giambagli

        if svd_initializer == 'optimized_uniform':
            self.svd_initializer = initializers.random_normal(mean=phi_mean,
                                                              stddev=phi_stddev)  # (mean=0, stddev=0.3)#RandomUniform(-0.01, 0.01)#(-0.5, 0.5)
        else:
            self.svd_initializer = initializers.get(svd_initializer)

        if eig_in_initializer == 'optimized_uniform':
            self.eig_in_initializer = initializers.RandomUniform(-eig_in, eig_in)  # (-0.01, 0.01)
        else:
            self.eig_in_initializer = initializers.get(eig_in_initializer)

        if eig_out_initializer == 'optimized_uniform':
            self.eig_out_initializer = initializers.RandomUniform(-eig_out, eig_out)  # (-0.0001, 0.0001)
        else:
            self.eig_out_initializer = initializers.get(eig_out_initializer)

        self.bias_initializer = initializers.get(bias_initializer)

        self.svd_regularizer = regularizers.get(svd_regularizer)

        self.eig_in_regularizer = regularizers.get(eig_in_regularizer)

        self.eig_out_regularizer = regularizers.get(eig_out_regularizer)

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.svd_constraint = constraints.get(svd_constraint)

        self.eig_in_constraint = constraints.get(eig_in_constraint)

        self.eig_out_constraint = constraints.get(eig_out_constraint)

        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        # eigenvector matrix to be decomposed with SVD

        base = self.svd_initializer(shape=(self.units, input_shape[-1]))
        self.s_debug, self.u_debug, self.v_debug = svd(base, full_matrices=False, compute_uv=True)
        v = transpose(self.v_debug)
        u = self.u_debug

        u_in = initializers.random_normal(mean=0, stddev=0.04)

        self.u = self.add_weight(
            name='U',
            shape=u.shape,
            initializer=u_in,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False
        )

        self.v = self.add_weight(
            name='V',
            shape=v.shape,
            initializer=initializers.random_normal(mean=0.0, stddev=0.04),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False
        )

        s_shape = min(self.units, input_shape[-1])

        self.s = self.add_weight(
            name='S',
            shape=(s_shape, 1),
            initializer=initializers.RandomUniform(minval=0, maxval=3),
            regularizer=self.svd_regularizer,
            constraint=self.svd_constraint,
            dtype=self.dtype,
            trainable=self.is_svd_trainable
        )

        # trainable input eigenvalues

        self.eig_in = self.add_weight(
            name='eig_in',
            shape=(1, input_shape[-1]),
            initializer=self.eig_in_initializer,
            regularizer=self.eig_in_regularizer,
            constraint=self.eig_in_constraint,
            dtype=self.dtype,
            trainable=self.is_eig_in_trainable
        )

        # trainable output eigenvalues

        self.eig_out = self.add_weight(
            name='eig_out',
            shape=(self.units, 1),
            initializer=self.eig_out_initializer,
            regularizer=self.eig_out_regularizer,
            constraint=self.eig_out_constraint,
            dtype=self.dtype,
            trainable=self.is_eig_out_trainable
        )

        # bias

        if self.use_bias:

            self.bias = self.add_weight(
                name='bias',
                shape=(1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)

        else:
            self.bias = None

        self.built = True

    def call(self, inputs, **kwargs):

        phi = mat_mul(self.u, mul(self.s, self.v))
        lin = mul(self.eig_in - self.eig_out, phi)
        if self.use_bias:
            action = mat_mul(inputs, transpose(lin)) + self.bias
        else:
            action = mat_mul(inputs, transpose(lin))
        return self.activation(action)

