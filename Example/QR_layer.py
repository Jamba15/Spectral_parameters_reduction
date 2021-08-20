import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops.gen_math_ops import mul, mat_mul


class MatrixInitializer(tf.keras.initializers.Initializer):

    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, shape, dtype=None):
        return tf.reshape(self.matrix, shape=shape)


@keras_export('keras.layers.QR')
class QR(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 is_eig_in_trainable=True,
                 is_eig_out_trainable=True,
                 use_bias=False,
                 eig_in_initializer='optimized_uniform',
                 eig_out_initializer='optimized_uniform',
                 qr_initializer='optimized_uniform',
                 bias_initializer='zeros',
                 eig_in_regularizer=None,
                 eig_out_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 eig_in_constraint=None,
                 eig_out_constraint=None,
                 bias_constraint=None,
                 density=1,
                 phi_mean=0,
                 phi_stddev=0.036,
                 eig_in=0.97,
                 eig_out=0.13,
                 **kwargs):

        super(QR, self).__init__(
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.bias = None
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.is_eig_in_trainable = is_eig_in_trainable
        self.is_eig_out_trainable = is_eig_out_trainable
        self.use_bias = use_bias

        # 'optimized_uniform' initializers optmized by Buffoni and Giambagli

        if qr_initializer == 'optimized_uniform':
            self.qr_initializer = initializers.random_normal(mean=phi_mean,
                                                             stddev=phi_stddev)  # (mean=0, stddev=0.3)#RandomUniform(-0.01, 0.01)#(-0.5, 0.5)
        else:
            self.qr_initializer = initializers.get(qr_initializer)

        if eig_in_initializer == 'optimized_uniform':
            self.eig_in_initializer = initializers.RandomUniform(-eig_in, eig_in)  # (-0.01, 0.01)
        else:
            self.eig_in_initializer = initializers.get(eig_in_initializer)

        if eig_out_initializer == 'optimized_uniform':
            self.eig_out_initializer = initializers.RandomUniform(-eig_out, eig_out)  # (-0.0001, 0.0001)
        else:
            self.eig_out_initializer = initializers.get(eig_out_initializer)

        self.bias_initializer = initializers.get(bias_initializer)

        self.eig_in_regularizer = regularizers.get(eig_in_regularizer)

        self.eig_out_regularizer = regularizers.get(eig_out_regularizer)

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.eig_in_constraint = constraints.get(eig_in_constraint)

        self.eig_out_constraint = constraints.get(eig_out_constraint)

        self.bias_constraint = constraints.get(bias_constraint)

        # Sparsity
        self.p = density

    def build(self, input_shape):

        # eigenvector matrix to be decomposed with QR
        base = self.qr_initializer(shape=(self.units, input_shape[-1]))
        q_debug, r_debug = tf.linalg.qr(tf.transpose(base), full_matrices=False)
        self.q_debug = tf.transpose(q_debug)
        self.r_debug = tf.transpose(r_debug)

        self.q = self.add_weight(
            name='Q',
            shape=self.q_debug.shape,
            initializer=MatrixInitializer(self.q_debug),  # SVDInitializer(u),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False
        )

        self.r = self.add_weight(
            name='R',
            shape=self.r_debug.shape,
            initializer=MatrixInitializer(self.r_debug),  # SVDInitializer(u),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True
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

        self.built = True

    def call(self, inputs, **kwargs):

        # QR is transposed due to parameters' number optimization. R matrix is the smallest possible
        self.phi = mat_mul(self.r, self.q)

        lin = mul(self.eig_in - self.eig_out, self.phi)
        if self.use_bias:
            action = mat_mul(inputs, tf.transpose(lin)) + self.bias
        else:
            action = mat_mul(inputs, tf.transpose(lin))
        return self.activation(action)
