import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec


class DECLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` 
        witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. 
        Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        # If input_dim was inputted instead of input_shape, then change this
        # object's key into "input_shape"
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        # Add all fuctionality from the class of the Layer object to this class
        super(DECLayer, self).__init__(**kwargs)
        # Set parameters of self to input or default values
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        # Check if input_shape from Layer contains two values. Otherwise throw
        # error
        assert len(input_shape) == 2
        # Set input dimension
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform',
                                        name='clusters')
        # If inital weights were supplied; apply them.
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        # Here we compute the soft assignment of each sample. In other words,
        # for each cluster we compute the probability for each sample that it
        # belongs to the given cluster. The computation of q is based on
        # formula 1 from the paper by Junyuan Xie 2016.
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - 
                                         self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(DECLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
