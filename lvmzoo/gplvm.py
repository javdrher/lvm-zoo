# LVM Zoo, some Latent Variable Models using GPflow
# Copyright (C) 2017  Nicolas Knudde, Joachim van der Herten

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
import numpy as np
from GPflow.gplvm import BayesianGPLVM
from GPflow.param import AutoFlow
from GPflow.mean_functions import Constant, Zero
from GPflow import settings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type


class GPLVM(BayesianGPLVM):

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood for the training data (all dimensions).
        """
        # E_q
        bound = self._build_marginal_bound(self.X_mean, self.X_var, self.Y)

        # KL[q(x) || p(x)]
        KL = self._build_kl(self.X_mean, self.X_var, self.X_prior_mean, self.X_prior_var)

        return bound - KL

    def _build_marginal_bound(self, X_mean, X_var, Y):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood given a Gaussian multivariate distribution representing
        X (and its priors) and observed Y

        Split from the general build_likelihood method, as the graph is reused by the held_out_data_objective
        method for inference of latent points for new data points
        """

        num_inducing = tf.shape(self.Z)[0]
        D = tf.constant(self.output_dim, dtype=float_type)

        psi0 = tf.reduce_sum(self.kern.eKdiag(X_mean, X_var), 0)
        psi1 = self.kern.eKxz(self.Z, X_mean, X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_mean, X_var), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / tf.sqrt(sigma2)
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, Y), lower=True) / tf.sqrt(sigma2)

        # Bound
        ND = tf.cast(tf.size(Y), float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        return bound

    def _build_kl(self, X_mean, X_var, X_prior_mean, X_prior_var):
        dX_var = X_var if len(X_var.get_shape()) == 2 else tf.matrix_diag_part(X_var)
        NQ = tf.cast(tf.size(X_mean), float_type)
        KL = -0.5 * tf.reduce_sum(tf.log(dX_var)) \
             + 0.5 * tf.reduce_sum(tf.log(X_prior_var)) \
             - 0.5 * NQ \
             + 0.5 * tf.reduce_sum((tf.square(X_mean - X_prior_mean) + dX_var) / X_prior_var)
        return KL

    def build_predict_density(self, Xstarmu, Xstarvar):
        """
        Build the graph to map a latent point to its corresponding output. Unlike build_predict, here a
        Gaussian density is mapped rather than only its mean as in build_predict. Details of the calculation
        are in the gplvm predict x notebook.

        :param Xstarmu: mean of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :param Xstarvar: variance of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :return: tensor for computation of the moments of the output distribution.
        """
        num_inducing = tf.shape(self.Z)[0] # M
        num_predict = tf.shape(Xstarmu)[0] # N*
        num_out = self.output_dim     # p

        # Kernel expectations, w.r.t q(X) and q(X*)
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var) # N x M
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0) # M x M
        psi0star = self.kern.eKdiag(Xstarmu, Xstarvar) # N*
        psi1star = self.kern.eKxz(self.Z, Xstarmu, Xstarvar) # N* x M
        psi2star = self.kern.eKzxKxz(self.Z, Xstarmu, Xstarvar) # N* x M x M

        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * settings.numerics.jitter_level # M x M
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu) # M x M

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma # M x N
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True) # M x M
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B) # M x M
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma # M x p
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)

        # All of these: N* x M x M
        L3 = tf.tile(tf.expand_dims(L, 0), [num_predict, 1, 1])
        LB3 = tf.tile(tf.expand_dims(LB, 0), [num_predict, 1, 1])
        tmp3 = tf.matrix_triangular_solve(LB3, tf.matrix_triangular_solve(L3, tf.expand_dims(psi1star, -1)))
        tmp4 = tf.matmul(tmp3, tmp3, transpose_b=True)
        tmp5 = tf.matrix_triangular_solve(L3, tf.transpose(tf.matrix_triangular_solve(L3, psi2star), perm=[0, 2, 1]))
        tmp6 = tf.matrix_triangular_solve(LB3, tf.transpose(tf.matrix_triangular_solve(LB3, tmp5), perm=[0, 2, 1]))

        c3 = tf.tile(tf.expand_dims(c, 0), [num_predict, 1, 1])  # N* x M x p
        TT = tf.trace(tmp5 - tmp6)  # N*
        diagonals = tf.einsum("ij,k->ijk", tf.eye(num_out, dtype=float_type), psi0star - TT) # p x p x N*
        covar1 = tf.matmul(c3, tf.matmul(tmp6 - tmp4, c3), transpose_a=True)  # N* x p x p
        covar2 = tf.transpose(diagonals, perm=[2, 0, 1])  # N* x p x p
        covar = covar1 + covar2
        return mean + self.mean_function(Xstarmu), covar

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (int_type, [None]))
    def held_out_data_objective(self, Ynew, mu, var, observed):
        """
        TF computation of likelihood objective + gradients, given new observed points and a candidate q(X*)
        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions), with k <= D.
        :param mu: candidate mean, np.ndarray of size Nnew (number of new points) x Q (latent dimensions)
        :param var: candidate variance, np.ndarray of size Nnew (number of new points) x Q (latent dimensions)
        :param observed: indices for the observed dimensions np.ndarray of size k
        :return: returning a tuple (objective,gradients). gradients is a list of 2 matrices for mu and var of size
        Nnew x Q
        """
        idx = tf.expand_dims(observed, -1)
        Y_obs = tf.transpose(tf.gather_nd(tf.transpose(self.Y), idx))
        X_mean = tf.concat([self.X_mean, mu], 0)
        X_var = tf.concat([self.X_var, var], 0)
        Y = tf.concat([Y_obs, Ynew], 0)

        X_prior_mean = tf.concat((self.X_prior_mean, tf.zeros((tf.shape(mu)[0], self.num_latent), float_type)), axis=0)
        X_prior_var = tf.concat((self.X_prior_var, tf.ones((tf.shape(mu)[0], self.num_latent), float_type)), axis=0)

        # Build the likelihood graph for the suggested q(X,X*) and the observed dimensions of Y and Y*
        objective = self._build_marginal_bound(X_mean, X_var, Y)
        objective -= self._build_kl(X_mean, X_var, X_prior_mean, X_prior_var)

        # Collect gradients
        gradients = tf.gradients(objective, [mu, var])

        f = tf.negative(objective, name='objective')
        g = tf.negative(gradients, name='grad_objective')
        return f, g

    def _held_out_data_wrapper_creator(self, Ynew, observed):
        """
        Private wrapper function for returning an objective function accepted by scipy.optimize.minimize
        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions)
        :return: function accepting a flat numpy array of size 2 * Nnew (number of new points) * Q (latent dimensions)
        and returning a tuple (objective,gradient)
        """
        infer_number = Ynew.shape[0]
        half_num_param = infer_number * self.num_latent

        def fun(x_flat):
            # Unpack q(X*) candidate
            mu_new = x_flat[:half_num_param].reshape((infer_number, self.num_latent))
            var_new = x_flat[half_num_param:].reshape((infer_number, self.num_latent))

            # Compute likelihood & flatten gradients
            f,g = self.held_out_data_objective(Ynew, mu_new, var_new, observed)
            return f, np.hstack(map(lambda gradient: gradient.flatten(), g))

        return fun

    def infer_latent_inputs(self, Ynew, method='L-BFGS-B', tol=None, return_logprobs=False, observed=None, **kwargs):
        """
        Computes the latent representation of new observed points by maximizing
        .. math::

            p(Y*|Y)

        It is automatically assumed all dimensions D were observed unless the observed parameter is specified.

        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions). with k <= D.
        :param method: method is a string (default 'L-BFGS-B') specifying the scipy optimization routine
        :param tol: tol is the tolerance to be passed to the optimization routine
        :param kern: kernel specification, by default RBF
        :param return_logprobs: return the likelihood probability after optimization (default: False)
        :param observed: list of dimensions specified with length k. None (the default) indicates all D were observed
        :param kwargs: passed on to the options field of the scipy minimizer

        :returns (mean, var) or (mean, var, prob) in case return_logprobs is true.
        :rtype mean, var: np.ndarray, size Nnew (number of new points ) x Q (latent dim)
        """

        observed = np.arange(self.Y.shape[1], dtype=np.int32) if observed is None else np.atleast_1d(observed)
        assert (Ynew.shape[1] == observed.size)
        infer_number = Ynew.shape[0]

        # Initialization: could do this with tf?
        nearest_idx = np.argmin(cdist(self.Y.value[:, observed], Ynew), axis=0)
        x_init = np.hstack((self.X_mean.value[nearest_idx, :].flatten(),
                            self.X_var.value[nearest_idx, :].flatten()))

        # Objective
        f = self._held_out_data_wrapper_creator(Ynew, observed)

        # Optimize - restrict var to be positive
        result = minimize(fun=f,
                          x0=x_init,
                          jac=True,
                          method=method,
                          tol=tol,
                          bounds = [(None, None)]*int(x_init.size/2) + [(0, None)]*int(x_init.size/2),
                          options=kwargs)
        x_hat = result.x
        mu = x_hat[:infer_number * self.num_latent].reshape((infer_number, self.num_latent))
        var = x_hat[infer_number * self.num_latent:].reshape((infer_number, self.num_latent))

        if return_logprobs:
            return mu, var, -result.fun
        else:
            return mu, var

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_f_density(self, Xstarmu, Xstarvar):
        """
        Predicts the first and second moment of the (non-Gaussian) distribution of the latent function by propagating a
        Gaussian distribution.

        Note: this method is only available in combination with Constant or Zero mean functions.

        :param Xstarmu: mean of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :param Xstarvar: variance of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :returns (mean, covar)
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D
        covar: np.ndarray, size Nnew (number of new points ) x D x D
        """
        assert isinstance(self.mean_function, (Zero, Constant))
        return self.build_predict_density(Xstarmu, Xstarvar)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_y_density(self, Xstarmu, Xstarvar):
        """
        Predicts the first and second moment of the (non-Gaussian) distribution by propagating a
        Gaussian distribution.

        Note: this method is only available in combination with Constant or Zero mean functions.
        """
        assert isinstance(self.mean_function, (Zero, Constant))
        mean, covar = self.build_predict_density(Xstarmu, Xstarvar)
        num_predict = tf.shape(mean)[0]
        num_out = tf.shape(mean)[1]
        noise = tf.tile(tf.expand_dims(self.likelihood.variance * tf.eye(num_out, dtype=float_type), 0), [num_predict, 1, 1])
        return mean, covar+noise

    def predict_f_partial(self, Ynew, observed):
        """
        Given a partial observation, predict the first and second moments of the non-Gaussian distribution over the
        unobserved part of the latent functions:
        .. math::

            p(F^U_* | Y^O_*, X, X_*)

        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions),
        with k <= D.
        :param observed: 1D list or array of indices of observed dimensions, size D-k. Should at least contain one
        observed dimension, and at most all dimensions.
        :returns (mean, covar) of non-Gaussian predictive distribution over the unobserved dimensions
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D-k
        covar: np.ndarray, size Nnew (number of new points ) x D-k x D-k
        """
        observed = np.unique(np.atleast_1d(observed))
        assert(0 < observed.size <= self.output_dim)
        unobserved = np.setdiff1d(np.arange(self.output_dim), observed)
        assert(Ynew.shape[1] == observed.size)

        # obtain q(X*), only consider observed dimensions
        Xstarmu, Xstarvar = self.infer_latent_inputs(Ynew, observed=observed)

        # Perform (full) prediction w.r.t q(X*)
        unobserved_mu, unobserved_covar = self.predict_f_density(Xstarmu, Xstarvar)

        # Return only predictions for unobserved dimensions
        return unobserved_mu[:, unobserved], \
               unobserved_covar[:, unobserved, :][:, :, unobserved]

    def predict_y_partial(self, Ynew, observed):
        """
        Given a partial observation, predict the first and second moments of the non-Gaussian distriubtion over the
        unobserved part:
        .. math::

            p(Y^U_* | Y^O_*, X, X_*)

        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions),
        with k <= D.
        :param observed: 1D list or array of indices of observed dimensions, size D-k. Should at least contain one
        observed dimension, and at most all dimensions.
        :returns (mean, covar) of non-Gaussian predictive distribution over the unobserved dimensions
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D-k
        covar: np.ndarray, size Nnew (number of new points ) x D-k x D-k
        """
        observed = np.unique(np.atleast_1d(observed))
        assert(0 < observed.size <= self.output_dim)
        unobserved = np.setdiff1d(np.arange(self.output_dim), observed)
        assert(Ynew.shape[1] == observed.size)

        # obtain q(X*), only consider observed dimensions
        Xstarmu, Xstarvar = self.infer_latent_inputs(Ynew, observed=observed)

        # Perform (full) prediction w.r.t q(X*)
        unobserved_mu, unobserved_covar = self.predict_y_density(Xstarmu, Xstarvar)

        # Return only predictions for unobserved dimensions
        return unobserved_mu[:, unobserved], \
               unobserved_covar[:, unobserved, :][:, :, unobserved]
