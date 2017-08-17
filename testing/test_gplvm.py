from __future__ import print_function
import GPflow
import numpy as np
import unittest
from GPflow import ekernels

import lvmzoo

np.random.seed(0)


class TestGPLVM(unittest.TestCase):
    def setUp(self):
        # data
        self.N = 20  # number of data points
        self.D = 5  # data dimension
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.N, self.D)
        # model
        self.M = 10  # inducing points

    def test_2d(self):
        # test default Z on 2_D example
        Q = 2  # latent dimensions
        X_mean = GPflow.gplvm.PCA_reduce(self.Y, Q)
        k = ekernels.RBF(Q, ARD=False)
        m = lvmzoo.gplvm.GPLVM(X_mean=X_mean, X_var=np.ones((self.N, Q)), Y=self.Y, kern=k, M=self.M)

        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)

        # test prediction
        Xtest = self.rng.randn(10, Q)
        mu_f, var_f = m.predict_f(Xtest)
        mu_fFull, var_fFull = m.predict_f_full_cov(Xtest)
        self.assertTrue(np.allclose(mu_fFull, mu_f))
        # check full covariance diagonal
        for i in range(self.D):
            self.assertTrue(np.allclose(var_f[:, i], np.diag(var_fFull[:, :, i])))

        # inverse prediction
        m.optimize()
        mu_inferred, var_inferred = m.infer_latent_inputs(self.Y)
        self.assertTupleEqual(mu_inferred.shape, (self.N, Q))
        self.assertTupleEqual(var_inferred.shape, (self.N, Q))
        self.assertTrue(np.allclose(m.X_mean.value, mu_inferred, atol=1e-3))
        self.assertTrue(np.allclose(m.X_var.value, var_inferred, atol=1e-3))
        mu_inferred, var_inferred, prob = m.infer_latent_inputs(np.atleast_2d(self.Y[0, :]), return_logprobs=True)
        self.assertLess(abs(prob - m.compute_log_likelihood()), 15)

        # predict with input densities: no variance -> mean equals predict_f
        uncertain_mu, uncertain_covar = m.predict_f_density(m.X_mean.value, np.zeros((self.N, Q)))
        certain_mu, certain_covar = m.predict_f(m.X_mean.value)
        self.assertTrue(np.allclose(uncertain_mu, certain_mu))
        self.assertTrue(np.allclose(uncertain_covar[:, np.arange(self.D), np.arange(self.D)], certain_covar))

        uncertain_mu, uncertain_covar = m.predict_y_density(m.X_mean.value, np.zeros((self.N, Q)))
        certain_mu, certain_covar = m.predict_y(m.X_mean.value)
        self.assertTrue(np.allclose(uncertain_mu, certain_mu))
        self.assertTrue(np.allclose(uncertain_covar[:, np.arange(self.D), np.arange(self.D)], certain_covar))

        # Partial prediction
        observed = [0, 2, 4]
        mu_predict_f, var_predict_f = m.predict_f_partial(self.Y[:, observed], observed=observed)
        self.assertTupleEqual(mu_predict_f.shape, (self.N, 2))
        self.assertTupleEqual(var_predict_f.shape, (self.N, 2, 2))

        mu_predict_y, var_predict_y = m.predict_y_partial(self.Y[:, observed], observed=observed)
        self.assertTrue(np.allclose(mu_predict_f, mu_predict_y))
        diagonals = (var_predict_y - var_predict_f)[:, [0, 1], [0, 1]]
        self.assertTrue(np.allclose(diagonals, m.likelihood.variance.value))


if __name__ == "__main__":
    unittest.main()
