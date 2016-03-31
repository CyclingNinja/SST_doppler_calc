from sunkitsst.read_cubes import read_cubes
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from scipy.special import gammaln
from skimage import img_as_float
from astropy.constants import c

# need to import the _fitter_to_model_params helper function
from astropy.modeling.fitting import _fitter_to_model_params

imfile = '/data_swat/arlimb/crispex.6563.icube'
spfile = '/data_swat/arlimb/crispex.6563.sp.icube'
wave_ind = np.loadtxt('spect_ind.txt')

imheader, icube, spheader, spcube = read_cubes(imfile,spfile)

SST_cad = 2.5
SST_pix = 0.05
l_core = 6562.8


class PoissonlikeDistr(object):
    # Daniela: you actually want to input an astropy model here
    # not a function
    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model
        return

    # Daniela: evaluate needs to take a list of parameters as input
    def evaluate(self, params):
        # Daniela: set the input parameters in the astropy model using the
        # list of parameters in params
        _fitter_from_model_params(self.model, params)

        # Daniela: make the mean model
        mean_model = self.model(x)

        # Daniela: not sure what your 'x' in this log-likelihood is, but it should be
        # the mean model
        loglike = np.sum(-mean_model + self.y*np.log(mean_model) - scipy.special.gammaln(self.y + 1))

        return loglike

    # Daniela: __call__ needs to take self and params as input
    def __call__(self, params):
        # Daniela: __call__ should return the log-likelihood
        return self.evaluate(params)


small_cube = np.array(icube[-2:-1,:, 600:,450:570])

small_cube = img_as_float(small_cube)
dop_arr = np.zeros(small_cube[:, 0, :, :].shape)
param_arr = np.zeros(small_cube[:, 0, :, :].shape)
plt.ioff()
for T in range(small_cube.shape[0]):
    print T
    # define the box to do it in
    for xi in range(small_cube[0].shape[1]):
        for yi in range(small_cube[0].shape[2]):

            # flip the y axis data points
            y = small_cube[T,:, xi, yi]
            ysg = y[:]
            ysg -= np.min(y)
            x = wave_ind

            # single gaussian fit
            # this definitley works (ish)
            ysg = ysg*-1 + np.max(y)
            # Daniela: is there a reason why there are round brackets around the Gaussian model?
            gaus_sing = (models.Gaussian1D(amplitude=np.max(ysg), mean=x[19], stddev=np.std(ysg)))

            # Daniela: instantiate the log-likelihood object;
            # please check whether it uses the right arrays for the data
            loglike_sing = PoissonlikeDistr(x, ysg, gaus_sing)

            # initial parameters
            init_params = [np.max(ysg), x[19], np.std(ysg)]

            # Daniela: for maximum likelihood fitting, we need to define the *negative*
            # log-likelihood:
            neg_loglike_sing = lambda x: -loglike_sing(x)

            # Daniela: here's the optimization:
            opt = scipy.optimize.minimize(neg_loglike_sing, init_params, method="L-BFGS-B", tol=1.e-10)

            # Daniela: print the negative log-likelihood:
            print("The value of the negative log-likelihood: " + str(opt.fun))

            # Daniela: the parameters at the maximum of the likelihood is in opt.x:
            fit_pars = opt.x

            # Daniela : now we can put the parameters back into the Gaussian model
            _fitter_to_model_params(gaus_sing, fit_pars)


            # Bayesian information criterion
            # see also: https://en.wikipedia.org/wiki/Bayesian_information_criterion
            # bic = -2*loglike + n_params * log(n_datapoints)
            # note to myself: opt.fun is -loglike, so we'll just use that here
            bic = 2.*opt.fun + fit_pars.shape[0]*np.log(x.shape[0])

            # Daniela: from here on, you can do the same for the model with two Gaussians
            # Then you can compare the two BICs for a slightly hacky way of model
            # comparison

            fit_g2 = fitting.LevMarLSQFitter()
            g2 = fit_g2(gaus_sing, x, ysg)
            gsg = lambda x: -1 * g2(x)
            ysg = ysg*-1
            t_mean = g2.mean.value

            # revert to an interpolation to find the minima
            # need to keep the regualar orientation of the y dist
            if fit_g2.fit_info['param_cov'] is None:

                ydg = y[:]
                Imax = np.max(ydg)

                g_init = (models.Gaussian1D(amplitude=Imax, mean=x[12], stddev=0.2) +
                         models.Gaussian1D(amplitude=Imax, mean=x[24], stddev=0.2))
                fit_gdg = fitting.LevMarLSQFitter()
                gdg = fit_gdg(g_init, x, ydg)

                res = minimize(gdg, [6562.8], method='L-BFGS-B', bounds=[[x[19 - 5], x[19 + 5]],])
                t_mean = res.x
                if ((t_mean[0] - l_core) > 1) | ((t_mean[0] - l_core) < -1):
                    t_mean = l_core
            dop_arr[T,xi,yi] = t_mean


np.save('/storage2/jet/SST/dopplergram.npy', dop_arr)



# double gaussian fit
# check the fit parameter
#            if fit_g2.fit_info['param_cov'] is None:
#                Imax = np.max(y)
#                g_init = (models.Gaussian1D(amplitude=Imax, mean=x[-12], stddev=1.) +
#                          models.Gaussian1D(amplitude=Imax, mean=x[12], stddev=1.))
#                fit_g = fitting.SLSQPLSQFitter()
#                g2 = fit_g(g_init, x, y)
#                res = minimize(g2, [6562.8], method='L-BFGS-B', bounds=[[x[19 - 7], x[19 + 7]],])
            #t_mean = res.x



## Plot the data with the best-fit model
#plt.figure(figsize=(8,5))
#plt.plot(x, y, 'ko')
#plt.plot(x, g(x), label='Gaussian')
#plt.xlabel('Position')
#plt.ylabel('Flux')
#
#plt.plot(res.x, g(res.x), 'ro')
#plt.show()
