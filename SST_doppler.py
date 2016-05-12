from sunkitsst.read_cubes import read_cubes
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from scipy.optimize import minimize
#from scipy.signal import argrelextrema
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
SST_pix = 0.059
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
        _fitter_to_model_params(self.model, params)

        # Daniela: make the mean model
        mean_model = self.model(x)

        # Daniela: not sure what your 'x' in this log-likelihood is, but it should be
        # the mean model
        loglike = np.sum(-mean_model + self.y*np.log(mean_model) - gammaln(self.y + 1))

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

            # SINGLE GAUSSIAN FITTING
            # this definitley works (ish)
            ysg = ysg*-1 + np.max(y)
            # Daniela: is there a reason why there are round brackets around the Gaussian model?
            gaus_sing = (models.Gaussian1D(amplitude=np.max(ysg), mean=x[19], stddev=np.std(ysg)))

            # Daniela: instantiate the log-likelihood object;
            # please check whether it uses the right arrays for the data
            loglike_sing = PoissonlikeDistr(x, ysg, gaus_sing)

            # initial parameters
            init_params_s = [np.max(ysg), x[19], np.std(ysg)]

            # Daniela: for maximum likelihood fitting, we need to define the *negative*
            # log-likelihood:
            neg_loglike_sing = lambda x: -loglike_sing(x)

            # Daniela: here's the optimization:
            opt_sing = minimize(neg_loglike_sing, init_params_s,
                                method="L-BFGS-B", tol=1.e-10)

            # Daniela: print the negative log-likelihood:
            #print("The value of the negative log-likelihood: " + str(opt_sing.fun))

            # Daniela: the parameters at the maximum of the likelihood is in opt.x:
            fit_pars = opt_sing.x

            # Daniela : now we can put the parameters back into the Gaussian model
            _fitter_to_model_params(gaus_sing, fit_pars)


            # Bayesian information criterion
            # see also: https://en.wikipedia.org/wiki/Bayesian_information_criterion
            # bic = -2*loglike + n_params * log(n_datapoints)
            # note to myself: opt.fun is -loglike, so we'll just use that here
            bic_sing = 2.*opt_sing.fun + fit_pars.shape[0]*np.log(x.shape[0])

            # Daniela: from here on, you can do the same for the model with two Gaussians
            # Then you can compare the two BICs for a slightly hacky way of model
            # comparison


            # DOUBLE GAUSSIAN FITTING
            ydg = y[:]
            Imax = np.max(ydg)
            gaus_double = (models.Gaussian1D(amplitude=Imax, mean=x[12], stddev=0.2) +
                         models.Gaussian1D(amplitude=Imax, mean=x[24], stddev=0.2))

            init_params_double = [np.max(ydg), x[12], np.std(ydg),
                                  np.max(ydg), x[24], np.std(ydg)]


            loglike_double = PoissonlikeDistr(x, ysg, gaus_double)
            neg_loglike_doub = lambda x: -loglike_double(x)

            opt_doub = minimize(neg_loglike_doub, init_params_double,
                                method="L-BFGS-B", tol=1.e-10)

            loglike_doub = PoissonlikeDistr(x, ydg, gaus_double)

            fit_pars_dg = opt_doub.x

            _fitter_to_model_params(gaus_double, fit_pars_dg)

            bic_doub = 2.*opt_doub.fun + fit_pars.shape[0]*np.log(x.shape[0])

             # use the bic values to assign to fit again and calc the doppler array
            if bic_doub < bic_sing:
                fit_sing_g_2 = fitting.LevMarLSQFitter()
                gs2 = fit_sing_g_2(gaus_sing, x, ysg)
                gsg = lambda x: -1 * gs2(x)
                ysg = ysg*-1
                t_mean = gs2.mean.value
            else:
                fit_doub_g_2 = fitting.LevMarLSQFitter()
                ydg = y[:]
                gd2 = fit_doub_g_2(gaus_double, x, ydg)
                res = minimize(gd2, [6562.8], method='L-BFGS-B', bounds=[[x[19 - 5], x[19 + 5]],])
                t_mean = res.x

            dop_arr[xi,yi] = t_mean
    np.save(dop_arr, '/storage2/jet/dop_arrs/dop_arr_{:03d}'.format(T))


#            # revert to an interpolation to find the minima
#            # need to keep the regualar orientation of the y dist
#            if fit_g2.fit_info['param_cov'] is None:
#
#                ydg = y[:]
#                Imax = np.max(ydg)
#
#                g_init = (models.Gaussian1D(amplitude=Imax, mean=x[12], stddev=0.2) +
#                         models.Gaussian1D(amplitude=Imax, mean=x[24], stddev=0.2))
#                fit_gdg = fitting.LevMarLSQFitter()
#                gdg = fit_gdg(g_init, x, ydg)
#
#                res = minimize(gdg, [6562.8], method='L-BFGS-B', bounds=[[x[19 - 5], x[19 + 5]],])
#                t_mean = res.x
#                if ((t_mean[0] - l_core) > 1) | ((t_mean[0] - l_core) < -1):
#                    t_mean = l_core
#            dop_arr[T,xi,yi] = t_mean


#np.save('/storage2/jet/SST/dopplergram.npy', dop_arr)






## Plot the data with the best-fit model
#plt.figure(figsize=(8,5))
#plt.plot(x, y, 'ko')
#plt.plot(x, g(x), label='Gaussian')
#plt.xlabel('Position')
#plt.ylabel('Flux')
#
#plt.plot(res.x, g(res.x), 'ro')
#plt.show()
