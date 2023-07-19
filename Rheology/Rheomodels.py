import numpy, scipy.stats
import math
import treelog

####################
# model base class #
####################

class Model:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.param_dists = list(self.params.values())
        self.param_names = list(self.params.keys())
        self.nparams = len(self.param_names)

    def sample_prior(self, size):
        return numpy.array([dist.rvs(size) for dist in self.param_dists]).T

    def log_prior(self, params):
        return sum(dist.logpdf(param) for param, dist in zip(params,self.param_dists))

    def prior(self, params):
        return numpy.array([dist.pdf(param) for param, dist in zip(params,self.param_dists)])

    # likelihood used for experimental data
    def log_likelihood(self, params, γdata, ηdata):
       mugammadata = numpy.mean(γdata, axis=1)
       sigmagammadata = numpy.std(γdata, axis=1)
       assert (sigmagammadata*sigmagammadata).sum() < 1e-5*(mugammadata*mugammadata).sum()
       muetadata = numpy.mean(ηdata,axis=1)
       sigmaetadata = numpy.std(ηdata,axis=1) ###### moet dit nu 1 of 2*sigma zijn. Alleen met 2*sigma neemt ie het hele gebied mee
       ηmodel  = self.η(params, mugammadata)
       # treelog.user(ηmodel)

       if numpy.isnan(ηmodel).sum() > 0:
           return -numpy.inf

       return scipy.stats.multivariate_normal.logpdf(ηmodel,muetadata,sigmaetadata**2)
       # return scipy.stats.multivariate_normal.logpdf(ηmodel, muetadata, sigmaetadata ** 2) # zou dit fout geweest zijn?

    # # likelihood used for virtual data
    # def log_likelihood(self, params, γdata, ηdata):
    #     ηmodel  = self.η(params, γdata)
    #     σnoise  = self.Vnoise*numpy.sqrt(γdata)
    #     return scipy.stats.multivariate_normal.logpdf(ηmodel,ηdata,σnoise)

    def log_probability(self, params, γdata, ηdata):
        return self.log_prior(params)+self.log_likelihood(params, γdata, ηdata)


####################
# model classes    #
####################

class Newtonian(Model):
    def η(self, params, γ):
        eta0d = params
        eta0 = numpy.exp(eta0d) # when in need of log transfrom uncomment this line
        return eta0*numpy.ones_like(γ)

class PowerLaw(Model):
    def η(self, params, γ):
        K, n = params
        return K*(γ**(n-1.0))

class Carreau(Model):
    def η(self, params, γ):
        λ, η0, n, ηinf = params
        return ηinf + (η0-ηinf) / ((1+(λ*γ)**2.0)**(0.5-n/2.0))

class TruncatedPowerLaw(Model):
    def η(self, params, γ):
        tcr, eta0, n, etainf = params

        # construct the exp (later in function) #
        tcr = math.exp(tcr)
        etainf = math.exp(etainf)
        eta0 = math.exp(eta0)
        #########################################

        K = eta0 * tcr**(n-1)
        gdot1 = (eta0 / K)**(1/(n-1))
        gdot2 = (etainf / K)**(1/(n-1))
        eta = numpy.zeros_like(γ)
        for i in range(γ.shape[0]):
            if γ[i] <= gdot1:
                eta[i] = eta0
            elif γ[i] > gdot1 and γ[i] < gdot2:
                eta[i] = eta0 * tcr**(n-1) * γ[i]**(n-1)
            elif γ[i] >= gdot2:
                eta[i] = etainf
        return eta


