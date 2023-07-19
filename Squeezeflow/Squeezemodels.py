import scipy.stats
from nutils import cli, types, export
import treelog, numpy, pandas
import TruncatedPowerLawCLEMENS2
#####################
# implement units   #
#####################
unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s',
                  hr='60min')
_ = numpy.newaxis

####################
# model base class #
####################

class Model:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.param_dists = list(self.params.values())
        self.param_names = list(self.params.keys())
        self.nparams = len(self.param_names)
        self.paramNan = []

    def get_nan_params(self):
        if len(self.paramNan) == 0:
            return

        param_table = numpy.array(self.paramNan)
        assert self.nparams == param_table.shape[1]

        nan_dict = {self.param_names[col]: param_table[:, col] for col in range(self.nparams)}

        return pandas.DataFrame(nan_dict)

    def sample_prior(self, size):
        return numpy.array([dist.rvs(size) for dist in self.param_dists]).T

    def log_prior(self, params):
        return sum(dist.logpdf(param) for param, dist in zip(params,self.param_dists))

    def prior(self, params):
        return numpy.array([dist.pdf(param) for param, dist in zip(params,self.param_dists)])

    # likelihood used for experimental data
    def log_likelihood(self, params, texp, Rexp):
        mutexp = numpy.nanmean(texp, axis=1)
        sigmatexp = numpy.nanstd(texp, axis=1)
        assert (sigmatexp * sigmatexp).sum() < 1e-5 * (mutexp * mutexp).sum()
        muRexp = numpy.nanmean(Rexp, axis=1)
        sigmaRexp = numpy.nanstd(Rexp, axis=1)  # moet dit nu 1 of 2*sigma zijn. Alleen met 2*sigma neemt ie het hele gebied mee (in eerste instantie = 1)
        Rmodel, dpdrt0, dpdrt1, dpdrt2, rt0, rt1, rt2 = self.R(params, mutexp)

        if (Rmodel is None) or numpy.isnan(Rmodel).sum() > 0:
            self.paramNan.append(params)
            return -numpy.inf

        return scipy.stats.multivariate_normal.logpdf(Rmodel, muRexp, sigmaRexp**2)

    def log_probability(self, params, texp, Rexp):
        return self.log_prior(params)+self.log_likelihood(params, texp, Rexp)

####################
# model classes    #
####################

class Newtonian(Model):
    def R(self, params, t):
        F, Vd, R0d, etad = params

        V   = numpy.exp(Vd)
        R0  = numpy.exp(R0d)
        eta = numpy.exp(etad)

        h0      = 0.5 * ( V / (R0**2 * numpy.pi) )
        h_ana   = h0 * (1 + (8 * F * (2 * h0) ** 2) / (3 * numpy.pi * eta * (R0 ** 4)) * t) ** (-1 / 4)
        return R0 * numpy.sqrt(h0 / h_ana)



class NewtonianLaplace(Model):
    def R(self,params,texp):
        F, Vd, R0d, etad, gamma, alpha = params
        interptimes = texp

        V = numpy.exp(Vd)
        R0 = numpy.exp(R0d)
        eta = numpy.exp(etad)

        h0 = 0.5 * (V / (numpy.pi * R0 ** 2))
        told = 0
        T = 350.
        dtmax = 1.
        s=0.01
        dt = abs(s * (3 * numpy.pi * eta * R0 ** 4) / (8 * F * h0 ** 2))
        hold = h0

        h = []
        R = []
        t = []

        h.append(h0)
        R.append(R0)
        t.append(told)

        while told < T:
            hdot = - (32 * numpy.pi * (hold ** 5 * F - ( gamma * alpha / hold) * V * hold ** 4)) / ( 3 * eta * V ** 2)  # kappa = 1 / Ri = 1 / (alpha * h)

            hnew = hold + hdot*dt
            Rnew = R0 * numpy.sqrt(h0 / hnew)
            tnew = told + dt


            # df = pandas.concat([df, pandas.DataFrame({'t': [tnew], 'h': [hnew], 'R': [Rnew]})], ignore_index=True)
            h.append(hnew)
            R.append(Rnew)
            t.append(told)

            dtprop = dt*1.5
            dt = min(dtprop,dtmax) # find more appropriate adaptive time stepper


            hold = hnew
            told = tnew

        Rinterp = numpy.interp(interptimes, t, R)

        return Rinterp



class PowerLaw(Model):
    def R(self, params, t):
        return

class TruncatedPowerLaw(Model):
    def R(self, params, t):

        T = 350.
        F, Vd, R0d, mu0d, muinfd, nd, tcrd, gamma, alpha = params

        interptimes = t

        V       = numpy.exp(Vd)
        R0      = numpy.exp(R0d)
        mu0     = numpy.exp(mu0d)
        muinf   = numpy.exp(muinfd)
        n       = numpy.exp(nd)
        tcr     = numpy.exp(tcrd)
        # treelog.user("What are the input parameters? \n", F, V, R0, mu0, muinf, n, tcr, gamma, alpha)

        h0 = 0.5 * (V / (numpy.pi * R0**2))
        m = 1000
        npicard = 500
        ntarget = 20
        tol = 1e-5
        s = 1e-3
        Dtmax = 1.

        # # # solve check om df heen voor het aantal iteraties
        relax = 0
        ntarget = 20
        convergence = None
        while convergence == None:
            try:
                df, timelist, dpdrlist, rlist = TruncatedPowerLawCLEMENS2.main(mu0=mu0, muinf=muinf, tcr=tcr, n=n, R0=R0, h0=h0, F=F, T=T, m=m, npicard=npicard, ntarget=ntarget, tol=tol, s=s, Dtmax=Dtmax, relax=relax, gamma=gamma, alpha=alpha)
                dpdrfun = scipy.interpolate.interp1d(numpy.array(timelist), numpy.array(dpdrlist), axis=0)
                rfun = scipy.interpolate.interp1d(numpy.array(timelist), numpy.array(rlist), axis=0)
                dpdrt0 = dpdrfun(2.e-2)
                dpdrt1 = dpdrfun(1.5)
                dpdrt2 = dpdrfun(10.)
                rt0 = rfun(2.e-2)
                rt1 = rfun(1.5)
                rt2 = rfun(10.)
                convergence = True
            except:
                if relax < 0.8:
                    relax += 0.1
                    ntarget = 20 / (1 - relax)

                else:
                    df = pandas.DataFrame({'t': [], 'h': [], 'R': []})
                    timelist = []
                    dpdrlist = []
                    rlist = []
                    dpdrfun = scipy.interpolate.interp1d(numpy.array(timelist), numpy.array(dpdrlist), axis=0)
                    rfun = scipy.interpolate.interp1d(numpy.array(timelist), numpy.array(rlist), axis=0)
                    dpdrt0 = dpdrfun(2.e-2)
                    dpdrt1 = dpdrfun(1.5)
                    dpdrt2 = dpdrfun(10.)
                    rt0 = rfun(2.e-2)
                    rt1 = rfun(1.5)
                    rt2 = rfun(10.)

                    convergence = False

            if not convergence:
                return None, None, None, None, None, None, None


        # df = TruncatedPowerLawCLEMENS2.main(mu0=mu0, muinf=muinf, tcr=tcr, n=n, R0=R0, h0=h0, F=F, T=T, m=m, npicard=npicard, ntarget=ntarget, tol=tol, s=s, Dtmax=Dtmax, relax=relax, gamma=gamma, alpha=alpha)
        Rtot = df.R.to_numpy()
        ttot = df.t.to_numpy()
        Rinterp = numpy.interp(interptimes, ttot, Rtot)

        return Rinterp, dpdrt0, dpdrt1, dpdrt2, rt0, rt1, rt2








