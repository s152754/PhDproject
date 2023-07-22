from re import A
from nutils import cli, export
import numpy, emcee, pandas, typing, treelog
import numpy.random, scipy.stats
import Rheomodels, corner
import glob
import math
from matplotlib import pyplot
from cycler import cycler
import pickle # check voor opslaan string model.param_dist

pyplot.rcParams.update(pyplot.rcParamsDefault)

preamble = r'\usepackage{amsmath}\usepackage{bm}\usepackage{stmaryrd}'
params = {
   'axes.labelsize': 25,
   'axes.linewidth': 0.9,
   'axes.prop_cycle' : cycler(color=['#88CCEE', '#882255', '#332288', '#CC6677', '#999933', '#661100', '#DDCC77', '#44AA99', '#117733', '#AA4499', '#888888']),#light-blue
   'figure.dpi': 120,
   'font.size': 25,
   'legend.fontsize': 25,
   'xtick.labelsize': 25,
   'ytick.labelsize': 25,
   'figure.dpi':300,
   'figure.figsize': [13, 12], # 5.2, 3.9
   'text.usetex': True,
   'text.latex.preamble': preamble
   }
pyplot.rcParams.update(params)

def mustdln(mu, sigma):
    muln = numpy.log(mu**2 / numpy.sqrt(mu**2 + sigma**2))
    stdln = numpy.sqrt( numpy.log(1 + (sigma**2 / mu**2)) )
    return [muln, stdln]

mymodels = {
    'newtonian': Rheomodels.Newtonian(
                                  eta=scipy.stats.norm(loc=mustdln(1.4,0.3)[0], scale=mustdln(1.4,0.3)[1])),
                                  #   eta=scipy.stats.norm(loc=1.4, scale=0.3)),
    'powerlaw': Rheomodels.PowerLaw(
                                K=scipy.stats.norm(loc=1.2, scale=0.2),
                                n=scipy.stats.norm(loc=0.6, scale=0.3)),
    'carreau': Rheomodels.Carreau(
                              λ=scipy.stats.uniform(loc=0.1, scale=0.3),
                              η0=scipy.stats.norm(loc=30, scale=20),
                              n=scipy.stats.uniform(loc=0., scale=1.),
                              ηinf=scipy.stats.norm(loc=0, scale=1)),
    'tpl': Rheomodels.TruncatedPowerLaw(
                              tcrd=scipy.stats.norm(loc=mustdln(0.9,0.3)[0], scale=mustdln(0.9,0.3)[1]),
                              eta0d=scipy.stats.norm(loc=mustdln(111.,30.)[0], scale=mustdln(111.,30.)[1]),
                              n=scipy.stats.beta(a=8, b=3, loc=0., scale=1.),
                              etainfd=scipy.stats.norm(loc=mustdln(0.001,0.0003)[0], scale=mustdln(0.001,0.0003)[1]))
}

def main(model: str, nwalkers: int, nsamples: int, case: str, fname:str, fmodelname:str, ppdmodelname: str, tracename:str):
    '''
    Bayesian inference
    .. arguments::
       model [newtonian]
         Model to be fitted to the data
       nwalkers [4]
         Number of chains
       nsamples [250]
         Number of steps
       case [rheometerGly/]
         define which material
       fname [BI_Rheoparam_PVP_TPLnhat.csv]
         Bayesian inference model parameters filename
       fmodelname [Rheoout_PVP_TPLnhat.csv]
         mu and sigma of eta(gammadot)
       ppdmodelname [ppd_Rheoout_PVP_TPLnhat.csv]
         posterior predictive distribution
       tracename [Trace_Rheoparam_PVP_TPLnhat]
         trace data per model parameter
    '''

    ############################
    # model definition         #
    ############################

    model = mymodels[model]

    # with open("priordist_Rheo_TPL.txt", "wb") as fp:
    #     pickle.dump(model.param_dists,fp)
    #
    # with open("testparamdist.txt", "rb") as fp:
    #     b = pickle.load(fp)

    ############################
    # experimental data        #
    ############################
    all_files = glob.glob(case + "*.csv")
    γpoints = []
    ηpoints = []
    for filename in all_files:
        datafile = pandas.read_csv(filename)
        γpointstemp = datafile['rate'].to_numpy()
        ηpointstemp = datafile['eta'].to_numpy()

        γpoints.append(γpointstemp)
        ηpoints.append(ηpointstemp)

    γpoints = numpy.array(γpoints).T
    ηpoints = numpy.array(ηpoints).T
    npoints = ηpoints.size
    # treelog.user("gamma \n",γpoints)
    # treelog.user("eta \n",ηpoints)

    ############################
    # sampling                 #
    ############################

    # start positions
    θstarts = model.sample_prior(nwalkers)

    # sampler configuration
    sampler = emcee.EnsembleSampler(nwalkers, model.nparams, model.log_probability, args=(γpoints, ηpoints))
    # sampling
    sampler.run_mcmc(θstarts, nsamples, progress=True)
    with treelog.context('post processing'):

        samples = sampler.get_chain()

        with export.mplfigure('trace.png') as fig:
            axes = fig.subplots(model.nparams, sharex=True)
            axes = axes if model.nparams > 1 else [axes]
            for i in range(model.nparams):
                ax = axes[i]
                ax.plot(samples[:, :, i], alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(model.param_names[i]) #
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("$\mathrm{step number}$")

        # print the auto-correlation
        acts = sampler.get_autocorr_time(quiet=True)
        for param_name, act in zip(model.param_names, acts):
            treelog.user('auto-correlation ({}): {}'.format(param_name, act))

        treelog.user("acts \n",acts)
        # burn-in and thinning
        act_max = int(numpy.max(acts))
        flat_samples = sampler.get_chain(discard=2 * act_max, thin=act_max // 2, flat=True)
        bins = 30
        binsprior = 20
        nsamples = flat_samples.shape[0]

        # show correlation via corner plot
        # treelog.user(flat_samples)
        with export.mplfigure('corner.pdf') as fig:
            fig = corner.corner(flat_samples, bins=bins, fig=fig, smooth=True, labels=['$\hat{\\tau}_{rc}$','$\hat{\eta}_0$','$n$','$\hat{\eta}_\infty$'], max_n_ticks=4, quantiles=[0.05, 0.95], verbose=True) #labels=[r'$\eta \ [\mathrm{Pa} \cdot \mathrm{s}]$'],
            for i, dist in enumerate(model.param_dists):
                ax = fig.axes[i * model.nparams + i]
                params = numpy.linspace(*ax.get_xlim(), binsprior)
                params_space = params[1] - params[0]
                space_prob = dist.cdf(params + 0.5 * params_space) - dist.cdf(params - 0.5 * params_space)
                probs = nsamples * (binsprior / bins) * space_prob
                ax.plot(params, probs, color='#882255', ls='-')

        # # experiment fit
        nsamples_plot = nsamples
        γplot = numpy.mean(γpoints, axis=1)
        nplot = len(γplot)

        # ppd param #
        mudraw = 0.  # mean should be similar for posterior predictive and posterior
        stddraw = numpy.nanstd(ηpoints, axis=1)  # noise in exp data
        ηmodelppd = numpy.empty_like(stddraw)
        with export.mplfigure('fit.png') as fig:
            ax = fig.subplots(1)
            inds = numpy.random.randint(len(flat_samples), size=nsamples_plot)
            data = numpy.zeros(shape=(nplot, nsamples_plot))
            datappd = numpy.zeros(shape=(nplot, nsamples_plot))
            for ind, dat, datppd in zip(inds, data.T, datappd.T):
                theta = flat_samples[ind]
                ηmodel = model.η(theta, γplot)
                for i in range(len(γplot)):
                    # poster predictive distribution samples #
                    draw = scipy.stats.multivariate_normal.rvs(mudraw, stddraw[i]**2)
                    ηmodelppd[i] = ηmodel[i]+draw
                dat[:] = ηmodel
                datppd[:] = ηmodelppd
                ax.plot(γplot, ηmodel, "C1", alpha=0.01)
                ax.plot(γplot, ηmodelppd, "#888888", alpha=0.01)
            ax.errorbar(γplot, numpy.mean(data, axis=-1), 2 * numpy.std(data, axis=-1), color='black', label=r'$\mu \, \pm \, 2\sigma$', fmt='o', markersize=3, capsize=4)
            ax.errorbar(γplot, numpy.mean(ηpoints,axis=1), 2 * numpy.std(ηpoints,axis=1), color='blue', label="data")
            ax.errorbar(γplot, numpy.mean(datappd, axis=-1), 2 * numpy.std(datappd, axis=-1), color='black', label=r'$\mu \, \pm \, 2\sigma$', fmt='o', markersize=3, capsize=4)
            ax.legend()
            ax.set_xscale('log')
            ax.set_ylabel(r'$\eta [\mathrm{Pa} \cdot \mathrm{s}]$')
            ax.set_xlabel(r'$\dot{\gamma} [1/\mathrm{s}]$')
            ax.grid()

        # ##### saving data #####
        #
        # # model response + credibility
        # mean_model = pandas.DataFrame({'rate': γplot, 'etap005':numpy.quantile(data,0.05,axis=-1), 'etap05':numpy.quantile(data,0.5,axis=-1), 'etap095':numpy.quantile(data,0.95,axis=-1)})
        # mean_model.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/' + 'w8N20000'+fmodelname)
        #
        # # posterior samples model input parameters
        # data_samples = pandas.DataFrame({v: flat_samples[:, ind] for ind, v in enumerate(model.param_names)})
        # data_samples.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/ParametricUncertaintyFiles/' + 'w8N20000' + fname)
        #
        # # Posterior predictive distribution (mu + 2sigma)
        # mean_ppdmodel = pandas.DataFrame({'time': γplot, 'muppdRm': numpy.nanmean(datappd, axis=-1), 'stdppdRm': numpy.nanstd(datappd, axis=-1)})
        # mean_ppdmodel.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/' + 'w8N20000' + ppdmodelname)
        #
        # # x number of chains in n samples for model input parameters
        # for i in range(model.nparams):
        #     trace_data = pandas.DataFrame(samples[:, :, i])
        #     trace_data.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/ParametricUncertaintyFiles/' + tracename + model.param_names[i] + 'w8N20000' + '.csv')

        #######################
cli.run(main)
