from nutils import cli, export
import numpy, emcee, pandas, typing, treelog
import numpy.random, scipy.stats
import corner
import glob
import math
import Squeezemodels
from matplotlib import pyplot
from cycler import cycler
import pickle

pyplot.rcParams.update(pyplot.rcParamsDefault)

preamble = r'\usepackage{amsmath}\usepackage{bm}\usepackage{stmaryrd}'
params = {
   'axes.labelsize': 50,
   'axes.linewidth': 0.9,
   'axes.prop_cycle' : cycler(color=['#332288','#AA4499','#44AA99','#999933','#88CCEE','#CC6677','#DDCC77','#117733','#888888','#6699CC','#661100','#882255']),
   'figure.dpi': 120,
   'font.size': 50,
   'legend.fontsize': 50,
   'xtick.labelsize': 50,
   'ytick.labelsize': 50,
   'figure.dpi':300,
   'figure.figsize': [40.0, 35.0],#[5.2*1.5, 3.9*1.8], # 5.2, 3.9
   'text.usetex': True,
   'text.latex.preamble': preamble
   }
pyplot.rcParams.update(params)

def mustdln(mu, sigma):
    muln = math.log(mu**2 / math.sqrt(mu**2 + sigma**2))
    stdln = math.sqrt(math.log(1 + sigma**2 / mu**2))
    return [muln, stdln]


mymodels = {
    'newtonian' : Squeezemodels.Newtonian(
                                                    F=scipy.stats.norm(loc=2.81, scale=1.01e-1),
                                                    V=scipy.stats.norm(loc=mustdln(2.04e-7, 2.22e-8)[0], scale=mustdln(2.04e-7, 2.22e-8)[1]),
                                                    R0=scipy.stats.norm(loc=mustdln(7.55e-3, 1.50e-4)[0], scale=mustdln(7.55e-3, 1.50e-4)[1]),
                                                    eta=scipy.stats.norm(loc=mustdln(8.67e-1, 1.22e-2)[0], scale=mustdln(8.67e-1, 1.22e-2)[1]),
                                                    gamma = scipy.stats.norm(loc=4.5e-2, scale=2.5e-3),
                                                    alpha = scipy.stats.beta(a=3, b=8, loc=0, scale=1)), # 1 / half the height of the fluid layer(0.5e-3)
    'truncatedpowerlaw' : Squeezemodels.TruncatedPowerLaw(
                                                    F=scipy.stats.norm(loc=8.48, scale=8.20e-2),
                                                    V=scipy.stats.norm(loc=mustdln(1.59e-7, 2.24e-8)[0], scale=mustdln(1.59e-7, 2.24e-8)[1]),
                                                    R0=scipy.stats.norm(loc=mustdln(6.38e-3, 4.66e-4)[0], scale=mustdln(6.38e-3, 4.66e-4)[1]),
                                                    eta0=scipy.stats.norm(loc=mustdln(3.20e1, 1.34e-1)[0], scale=mustdln(3.20e1, 1.34e-1)[1]),
                                                    etainf=scipy.stats.norm(loc=mustdln(1.01e-3, 3.04e-4)[0], scale=mustdln(1.01e-3, 3.04e-4)[1]),
                                                    n=scipy.stats.norm(loc=mustdln(7.71e-1, 4.72e-3)[0], scale=mustdln(7.71e-1, 4.72e-3)[1]),
                                                    tcr=scipy.stats.norm(loc=mustdln(7.11e-1, 3.33e-2)[0], scale=mustdln(7.11e-1, 3.33e-2)[1]),
                                                    gamma=scipy.stats.norm(loc=6.6e-2, scale=2.e-3),
                                                    alpha=scipy.stats.beta(a=3, b=8, loc=0, scale=1))  # 1 / half the height of the fluid layer(0.5e-3)
}

def main(model: str, nwalkers: int, nsamples: int, datafile: str, fname: str, fmodelname: str, ppdmodelname: str, tracename: str):
    '''
        Bayesian inference
        .. arguments::
           model [truncatedpowerlaw]
             Model to be fitted to the data
           nwalkers [27]
             Number of chains
           nsamples [10]
             Number of steps
           datafile [RoutExp_PVPF2V2.csv]
             Experiment case file
           fname [BI_Rmparam_F2V2_PVP_TPLLP.csv]
             Bayesian inference model parameters filename
           fmodelname [Rmout_F2V2_PVP_TPLLP.csv]
             output file with radius over time
           ppdmodelname [ppdRmout_F2V2_PVP_TPLLP.csv]
             outfile file with posterior predictive distrubution of radius over time
           tracename [Trace_RMparam_F2V2_PVP_TPLLP_]
             trace data per model parameter
    '''

    ############################
    # model definition         #
    ############################

    model = mymodels[model]
    with open("priordist_squeezePVP_F2V2_TPLLP.txt", "wb") as fp:
        pickle.dump(model.param_dists,fp)

    ############################
    # experimental data        #
    ############################
    path        = "Experiments/RoutExp/"
    datafile    = pandas.read_csv(path + datafile)
    expnum      = 10
    name        = []
    namet       = []
    for i in range(expnum):
        name.append('Radius_'+str(i+1))
        namet.append('Time_'+str(i+1))

    Rexp  = datafile[name].to_numpy()
    texp    = datafile[namet].to_numpy()

    # # adjust experimental array to maximum time T
    # T = 30.
    # texp = []
    # for i in range(len(texptot)):
    #     if texptot[i] < T:
    #         texp.append(texptot[i])
    #     else:
    #         break
    # Rexp = numpy.interp(texp,texptot,Rexptot)

    # meanRexp   = datafile['meanRadius'].to_numpy()
    # stdRexp    = datafile['stdRadius'].to_numpy()

    ############################
    # sampling                 #
    ############################

    # start positions
    θstarts = model.sample_prior(nwalkers)
    # treelog.user(θstarts)

    # sampler configuration
    sampler = emcee.EnsembleSampler(nwalkers, model.nparams, model.log_probability, args=(texp, Rexp))

    # sampler = emcee.EnsembleSampler(nwalkers, model.nparams, model.log_probability, args=(γpoints, ηpoints, len(all_files) // 1))
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
                ax.set_ylabel(model.param_names[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")


        iterations = numpy.arange(10,samples.shape[0], samples.shape[0]//25, dtype=int)
        moving_avg = numpy.empty(shape=samples[iterations,:,:].shape)
        moving_std = numpy.empty(shape=samples[iterations,:,:].shape)
        for iteration,val in enumerate(iterations):
            moving_avg[iteration,:,:] = numpy.mean(samples[:iteration,:,:], axis=0)
            moving_std[iteration,:,:] = numpy.std(samples[:iteration,:,:], axis=0)
            # treelog.user("what is the mean?\n",moving_avg)

        with export.mplfigure('moments.png') as fig:
            axes = fig.subplots(model.nparams, sharex=True)
            axes = axes if model.nparams > 1 else [axes]
            for i in range(model.nparams):
                ax = axes[i]
                for w in range(moving_avg.shape[1]):
                    ax.errorbar(iterations, moving_avg[:,w,i], yerr=moving_std[:,w,i])
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(model.param_names[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")



        # print the auto-correlation
        acts = sampler.get_autocorr_time(quiet=True)
        for param_name, act in zip(model.param_names, acts):
            treelog.user('auto-correlation ({}): {}'.format(param_name, act))

        # burn-in and thinning
        act_max = int(numpy.max(acts))
        flat_samples = sampler.get_chain(discard=2 * act_max, thin=act_max // 2, flat=True)
        bins = 30
        binsprior = 15
        nsamples = flat_samples.shape[0]

        with export.mplfigure('corner.pdf') as fig:
            fig = corner.corner(flat_samples, bins=bins, fig=fig, smooth=True, labels=[r'$\mathrm{F} \ [\mathrm{N}]$',r'$\hat{\mathrm{V}} \ [\mathrm{m}^3]$',r'$\hat{\mathrm{R}}_0 \ [\mathrm{m}]$',r'$\hat{\eta} \ [\mathrm{Pa} \cdot \mathrm{s}]$', r'$\gamma \ [\mathrm{N}/\mathrm{m}]$', r'$\alpha \ [-]$'], max_n_ticks=4, quantiles=[0.05, 0.95], verbose=True, color='#000000') #labels=model.param_names
            for i, dist in enumerate(model.param_dists):
                ax = fig.axes[i * model.nparams + i]
                params = numpy.linspace(*ax.get_xlim(), binsprior)
                params_space = params[1] - params[0]
                space_prob = dist.cdf(params + 0.5 * params_space) - dist.cdf(params - 0.5 * params_space)
                probs = nsamples * (binsprior / bins) * space_prob
                ax.plot(params, probs, color='#117733')

        # experiment fit
        nsamples_plot = nsamples
        # tplot = numpy.linspace(numpy.min(texp), numpy.max(texp), nplot) #tplot = numpy.geomspace(numpy.min(texp), numpy.max(texp), nplot)
        tplot = numpy.mean(texp, axis=1)
        nplot = len(tplot)  # =20

        # ppd param #
        mudraw = 0. # mean should be similar for posterior predictive and posterior
        stddraw = numpy.nanstd(Rexp, axis=1) # noise in exp data
        Rmodelppd = numpy.empty_like(stddraw)

        ################ w1, w2, v ################################################
        # m = 1000 # number of elements, check with BIModel script
        # ndpdrt0 = numpy.zeros(shape=(nsamples,m))
        # ndpdrt1 = numpy.zeros_like(ndpdrt0)
        # ndpdrt2 = numpy.zeros_like(ndpdrt0)
        # nrt0 = numpy.zeros_like(ndpdrt0)
        # nrt1 = numpy.zeros_like(ndpdrt0)
        # nrt2 = numpy.zeros_like(ndpdrt0)
        ###########################################################################

        with export.mplfigure('fit.png') as fig:
            ax = fig.subplots(1)
            inds = numpy.random.randint(len(flat_samples), size=nsamples_plot)
            data = numpy.zeros(shape=(nplot, nsamples_plot))
            datappd = numpy.zeros(shape=(nplot, nsamples_plot))
            for ind, dat, datppd in zip(inds, data.T, datappd.T):
                theta = flat_samples[ind]
                # treelog.user("theta: \n", theta)
                # Rmodel, ndpdrt0[ind,:], ndpdrt1[ind,:], ndpdrt2[ind,:], nrt0[ind,:], nrt1[ind,:], nrt2[ind,:] = model.R(theta, tplot)
                Rmodel = model.R(theta, tplot)
                for i in range(len(tplot)):
                    # poster predictive distribution samples #
                    draw = scipy.stats.multivariate_normal.rvs(mudraw, stddraw[i]**2)
                    Rmodelppd[i] = Rmodel[i]+draw
                ##########################################
                dat[:] = Rmodel
                datppd[:] = Rmodelppd
                ax.plot(tplot, Rmodel, "C1", alpha=0.01)
                ax.plot(tplot, Rmodelppd, "#888888", alpha=0.01)
            ax.plot(texp, Rexp,"bo", label="data")
            ax.errorbar(tplot, numpy.nanmean(data, axis=-1), 2 * numpy.nanstd(data, axis=-1), label=r'$\mu \pm \sigma$',color='#88CCEE')
            ax.errorbar(tplot, numpy.nanmean(datappd, axis=-1), 2 * numpy.nanstd(datappd, axis=-1), label=r'$\mu \pm \sigma$', color='#117733')
            ax.legend()
            ax.set_ylabel('$R$ [m]')
            ax.set_xlabel('$t$ [s]')
            ax.set_xscale('log')
            ax.grid()

            # BIR = pandas.DataFrame({'t': [tplot], 'muR': [numpy.nanmean(data, axis=-1)], 'stdR': [numpy.nanstd(data, axis=-1)]})
            # BIR.to_csv(BIRout+str(nsamples)+'.csv', index=False)

        # # ##### saving data #####
        # # # uncertainty in dpdr and r
        # numpy.savetxt('dpdrt0.csv', ndpdrt0, delimiter=',')
        # numpy.savetxt('dpdrt1.csv', ndpdrt1, delimiter=',')
        # numpy.savetxt('dpdrt2.csv', ndpdrt2, delimiter=',')
        # numpy.savetxt('rt0.csv', nrt0, delimiter=',')
        # numpy.savetxt('rt1.csv', nrt1, delimiter=',')
        # numpy.savetxt('rt2.csv', nrt2, delimiter=',')

        # # model response + credibility
        # mean_model = pandas.DataFrame({'time': tplot, 'Rm005': numpy.quantile(data, 0.05, axis=-1), 'Rm05': numpy.quantile(data, 0.5, axis=-1), 'Rm095': numpy.quantile(data, 0.95, axis=-1)})
        # mean_model.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/' + fmodelname)
        #
        # # Posterior predictive distribution (mu + 2sigma)
        # mean_ppdmodel = pandas.DataFrame({'time': tplot, 'muppdRm': numpy.nanmean(datappd, axis=-1), 'stdppdRm': numpy.nanstd(datappd, axis=-1)})
        # mean_ppdmodel.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/' + ppdmodelname)
        #
        # # posterior samples model input parameters
        # data_samples = pandas.DataFrame({v: flat_samples[:, ind] for ind, v in enumerate(model.param_names)})
        # data_samples.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/ParametricUncertaintyFiles/' + fname)
        #
        # # x number of chains in n samples for model input parameters
        # for i in range(model.nparams):
        #     trace_data = pandas.DataFrame(samples[:, :, i])
        #     trace_data.to_csv('C:/Users/s152754/PycharmProjects/nutils-squeezeflow/ParametricUncertaintyFiles/' + tracename + model.param_names[i] + '.csv')
        #
        # # how to load data concerning w1, w2, h and vr (op welke tijdstippen?)
        #
        #
        # # treelog.user("muR \n", numpy.nanmean(data, axis=-1))
        # # treelog.user("stdR \n", numpy.nanstd(data, axis=-1))

cli.run(main)