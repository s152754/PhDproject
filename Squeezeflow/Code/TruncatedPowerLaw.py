from nutils import cli, types, export
import treelog, numpy, pandas
from matplotlib import pyplot
from cycler import cycler
import pickle

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min')
_    = numpy.newaxis

######## options ########
# run solely TPL.py (cli.run(main), post-processing on)
# run to get timeseries.csv --> return the timeseries
#########################################################

pyplot.rcParams.update(pyplot.rcParamsDefault)

preamble = r'\usepackage{amsmath}\usepackage{bm}\usepackage{stmaryrd}'
params = {
   'axes.labelsize': 15,
   'axes.linewidth': 0.9,
   # 'axes.prop_cycle' : cycler(color=['#88CCEE', '#882255', '#332288', '#CC6677', '#999933', '#661100', '#DDCC77', '#44AA99', '#117733', '#AA4499', '#888888']), elisa
   'axes.prop_cycle' : cycler(color=['#332288','#AA4499','#44AA99','#999933','#88CCEE','#CC6677','#DDCC77','#117733','#888888','#6699CC','#661100','#882255']),
   # 'figure.dpi': 120,
   'font.size': 12,
   'legend.fontsize': 15,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'figure.dpi':300,
   'figure.figsize': [5.5, 4.8], #4 plots excl. \Deltap, #5.2 3.9 iets: [6.6,5.0],
   'text.usetex': True,
   'text.latex.preamble': preamble
   }
pyplot.rcParams.update(params)

def main(mu0: unit['Pa*s'], muinf: unit['Pa*s'], tcr: unit['s'], n: float, R0: unit['m'], h0: unit['m'], F: unit['N'], T: unit['s'], m: int, npicard: int, ntarget: int, tol: float, s: float, Dtmax: unit['s'], relax: float, gamma: unit['N/m'], alpha: float):
    '''
    Radial squeeze flow of a non-newtonian fluid
    .. arguments::
        mu0 [32.0Pa*s]
            Viscosity at zero shear rate
        muinf [0.001Pa*s]
            Viscosity at infinite shear rate
        tcr [0.712s]
            Newton/Power law cross over time scale
        n [0.771]
            Power index
        R0 [1.18cm]
            Initial radius of fluid domain
        h0 [0.575mm]
            Initial semi-height of fluid domain
        F [2.82N]
            Loading
        T [350s]
            Final time
        m [1000]
            Number of (radial) elements (for integration)
        s [0.001]
            Initial time step scaling
        Dtmax [1.s]
            Maximum time step size
        npicard [500]
            Number of Picard iterations
        ntarget [20]
            Target number of Picard iterations
        relax [0.25]
            Picard relation parameter
        tol [0.00001]
            Tolerance for Picard iterations
        gamma [66.mN/m]
            surface tension
        alpha [1.]
            factor to decide on curvature laplace boundary
    '''
    # of h=1.26 mm
    # initialization
    t = 0
    h = h0
    R = R0 # moet voor Laplace initialization
    r = numpy.linspace(0,R0,m+1)
    p = 2*F/(numpy.pi*R0**2)*(1-(r/R0)**2)

    # post processing
    pp = PostProcessing(mu0, muinf, tcr, n)

    # initial time step based on lowest possible viscosity solution
    Δt = s*(3*numpy.pi*muinf*R0**4)/(8*F*h0**2) # s=1 kan vgm ook
    # treelog.user("time step initial\n",Δt)
    # treelog.user("what is s\n", s)
    # differentiate the pressure
    dpdr = differentiate(p, r)

    timelist = []
    dpdrlist = []
    rlist = []

    with treelog.context('solving for t={:4.2e} [s]', t/unit('s')) as printtime:
        while t < T:

            # print the time step
            printtime(t/unit('s'))

            converged = numpy.zeros(shape=dpdr.shape, dtype=bool)
            dpdr0     = dpdr.copy()
            with treelog.iter.plain('picard', range(npicard)) as iterations:
                for iteration in iterations:

                    # compute the interface positions
                    w1 = numpy.minimum(h, (mu0/tcr)*(1/abs(dpdr)))
                    w2 = numpy.minimum(h, (mu0/tcr)*(1/abs(dpdr))*(muinf/mu0)**(n/(n-1)))

                    # compute the diffusion coefficients
                    C1 = (1/mu0)*((mu0/muinf)*w2**2*w1-(mu0/muinf)*h**2*w1-(2/3)*w1**3) + 2*(tcr**(1-n)/mu0)**(1/n)*abs(dpdr)**((1-n)/n)*(n/(n+1))*(w1**((n+1)/n)-w2**((n+1)/n))*w1
                    C2 = 2*(tcr**(1-n)/mu0)**(1/n)*abs(dpdr)**((1-n)/n)*(n/(n+1))*(-(n+1)/(2*n+1)*w2**((2*n+1)/n)-(n/(2*n+1))*w1**((2*n+1)/n)+w2**((n+1)/n)*w1) + (1/muinf)*(w2**2-h**2)*(w2-w1)
                    C3 = (1/muinf)*(-(2/3)*h**3-(1/3)*w2**3+h**2*w2)
                    C  = -(C1+C2+C3)

                    rc   = 0.5*(r[:-1]+r[1:])
                    dr   = r[1:]-r[:-1]
                    f    = (rc/C*dr)[::-1].cumsum()[::-1]
                    # hdot = -F/(2*numpy.pi*(rc*f*dr).sum())  # excluding Laplace
                    hdot = -F/(2*numpy.pi*(rc*f*dr).sum()) + ( (gamma*alpha)/ h ) * R**2 / ((rc*f*dr).sum()) # including Laplace


                    ddpdr  = rc*hdot/C-dpdr
                    dpdr[~converged] += (1-relax)*ddpdr[~converged]
                    Δdpdr  = dpdr-dpdr0

                    err = abs(ddpdr/Δdpdr)
                    converged = (err<tol)
                    # treelog.info(f'{converged.sum()}/{converged.size} converged, max error={numpy.max(err):4.3e}')
                    # treelog.user("Delta dpdr", numpy.max(abs(Δdpdr)))
                    if all(converged):
                        break

            if all(converged):
                # plot the converged solution
                # pp.plot(t, r, dpdr, h, w1, w2)
                pp.sampledata(t, r, h)

                timelist.append(t)
                dpdrlist.append(dpdr)
                rlist.append(r)
                ################# convergence study #####################
                # if t >= 2.00e-3:                                      #
                #     pp.convergence(t,r,dpdr,m)                        #
                #     treelog.user("wat is bijbehorende t?\n",t)        #
                #     break                                             #
                #########################################################

                # call function to save dpdr for t in between 0.5 and 2.5

                # increment the time step
                Δt *= ntarget/iteration
                Δt  = min(Δt, Dtmax)
                h  += hdot*Δt
                R   = R0*numpy.sqrt(h0/h)
                r   = numpy.linspace(0,R,m+1)
                t  += Δt
            else:
                treelog.info(f'picard solver did not converge in {npicard} iterations for {converged.size-converged.sum()} point(s)')
                Δt *= ntarget/iteration
                dpdr = dpdr0.copy()
                raise Exception("no convergence")
                break # check to go to next simulation

        # dpdrfun = scipy.interpolate.interp1d(numpy.array(timelist), numpy.array(dpdrlist), axis=0)
        # rfun = scipy.interpolate.interp1d(numpy.array(timelist), numpy.array(rlist), axis=0)
        # dpdrt1 = dpdrfun(1.5)
        # dpdrt2 = dpdrfun(10)

    # return pp.df, timelist, dpdrlist, rlist#, dpdrt1, dpdrt2

    # return the time series data frame
    # ahytf = pp.sampledata(t, r, h)
    # treelog.user(ahytf)


class PostProcessing:

    def __init__(self, mu0, muinf, tcr, n, nz=100):
        self.df = pandas.DataFrame({'t': [], 'h': [], 'R': []})
        self.dfc = pandas.DataFrame({'r': [], 'p': []})
        self.mu0   = mu0
        self.muinf = muinf
        self.tcr   = tcr
        self.n     = n
        self.nz    = nz

    def sampledata(self, t, r, h):
        # time plots
        R = r[-1]
        self.df = pandas.concat([self.df, pandas.DataFrame({'t': [t], 'h': [h], 'R': [R]})], ignore_index=True)
        # with treelog.userfile('timeseries.csv', 'w') as f:
        #     self.df.to_csv(f, index=False)
        # return pandas.read_csv("timeseries.csv")

    def convergence(self,t,r,dpdr,m):
        case = "TPL0.csv"
        path = "C:/Users/s152754/PycharmProjects/nutils-squeezeflow/convergence/"

        # pressure
        rc = 0.5 * (r[:-1] + r[1:])
        dr = differentiate(r)
        p = numpy.append(0, numpy.cumsum(dpdr * dr)) - (dpdr * dr).sum()
        # p en r en m opslaan

        self.dfc = pandas.DataFrame({'r': r, 'p': p})
        self.dfc.to_csv(path+case)



    def plot(self, t, r, dpdr, h, w1, w2):

        rc = 0.5*(r[:-1]+r[1:])
        # dr = differentiate(r)
        #
        # p = numpy.append(0,numpy.cumsum(dpdr*dr))-(dpdr*dr).sum()
        #
        # with export.mplfigure('p.png') as fig:
        #     ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='p [Pa]')
        #     ax.plot(r / unit('mm'), p / unit('Pa'))
        #     ax.grid()
        #
        # with export.mplfigure('dpdr.png') as fig:
        #     ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='dp/dr [Pa/mm]')
        #     ax.plot(rc / unit('mm'), dpdr / (unit('Pa')/unit('mm')))
        #     ax.grid()
        #
        # # contour plots
        # z  = numpy.linspace(0,h,self.nz)
        # zc = 0.5*(z[:-1]+z[1:])
        # rm, zm = numpy.meshgrid(rc, zc)
        #
        # region = (zm>w1).astype(int)+(zm>w2).astype(int)
        #
        # γ1 = abs(1/self.mu0*dpdr[_,:]*zm)
        # γ2 = abs(dpdr[_,:]*(self.tcr**(1-self.n)/self.mu0)**(1/self.n)*abs(dpdr[_,:])**((1-self.n)/self.n)*(zm**(1/self.n)))
        # γ3 = abs(1/self.muinf*dpdr[_,:]*zm)
        # γ  = numpy.choose(region, [γ1, γ2, γ3])
        #
        # μ1 = self.mu0*numpy.ones_like(γ1)
        # μ2 = self.mu0*(self.tcr*γ2)**(self.n-1)
        # μ3 = self.muinf*numpy.ones_like(γ2)
        # μ  = numpy.choose(region, [μ1, μ2, μ3])
        #
        # with export.mplfigure('regions.png') as fig:
        #     ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]', ylim=(0, 1.1*h))
        #     ax.plot(rc / unit('mm'), h * numpy.ones_like(rc), label='$h$')
        #     ax.plot(rc / unit('mm'), w1, label='$w_1$')
        #     ax.plot(rc / unit('mm'), w2, label='$w_2$')
        #     ax.grid()
        #     ax.legend()

        with export.mplfigure('regionsppt.png') as fig:
            ax = fig.add_subplot(111, xlabel=r'$r \ [\mathrm{mm}]$', ylabel=r'$z \ [\mathrm{mm}]$', ylim=(0, 1.1*h))
            ax.plot(rc / unit('mm'), h * numpy.ones_like(rc) / unit('mm'), label=r'$h$', color='#888888')
            ax.plot(rc / unit('mm'), w2 / unit('mm'), label=r'$w_2$', color='#882255')
            ax.plot(rc / unit('mm'), w1 / unit('mm'), label=r'$w_1$', color='#6699CC')
            ax.axvline(x=rc[-1] / unit('mm'), ymax = ((h / unit('mm'))/0.7), color='#000000')
            ax.set_xlim([0, 22])
            ax.set_ylim([0, 0.7])
            ax.grid()
            ax.legend()

        # with export.mplfigure('gamma.png') as fig:
        #     ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]')
        #     img = ax.contourf(rm/unit('mm'), zm/unit('mm'), γ*unit('s'))
        #     fig.colorbar(img, label='γ [1/s]')
        #
        # with export.mplfigure('viscosity.png') as fig:
        #     ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]')
        #     img = ax.contourf(rm/unit('mm'), zm/unit('mm'), μ/unit('Pa*s'))
        #     fig.colorbar(img, label='μ [Pa s]')
        #
        # # time plots
        # R = r[-1]
        # self.df = pandas.concat([self.df, pandas.DataFrame({'t': [t], 'h': [h], 'R': [R]})], ignore_index=True)
        #
        # if self.df.shape[0]>0:
        #     with export.mplfigure('R.png') as fig:
        #         ax = fig.add_subplot(111, xlabel='t [s]', ylabel='R [mm]')
        #         ax.plot(self.df['t'] / unit('s'), self.df['R'] / unit('mm'), '.-')
        #         ax.grid()
        #
        #     with export.mplfigure('h.png') as fig:
        #         ax = fig.add_subplot(111, xlabel='$t$ [s]', ylabel='$h$ [mm]')
        #         ax.plot(self.df['t'] / unit('s'), self.df['h'] / unit('mm'), '.-')
        #         ax.grid()
        #         ax.legend()
        #
        # if self.df.shape[0]>1:
        #     hdot = differentiate(self.df['h'].to_numpy(), self.df['t'].to_numpy())
        #     with export.mplfigure('hdot.png') as fig:
        #         ax = fig.add_subplot(111, xlabel='t [s]', ylabel='hdot [mm/s]')
        #         ax.plot(self.df['t'][1:] / unit('s'), hdot / (unit('mm')/unit('s')), '.-')
        #         ax.grid()
        #
        # # save data frame to file
        # with treelog.userfile('timeseries.csv', 'w') as f:
        #     self.df.to_csv(f, index=False)


def differentiate(f, x=None):
    if x is None:
        return f[1:]-f[:-1]
    return (f[1:]-f[:-1])/(x[1:]-x[:-1])

cli.run(main)
