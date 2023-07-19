from nutils import cli, mesh, function, solver, export, types, testing
import numpy, treelog, typing, pandas,os, os.path

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min', m3='m*m*m')
_ = numpy.newaxis

def main(dtmax: unit['s'], s: float, T: unit['s'], R0: unit['m'], eta: unit['Pa*s'], h0: unit['m'], F: unit['N'], gamma: unit['N/m'], alpha: float):
    '''
    Radial squeeze flow of a truncated power law fluid

    .. arguments::
            dtmax [1.s]
                maximum time step
            s [1.]
                factor increasing/decreasing first time step
            T [300s]
                Final time
            R0 [0.733cm]
                initial radius
            eta [32.Pa*s]
                viscosity
            h0 [0.036mm]
                initial height
            F [8.56N]
                applied force
            gamma [0.045N/m]
                surface tension
            alpha [1]
                factor radius of gyration dependence on h
    '''
    pp = PostProcessing()

    V = numpy.pi * 2 * h0 * R0**2
    h0 = 0.5 * (V / (numpy.pi * R0 ** 2))
    told = 0

    dt = abs(s * (3 * numpy.pi * eta * R0 ** 4) / (8 * F * h0 ** 2))
    hold = h0
    Rold = R0
    pp.propfile(told, Rold, hold)
    while told < T:
        hdot = - (32 * numpy.pi * (hold ** 5 * F - ( gamma * alpha / hold) * V * hold ** 4)) / ( 3 * eta * V ** 2)  # kappa = 1 / Ri = 1 / (alpha * h)
        hnew = hold + hdot * dt
        Rnew = R0 * numpy.sqrt(h0 / hnew)
        tnew = told + dt

        pp.propfile(tnew, Rnew, hnew)

        dtprop = dt * 1.1
        dt = min(dtprop, dtmax)  # find more appropriate adaptive time stepper

        hold = hnew
        told = tnew
    treelog.user(Rnew)
    return pp.df

class PostProcessing:

    def __init__(self):
        self.df = pandas.DataFrame({'t': [], 'h': [], 'R': []})

    def propfile(self,t,R,h):
        self.df = pandas.concat([self.df, pandas.DataFrame({'t': [t], 'h': [h], 'R': [R]})], ignore_index=True)

cli.run(main)