import numpy as np
import scipy.optimize
import pylab as plt

def fit_sin(tt, yy):
    """Fit sin to the input time sequence, and return fitting parameters 'amp',
    'omega', 'phase', 'offset', 'freq', 'period' and 'fitfunc'."""
    
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    
    return {
        "amp": A, "omega": w, "phase": p, "offset": c, 
        "freq": f, "period": 1. / f, "fitfunc": fitfunc, 
        "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)
    }

# Parameters
N, amp, omega, phase, offset, noise = 500, 1., 2., .5, 4., 3

tt = np.linspace(0, 10, N)
tt2 = np.linspace(0, 10, 10 * N)
yy = amp * np.sin(omega * tt + phase) + offset
yynoise = yy + noise * (np.random.random(len(tt)) - 0.5)

res = fit_sin(tt, yynoise)

# Printing results with the correct format
print(f"Amplitude={res['amp']}, Angular freq.={res['omega']}, "
      f"Phase={res['phase']}, Offset={res['offset']}, "
      f"Max. Cov.={res['maxcov']}")

# Plotting the results
plt.title("Least Squares Fit to a Sinusoidal Function")
plt.plot(tt, yy, "-k", label="y", linewidth=2)
plt.plot(tt, yynoise, "ok", label="y with noise")
plt.plot(tt2, res["fitfunc"](tt2), "r-", label="y fit curve", linewidth=2)
plt.legend(loc="best")
info_text = (f"Amplitude = {res['amp']}\n"
             f"Angular Frequency = {res['omega']}\n"
             f"Phase = {res['phase']:.2f}\n"
             f"Offset = {res['offset']}\n"
             f"Max Covariance = {res['maxcov']}")
plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()
