import numpy as np
import matplotlib.pyplot as plt

def fit_sin(tt, yy, max_iter=10000, tol=1e-6, learning_rate=1e-6):
    """Fit sin to the input time sequence using gradient descent.
    Return fitting parameters 'amp', 'omega', 'phase', 'offset',
    'freq', 'period' and 'fitfunc'."""
    
    tt = np.array(tt)
    yy = np.array(yy)
    
    # Initial guesses
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
    guess_amp = np.std(yy) * np.sqrt(2)
    guess_offset = np.mean(yy)
    A, w, p, c = guess_amp, 2. * np.pi * guess_freq, 0., guess_offset

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    # Gradient Descent for parameter optimization
    def loss(params):
        A, w, p, c = params
        return np.sum((yy - sinfunc(tt, A, w, p, c))**2)

    def gradient(params):
        """Compute the numerical gradient."""
        grad = np.zeros_like(params)
        h = 1e-8  # Small step size for numerical differentiation
        for i in range(len(params)):
            params_step = np.array(params)
            params_step[i] += h
            grad[i] = (loss(params_step) - loss(params)) / h
        return grad

    # Gradient descent loop
    params = np.array([A, w, p, c])
    for _ in range(max_iter):
        grad = gradient(params)
        new_params = params - learning_rate * grad
        if np.linalg.norm(new_params - params) < tol:
            break
        params = new_params

    A, w, p, c = params
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c

    # No covariance matrix, but return fit results
    return {
        "amp": A, "omega": w, "phase": p, "offset": c, 
        "freq": f, "period": 1. / f, "fitfunc": fitfunc
    }

# User input for test data parameters
N = int(input("Enter the number of samples (e.g., 500): "))
amp = float(input("Enter the amplitude (e.g., 1.0): "))
omega = float(input("Enter the angular frequency ω (e.g., 2.0): "))
phase = float(input("Enter the phase (in radians, e.g., 0.5): "))
offset = float(input("Enter the offset value (e.g., 4.0): "))
noise = float(input("Enter the noise level (e.g., 3.0): "))

# Generate the time sequence and noisy data
tt = np.linspace(0, 10, N)
tt2 = np.linspace(0, 10, 10 * N)
yy = amp * np.sin(omega * tt + phase) + offset
yynoise = yy + noise * (np.random.random(len(tt)) - 0.5)

# Perform the fitting
res = fit_sin(tt, yynoise)

# Print the fitting results
print(f"Amplitude = {res['amp']}, Angular frequency = {res['omega']}, "
      f"Phase = {res['phase']}, Offset = {res['offset']}")

# Plot the results
plt.title("Least Squares Fit to a Sinusoidal Function")
plt.plot(tt, yy, "-k", label="True function", linewidth=2)
plt.plot(tt, yynoise, "ok", label="Noisy data")
plt.plot(tt2, res["fitfunc"](tt2), "r-", label="Fitted curve", linewidth=2)
plt.legend(loc="best")

# Display the results on the plot
info_text = (f"Amplitude = {res['amp']}\n"
             f"Angular Frequency = {res['omega']}\n"
             f"Phase = {res['phase']:.2f}\n"
             f"Offset = {res['offset']}")
plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Show the plot
plt.show()
