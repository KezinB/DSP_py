import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Generate synthetic noise for demonstration
np.random.seed(0)
n = 1000

# Non-stationary noise example (random walk)
non_stationary_noise = np.cumsum(np.random.normal(size=n))

# Stationary noise example (white noise)
stationary_noise = np.random.normal(size=n)

# Plot the noises
plt.figure(figsize=(10, 6))
plt.plot(non_stationary_noise, label='Non-Stationary Noise (Random Walk)')
plt.plot(stationary_noise, label='Stationary Noise (White Noise)', alpha=0.75)
plt.title('Noise Signals')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}):', value)
    return result[1]

# Function to perform KPSS test
def kpss_test(series, regression='c'):
    result = kpss(series, regression=regression)
    print('KPSS Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[3].items():
        print(f'Critical Value ({key}):', value)
    return result[1]

# Perform tests on non-stationary noise
print("ADF Test Results for Non-Stationary Noise:")
adf_p_value_non_stationary = adf_test(non_stationary_noise)
print("\nKPSS Test Results for Non-Stationary Noise:")
kpss_p_value_non_stationary = kpss_test(non_stationary_noise)

# Perform tests on stationary noise
print("\nADF Test Results for Stationary Noise:")
adf_p_value_stationary = adf_test(stationary_noise)
print("\nKPSS Test Results for Stationary Noise:")
kpss_p_value_stationary = kpss_test(stationary_noise)

# Calculate rolling mean and variance
rolling_mean_non_stationary = pd.Series(non_stationary_noise).rolling(window=50).mean()
rolling_var_non_stationary = pd.Series(non_stationary_noise).rolling(window=50).var()
rolling_mean_stationary = pd.Series(stationary_noise).rolling(window=50).mean()
rolling_var_stationary = pd.Series(stationary_noise).rolling(window=50).var()

# Plot rolling statistics for non-stationary noise
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(non_stationary_noise, label='Non-Stationary Noise')
plt.plot(rolling_mean_non_stationary, label='Rolling Mean', color='red')
plt.plot(rolling_var_non_stationary, label='Rolling Variance', color='green')
plt.legend(loc='best')
plt.title('Non-Stationary Noise: Rolling Mean & Variance')

# Plot rolling statistics for stationary noise
plt.subplot(2, 1, 2)
plt.plot(stationary_noise, label='Stationary Noise')
plt.plot(rolling_mean_stationary, label='Rolling Mean', color='red')
plt.plot(rolling_var_stationary, label='Rolling Variance', color='green')
plt.legend(loc='best')
plt.title('Stationary Noise: Rolling Mean & Variance')

plt.tight_layout()
plt.show()

# Determine and print the stationarity
print("\nDetermining Stationarity:")
if adf_p_value_non_stationary < 0.05 and kpss_p_value_non_stationary >= 0.05:
    print("The non-stationary noise signal is stationary based on the tests.")
else:
    print("The non-stationary noise signal is non-stationary based on the tests.")

if adf_p_value_stationary < 0.05 and kpss_p_value_stationary >= 0.05:
    print("The stationary noise signal is stationary based on the tests.")
else:
    print("The stationary noise signal is non-stationary based on the tests.")
