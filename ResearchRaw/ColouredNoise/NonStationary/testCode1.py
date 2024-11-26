import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Generate synthetic noise for demonstration
np.random.seed(0)
n = 1000
# Non-stationary noise example (random walk)
noise = np.cumsum(np.random.normal(size=n))

# Plot the noise
plt.figure(figsize=(10, 6))
plt.plot(noise)
plt.title('Noise Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}):', value)

# Function to perform KPSS test
def kpss_test(series, regression='c'):
    result = kpss(series, regression=regression)
    print('KPSS Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[3].items():
        print(f'Critical Value ({key}):', value)

# Perform ADF test
print("ADF Test Results:")
adf_test(noise)

# Perform KPSS test
print("\nKPSS Test Results:")
kpss_test(noise)

# Calculate rolling mean and variance
rolling_mean = pd.Series(noise).rolling(window=50).mean()
rolling_var = pd.Series(noise).rolling(window=50).var()

# Plot rolling statistics
plt.figure(figsize=(12, 6))
plt.plot(noise, label='Original')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_var, label='Rolling Variance', color='green')
plt.legend(loc='best')
plt.title('Rolling Mean & Variance')
plt.show()
