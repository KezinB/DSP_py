import random
import matplotlib.pyplot as plt
import math

original_distribution = random.uniform
original_params = (0, 1)
sample_size = 5
num_samples = 10000
samples = [[original_distribution(*original_params) for _ in range(sample_size)]
           
for _ in range(num_samples)]
sample_means = [sum(sample) / len(sample) for sample in samples]
plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='g', label='SampleMeans')
mu = sum(sample_means) / len(sample_means)
sigma = math.sqrt(sum((x - mu) ** 2 for x in sample_means) / len(sample_means))
xmin, xmax = plt.xlim()
x = [xmin + i * (xmax - xmin) / 100 for i in range(100)]
p = [1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(i - mu) ** 2 / (2 * sigma **2)) for i in x]
plt.plot(x, p, 'k', linewidth=2, label='CLT')  

# Display the plot
plt.title('Central Limit Theorem Demonstration')
plt.xlabel('Sample Mean')
plt.ylabel('Probability Density')
plt.legend()
plt.show()