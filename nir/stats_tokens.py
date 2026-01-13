from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np

# KS-Test
ks_stat, ks_p = ks_2samp(syllable_counts, subword_counts)

# Wasserstein (EMD)
emd = wasserstein_distance(syllable_counts, subword_counts)

# MAE
mae = np.mean(np.abs(np.array(syllable_counts) - np.array(subword_counts)))

print(f"KS Statistic: {ks_stat:.4f}")
print(f"Earth Mover's Distance: {emd:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")



import matplotlib.pyplot as plt

plt.hist(syllable_counts, bins=range(1, max(syllable_counts + subword_counts) + 2), 
         alpha=0.5, label='Syllables', density=True)
plt.hist(subword_counts, bins=range(1, max(syllable_counts + subword_counts) + 2), 
         alpha=0.5, label='Subwords', density=True)
plt.xlabel('Units per Word')
plt.ylabel('Probability')
plt.title('Syllable vs. Subword Token Count per Word')
plt.legend()
plt.show()
