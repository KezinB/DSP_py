from scipy.signal import hilbert

# Phase analysis for RLS and FT-RLS
phase_rls = np.unwrap(np.angle(hilbert(y_rls)))
phase_ftrls = np.unwrap(np.angle(hilbert(y_ftrls)))

# Phase difference between RLS and FT-RLS
phase_diff_rls_ftrls = phase_rls - phase_ftrls

# Plotting phase difference
plt.figure(figsize=(10, 4))
plt.plot(t, phase_diff_rls_ftrls, label="Phase Difference (RLS - FT-RLS)", color="purple")
plt.title("Phase Difference Between RLS and FT-RLS Outputs")
plt.xlabel("Time")
plt.ylabel("Phase Difference (radians)")
plt.grid()
plt.legend()
plt.show()
