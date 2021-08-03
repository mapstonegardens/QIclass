import matplotlib.patches as patches
freq_low=0
freq_high=90
duration = 1.0 # panjang recording signal
fs = 500.0
samples = int(fs*duration) #number of samples sepanjang 1 second record
t = np.arange(samples) / fs
signal = chirp(t, freq_high, t[-2], freq_low)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
h_signal = hilbert(signal)
amp_env = np.abs(h_signal)
inst_ph = np.unwrap(np.angle(h_signal))
inst_freq = (np.diff(inst_ph) / (2.0*np.pi) * fs)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(12,8))
ax0.plot(t*1000, signal, 'k', label='signal')
ax0.plot(t*1000, amp_env, 'b', label='envelope', lw=2)
ax0.legend(loc='lower right', frameon=False, prop={'size':12})
ax0.set_ylabel("amplitude", size=12)
ax1.plot(t*1000, inst_ph,'k')
ax1.set_ylabel("phase", size=12)
ax2.plot(t[1:]*1000, inst_freq,'k')
ax2.set_xlabel("time", size=20)
ax2.set_ylabel("frequency", size=12)
ax2.set_ylim(0.0, 120.0)
rect_a = patches.Rectangle((200, -1.5), 200, 3, linewidth=3, edgecolor='r', facecolor='none')
rect_ph = patches.Rectangle((200, 100), 200, 100, linewidth=3, edgecolor='r', facecolor='none')
rect_freq = patches.Rectangle((200, 45), 200, 40, linewidth=3, edgecolor='r', facecolor='none')
ax0.add_patch(rect_a)
ax1.add_patch(rect_ph)
ax2.add_patch(rect_freq)
ax1.text(235, 220, "Interval Window", color='r', size=12)
fig.tight_layout()