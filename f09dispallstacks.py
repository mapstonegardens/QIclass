ang0=[0,1,2]
angA=[5,10,15]
angB=[20,25,30]
angC=[35,40,45]
boxvp, boxvs, boxrho=outputmodelvpvsrho(box, isi)
rc0, _, _ =partialstacks(boxvp,boxvs,boxrho,ang0)
rc5, rc10, rc15=partialstacks(boxvp,boxvs,boxrho,angA)
rc20, rc25, rc30=partialstacks(boxvp,boxvs,boxrho,angB)
rc35, rc40, rc45=partialstacks(boxvp,boxvs,boxrho,angC)
syn0=synthfrimage(rc0,wavelet); syn5=synthfrimage(rc5,wavelet)
syn10=synthfrimage(rc10,wavelet);syn15=synthfrimage(rc15,wavelet)
syn20=synthfrimage(rc20,wavelet);syn25=synthfrimage(rc25,wavelet)
syn30=synthfrimage(rc30,wavelet);syn35=synthfrimage(rc35,wavelet)
syn40=synthfrimage(rc40,wavelet);syn45=synthfrimage(rc45,wavelet)
#----------------The Wiggles...................
near=syn0
srt=2; gain=10; skip=10
[n_samples,n_traces]=syn0.shape
t=range(n_samples)
f, axs = plt.subplots(nrows=1, ncols=10, figsize=(12,10))
f.subplots_adjust(wspace=0, hspace=0)
ax0=axs[0]
for i in range(0, n_traces,skip):
    t0=gain*syn0[:,i] / np.max(np.abs(near))
    ax0.plot(i+t0,t,color='k', linewidth=0.6)
    ax0.fill_betweenx(t,t0+i,i, where=t0+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax0.fill_betweenx(t,t0+i,i, where=t0+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax1=axs[1]
for i in range(0, n_traces,skip):
    t5=gain*syn5[:,i] / np.max(np.abs(near))
    ax1.plot(i+t5,t,color='k', linewidth=0.6)
    ax1.fill_betweenx(t,t5+i,i, where=t5+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax1.fill_betweenx(t,t5+i,i, where=t5+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax2=axs[2];
for i in range(0, n_traces,skip):
    t10=gain*syn10[:,i] / np.max(np.abs(near))
    ax2.plot(i+t10,t,color='k', linewidth=0.6)
    ax2.fill_betweenx(t,t10+i,i, where=t10+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax2.fill_betweenx(t,t10+i,i, where=t10+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax3=axs[3];
for i in range(0, n_traces,skip):
    t15=gain*syn15[:,i] / np.max(np.abs(near))
    ax3.plot(i+t15,t,color='k', linewidth=0.6)
    ax3.fill_betweenx(t,t15+i,i, where=t15+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax3.fill_betweenx(t,t15+i,i, where=t15+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax4=axs[4];
for i in range(0, n_traces,skip):
    t20=gain*syn20[:,i] / np.max(np.abs(near))
    ax4.plot(i+t20,t,color='k', linewidth=0.6)
    ax4.fill_betweenx(t,t20+i,i, where=t20+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax4.fill_betweenx(t,t20+i,i, where=t20+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax5=axs[5];
for i in range(0, n_traces,skip):
    t25=gain*syn25[:,i] / np.max(np.abs(near))
    ax5.plot(i+t25,t,color='k', linewidth=0.6)
    ax5.fill_betweenx(t,t25+i,i, where=t25+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax5.fill_betweenx(t,t25+i,i, where=t25+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax6=axs[6];
for i in range(0, n_traces,skip):
    t30=gain*syn30[:,i] / np.max(np.abs(near))
    ax6.plot(i+t30,t,color='k', linewidth=0.6)
    ax6.fill_betweenx(t,t30+i,i, where=t30+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax6.fill_betweenx(t,t30+i,i, where=t30+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax7=axs[7];
for i in range(0, n_traces,skip):
    t35=gain*syn35[:,i] / np.max(np.abs(near))
    ax7.plot(i+t35,t,color='k', linewidth=0.6)
    ax7.fill_betweenx(t,t35+i,i, where=t35+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax7.fill_betweenx(t,t35+i,i, where=t35+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax8=axs[8];
for i in range(0, n_traces,skip):
    t40=gain*syn40[:,i] / np.max(np.abs(near))
    ax8.plot(i+t40,t,color='k', linewidth=0.6)
    ax8.fill_betweenx(t,t40+i,i, where=t40+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax8.fill_betweenx(t,t40+i,i, where=t40+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
ax9=axs[9];
for i in range(0, n_traces,skip):
    t45=gain*syn45[:,i] / np.max(np.abs(near))
    ax9.plot(i+t45,t,color='k', linewidth=0.6)
    ax9.fill_betweenx(t,t45+i,i, where=t45+i>i, facecolor="darkblue", linewidth=0, alpha=0.6)
    ax9.fill_betweenx(t,t45+i,i, where=t45+i<i, facecolor="crimson", linewidth=0, alpha=0.6)
for aa in axs:
    aa.grid(linestyle=":", color="gray"); 
    aa.set_ylim([0,n_samples])
    aa.invert_yaxis()
    aa.xaxis.set_label_position('top')
    aa.axis('off')
    aa.axhline(y=20, linewidth=0.6, linestyle="--", color='r')
    aa.axhline(y=40, linewidth=0.6, linestyle="--", color='b')
plt.show()