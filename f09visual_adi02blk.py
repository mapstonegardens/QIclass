int1=(adi2['Depth'] > top_plot) & (adi2['Depth'] <= pick1)
int2=(adi2['Depth'] > pick1) & (adi2['Depth'] <= pick2)
int3=(adi2['Depth'] > pick2) & (adi2['Depth'] <= pick3)
int4=(adi2['Depth'] > pick3) & (adi2['Depth'] <= pick4)
int5=(adi2['Depth'] > pick3) & (adi2['Depth'] <= base_plot)
layers=[int1,int2,int3,int4,int5]
vp_list=[]
vs_list=[]
rho_list=[]
for i in layers:
    vp=adi2.loc[i,['Vp_blk']].mean()
    vs=adi2.loc[i,['Vs_blk']].mean()
    rho=adi2.loc[i,['Rho_blk']].mean()
    i+=1
    vp_list.append(vp)
    vs_list.append(vs)
    rho_list.append(rho)
vp_list=np.array(vp_list, dtype='float').round(3)
vs_list=np.array(vs_list, dtype='float').round(3)
rho_list=np.array(rho_list, dtype='float').round(3)
#print("vp =",''.join(str(x) for x in vp_list))
#print("vs =",''.join(str(x) for x in vs_list))
#print("rho =",''.join(str(x) for x in rho_list))
xplot=np.arange(0, 5)
y1=xplot*0+top_plot; y2=xplot*0+pick1
y3=xplot*0+pick2; y4=xplot*0+pick3
y5=xplot*0+pick4; y6=xplot*0+base_plot
vsh = adi2.Vsh_blk
vp = adi2.Vp_blk
vs = adi2.Vs_blk
rho = adi2.Rho_blk
ai = vp*rho
vpvs = vp/vs
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(8,5), sharey=True)
axes=(ax1,ax2,ax3,ax4)
for aa in axes:
    aa.xaxis.set_label_position('top')
    aa.invert_yaxis()
    aa.plot(xplot,y2,'k', lw=0.5)
    aa.plot(xplot,y3,'k', lw=0.5)
    aa.plot(xplot,y4,'k', lw=0.5)
    aa.plot(xplot,y5,'k', lw=0.5)
    aa.fill_between(xplot,pick2,pick3, color='yellow', alpha = 0.2)
ax1.plot(vsh, z, 'C1', lw=4), ax1.set_xlabel('Vsh', size=20);
ax1.set_ylabel('Depth',size=20), ax1.set_xlim(0,1), ax1.set_ylim(base_plot,top_plot);
ax2.plot(vp, z, 'C2', lw=4), ax2.set_xlabel('Vp', size=20), ax2.set_xlim(3,3.55);
ax3.plot(vs, z, 'C2', lw=4), ax3.set_xlabel('Vs', size=20), ax3.set_xlim(1.7,2);
ax4.plot(rho, z, 'C2',lw=4), ax4.set_xlabel('Rho', size=20), ax4.set_xlim(2.2,2.55);
ax1.text(0.1, 1665, "shalysand", size=14); ax3.text(1.73, 1665, "vs=1.846", size=14); 
ax1.text(0.1, 1700, "shale", size=14); ax3.text(1.78, 1700, "vs=1.773", size=14);
ax1.text(0.25, 1730, "channel", size=14); ax3.text(1.75, 1730, "vs=1.720", size=14);
ax1.text(0.1, 1750, "bar", size=14); ax3.text(1.78, 1750, "vs=1.964", size=14);
ax1.text(0.1, 1790, "shale", size=14); ax3.text(1.73, 1790, "vs=1.906", size=14);
ax2.text(3.05, 1665, "vp=3.299", size=14); ax4.text(2.25, 1665, "rho=2.421", size=14);
ax2.text(3.05, 1700, "vp=3.278", size=14); ax4.text(2.25, 1700, "rho=2.445", size=14);
ax2.text(3.1, 1730, "vp=3.082", size=14); ax4.text(2.3, 1730, "rho=2.245", size=14);
ax2.text(3.05, 1750, "vp=3.413", size=14); ax4.text(2.25, 1750, "rho=2.323", size=14);
ax2.text(3.05, 1790, "vp=3.449", size=14); ax4.text(2.25, 1790, "rho=2.402", size=14);