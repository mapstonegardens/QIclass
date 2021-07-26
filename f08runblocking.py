top_picks=[top_plot,pick1,pick2,pick3,pick4,base_plot]
z=logs.index.values
jumlah=np.size(top_picks)-1
vp1,vs1,rho1,vsh1,phit1,swt1=(np.zeros(jumlah) for _ in range(6))
for aa in range(jumlah):
    bb=(logs.index>=top_picks[aa]) & (logs.index<top_picks[aa+1])
    vp1[aa]=logs.Vp[bb].mean()
    vs1[aa]=logs.Vs[bb].mean()
    rho1[aa]=logs.Rho[bb].mean()
    vsh1[aa]=logs.Vsh[bb].mean()
    phit1[aa]=logs.Phit[bb].mean()
    swt1[aa]=logs.Swt[bb].mean()
whole=int(base_plot-top_plot)
num=int((base_plot-top_plot)/search_window)
thickness=np.linspace(0,whole,num)
ztotal=z[(z>=top_picks[0]) & (z<=top_picks[5])]
z0=(ztotal>=top_picks[0]) & (ztotal<top_picks[1])
for cc in thickness:
    vp1_new=np.ones(ztotal.size)*vp1[4]
    vs1_new=np.ones(ztotal.size)*vs1[4]
    rho1_new=np.ones(ztotal.size)*rho1[4]
    vsh1_new=np.ones(ztotal.size)*vsh1[4]
    phit1_new=np.ones(ztotal.size)*phit1[4]
    swt1_new=np.ones(ztotal.size)*swt1[4]
    # interval-1
    vp1_new[z0]=vp1[0]
    vs1_new[z0]=vs1[0]
    rho1_new[z0]=rho1[0]
    vsh1_new[z0]=vsh1[0]
    phit1_new[z0]=phit1[0]
    swt1_new[z0]=swt1[0]
    # interval-2
    z1=(ztotal>=top_picks[1]) & (ztotal<top_picks[1]+cc)
    vp1_new[z1]=vp1[1]
    vs1_new[z1]=vs1[1]
    rho1_new[z1]=rho1[1]
    vsh1_new[z1]=vsh1[1]
    phit1_new[z1]=phit1[1]
    swt1_new[z1]=swt1[1]
    # interval-3
    z2=(ztotal>=top_picks[2]) & (ztotal<top_picks[2]+cc)
    vp1_new[z2]=vp1[2]
    vs1_new[z2]=vs1[2]
    rho1_new[z2]=rho1[2]
    vsh1_new[z2]=vsh1[2]
    phit1_new[z2]=phit1[2]
    swt1_new[z2]=swt1[2]
    # interval-4
    z3=(ztotal>=top_picks[3]) & (ztotal<top_picks[3]+cc)
    vp1_new[z3]=vp1[3]
    vs1_new[z3]=vs1[3]
    rho1_new[z3]=rho1[3]
    vsh1_new[z3]=vsh1[3]
    phit1_new[z3]=phit1[3]
    swt1_new[z3]=swt1[3]
    # interval-5
    z4=(ztotal>=top_picks[4]) & (ztotal<top_picks[4]+cc)
    vp1_new[z4]=vp1[4]
    vs1_new[z4]=vs1[4]
    rho1_new[z4]=rho1[4]
    vsh1_new[z4]=vsh1[4]
    phit1_new[z4]=phit1[4]
    swt1_new[z4]=swt1[4]
    # calculate new parameters
    vp_new=vp1_new
    vs_new=vs1_new
    rho_new=rho1_new
    vsh_new=vsh1_new
    phit_new=phit1_new
    swt_new=swt1_new

h1=pick1-top_plot
h2=pick2-pick1
h3=pick3-pick2
h4=pick4-pick3
h5=base_plot-pick4
thickness=[h1,h2,h3,h4,h5]
layer1_elog=[round(vp1[0],2),round(vs1[0],2),round(rho1[0],2)]
layer2_elog=[round(vp1[1],2),round(vs1[1],2),round(rho1[1],2)]
layer3_elog=[round(vp1[2],2),round(vs1[2],2),round(rho1[2],2)]
layer4_elog=[round(vp1[3],2),round(vs1[3],2),round(rho1[3],2)]
layer5_elog=[round(vp1[4],2),round(vs1[4],2),round(rho1[4],2)]
vplyr=[round(vp1[0],2),round(vp1[1],2),round(vp1[2],2),round(vp1[3],2),round(vp1[4],2)]
vslyr=[round(vs1[0],2),round(vs1[1],2),round(vs1[2],2),round(vs1[3],2),round(vs1[4],2)]
rholyr=[round(rho1[0],2),round(rho1[1],2),round(rho1[2],2),round(rho1[3],2),round(rho1[4],2)]
vshlyr=[round(vsh1[0],2),round(vsh1[1],2),round(vsh1[2],2),round(vsh1[3],2),round(vsh1[4],2)]
phitlyr=[round(phit1[0],2),round(phit1[1],2),round(phit1[2],2),round(phit1[3],2),round(phit1[4],2)]
swtlyr=[round(swt1[0],2),round(swt1[1],2),round(swt1[2],2),round(swt1[3],2),round(swt1[4],2)]
print("thickness =", thickness, sep="") 
print("layer1_elog =", layer1_elog, sep="") 
print("layer2_elog =", layer2_elog, sep="")
print("layer3_elog =", layer3_elog, sep="")
print("layer4_elog =", layer4_elog, sep="")
print("layer5_elog =", layer5_elog, sep="")
print("vplyr=", vplyr, sep="")
print("vslyr=", vslyr, sep="")
print("rholyr=", rholyr, sep="")
print("vshlyr=", vshlyr, sep="")
print("phitlyr=", phitlyr, sep="")
print("swtlyr=", swtlyr, sep="")