#kalkulasi Vp
pd.set_option('display.float_format', lambda x: '%.2f' % x)
vpa=logs.Vp.hist(bins=10, color='grey', by=logs.fac, figsize=(12,2), layout=(1,5), lw=0)
vp1=logs[(logs.fac>0.9) & (logs.fac<1.1)][['Vp']].describe([]).T
vp2=logs[(logs.fac>1.9) & (logs.fac<2.1)][['Vp']].describe([]).T
vp3=logs[(logs.fac>2.9) & (logs.fac<3.1)][['Vp']].describe([]).T
vp4=logs[(logs.fac>3.9) & (logs.fac<4.1)][['Vp']].describe([]).T
vp5=logs[(logs.fac>4.9) & (logs.fac<5.1)][['Vp']].describe([]).T
vpsum={'oil':vp1, 'ssd':vp2,'sh':vp3,'brine':vp4, 'coal':vp5}
vpsum_concat = pd.concat(vpsum).drop(columns=['count'])
#kalkulasi Vs
pd.set_option('display.float_format', lambda x: '%.2f' % x)
vsfacies=logs.Vs.hist(bins=10, color='grey', by=logs.fac, figsize=(12,2), layout=(1,5), lw=0)
vs1=logs[(logs.fac>0.9) & (logs.fac<1.1)][['Vs']].describe([]).T
vs2=logs[(logs.fac>1.9) & (logs.fac<2.1)][['Vs']].describe([]).T
vs3=logs[(logs.fac>2.9) & (logs.fac<3.1)][['Vs']].describe([]).T
vs4=logs[(logs.fac>3.9) & (logs.fac<4.1)][['Vs']].describe([]).T
vs5=logs[(logs.fac>4.9) & (logs.fac<5.1)][['Vs']].describe([]).T
vssum={'oil':vs1, 'ssd':vs2,'sh':vs3,'brine':vs4,'coal':vs5}
vssum_concat = pd.concat(vssum).drop(columns=['count'])

#kalkulasi Rho
pd.set_option('display.float_format', lambda x: '%.2f' % x)
rhofacies=logs.Rho.hist(bins=10, color='grey', by=logs.fac, figsize=(12,2), layout=(1,5), lw=0)
rho1=logs[(logs.fac>0.9) & (logs.fac<1.1)][['Rho']].describe([]).T
rho2=logs[(logs.fac>1.9) & (logs.fac<2.1)][['Rho']].describe([]).T
rho3=logs[(logs.fac>2.9) & (logs.fac<3.1)][['Rho']].describe([]).T
rho4=logs[(logs.fac>3.9) & (logs.fac<4.1)][['Rho']].describe([]).T
rho5=logs[(logs.fac>4.9) & (logs.fac<5.1)][['Rho']].describe([]).T
rhosum={'oil':rho1, 'ssd':rho2,'sh':rho3,'brine':rho4,'coal':rho5}
rhosum_concat = pd.concat(rhosum).drop(columns=['count'])
print(vpsum_concat)
print(vssum_concat)
print(rhosum_concat)