import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
import matplotlib as mpl
import readparams as rdp
from ploteb import sliceDomain,getisopyc


mpl.rcParams.update({'font.weight':'bold'})

def pvterms(firstrun,geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,savfil=None):

    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp.getgeom(geofil)
    
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)

    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)

    if firstrun == 1:
        em = np.zeros((nzp1,ny,nx))
        elm = np.zeros((nz,ny,nx))
        um = np.zeros((nz,ny,nx))
        vm = np.zeros((nz,ny,nx))
        vadvxm = np.zeros((nz,ny,nx))
        fvym = np.zeros((nz,ny,nx))
        fuxm = np.zeros((nz,ny,nx))
        Y1xm = np.zeros((nz,ny,nx))
        Y2xm = np.zeros((nz,ny,nx))
        for i in range(nt):
            (u,v,h,e,el) = rdp.getuvhe(fil,i)
            (frhatu,frhatv) = rdp.gethatuv(fil,i)
            em += e/nt
            elm += el/nt
            um += u/nt
            vm += v/nt
            (dudt,cau,pfu,dudtvisc,diffu,dudtdia,gkeu,rvxv) = rdp.getmomxterms(fil,i)
            (dvdt,cav,pfv,dvdtvisc,diffv,dvdtdia,gkev,rvxu) = rdp.getmomyterms(fil,i)
            (dhdt,uh,vh,wd,uhgm,vhgm) = rdp.getcontterms(fil,i)
            vadv = gkev + rvxu
            vadv = np.ma.filled(vadv.astype(float), 0)
            vadvx = np.diff(np.concatenate((vadv,vadv[:,:,[-1]]),axis=2),axis = 2)/dxbu
            vadvxm += vadvx/nt

            fv = cau - gkeu - rvxv
            fv = np.ma.filled(fv.astype(float), 0)
            fvy = np.diff(np.concatenate((fv,-fv[:,[-1],:]),axis=1),axis = 1)/dybu
            fvym += fvy/nt

            fu = cav - gkev - rvxu
            fu = np.ma.filled(fu.astype(float), 0)
            fux = np.diff(np.concatenate((fu,-fu[:,:,[-1]]),axis=2),axis = 2)/dxbu
            fuxm += fux/nt

            Y1 = diffv
            Y1 = np.ma.filled(Y1.astype(float), 0)
            Y1x = np.diff(np.concatenate((Y1,Y1[:,:,[-1]]),axis=2),axis = 2)/dxbu
            Y1xm += Y1x/nt

            Y2 = dvdtvisc
            Y2 = np.ma.filled(Y2.astype(float), 0)
            Y2x = np.diff(np.concatenate((Y2,Y2[:,:,[-1]]),axis=2),axis = 2)/dxbu
            Y2xm += Y2x/nt

            print((i+1)/nt*100)

        terms = np.concatenate((fuxm[:,:,:,np.newaxis],
                                fvym[:,:,:,np.newaxis],
                                Y1xm[:,:,:,np.newaxis],
                                Y2xm[:,:,:,np.newaxis],
                                vadvxm[:,:,:,np.newaxis]),axis=3)
        np.savez('pvterms', terms=terms, elm=elm, em=em, vm=vm, um=um)

    else:
        npzfile = np.load('pvterms.npz')
        terms = npzfile['terms']
        elm = npzfile['elm']

    res = np.sum(terms,axis=3)

    titl = ['hagum','hxm','-huwbm','-huuxpTm','-huvymTm','-ugmm']
    figa = ['(a)','(b)','(c)','(d)','(e)','(f)']
#    zs = 0
#    ze = nz
#    xstart = -1
#    xend = 0
#    ystart = 30
#    yend = 40
    (X,Y,epl,eplmn,eplmx),(xs,xe),(ys,ye) = getisopyc(em,xstart,xend,
                                                  ystart,yend,zs,ze,xh,yh,zi,meanax)
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(terms[:,:,:,0],elm,xstart,xend,
                                                  ystart,yend,zs,ze,xq,yh,zl,meanax)
    vmn = -np.amax(np.absolute(np.mean(terms[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    vmx = np.amax(np.absolute(np.mean(terms[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    Vctr = np.linspace(vmn,vmx,num=10,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    print(Vctr,Vcbar)
    plt.figure()
    for i in np.arange(5):
        ax = plt.subplot(3,2,i+1)
        (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(terms[:,:,:,i],elm,xstart,xend,
                                                      ystart,yend,zs,ze,xq,yh,zl,meanax)
        im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
        im2 = ax.contour(X,Y,epl,12,colors='k',linestyle='.',linewidth=0.5)
        cbar = plt.colorbar(im, ticks=Vcbar)
        ax.text(ax.get_xlim()[0]+0.1*(np.diff(ax.get_xlim())),
                ax.get_ylim()[1]-0.1*(np.diff(ax.get_ylim())),figa[i])
        cbar.formatter.set_powerlimits((-1, 1))
        cbar.update_ticks()
        ax.set_xticks([-70, -50, -30, -10])
        if (i+1) % 2 == 1:
            plt.ylabel('$z (m)$') 
        else:
            ax.set_yticklabels([])
        if np.in1d(i+1,[5,6]):
            plt.xlabel('$x (km)$')
        else:
            ax.set_xticklabels([])
   
    ax = plt.subplot(3,2,6)
    im = ax.contourf(X, Y, np.mean(res[zs:ze,ys:ye,xs:xe], axis=1),
                     Vctr, cmap=plt.cm.RdBu_r)
    im2 = ax.contour(X,Y,epl,12,colors='k')
    plt.xlabel('$x (km)$')
    ax.set_yticklabels([])
    ax.text(ax.get_xlim()[0]+0.1*(np.diff(ax.get_xlim())),
                ax.get_ylim()[1]-0.1*(np.diff(ax.get_ylim())),figa[5])
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-1, 1))
    cbar.update_ticks()
    ax.set_xticks([-70, -50, -30, -10])
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

