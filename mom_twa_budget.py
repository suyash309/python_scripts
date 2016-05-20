import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
import matplotlib as mpl
import readparams as rdp
from ploteb import sliceDomain,getisopyc

#mpl.rcParams.update({'font.weight':'bold'})

def mom_twa_x(firstrun,geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,savfil1=None,savfil2=None):
    
    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp.getgeom(geofil)
    
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)
    
    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)
    if firstrun == 1:
        hm_cu = np.zeros((nz,ny,nx))
        hm_cv = np.zeros((nz,ny,nx))
        hm = np.zeros((nz,ny,nx))
        em = np.zeros((nzp1,ny,nx))
        elm = np.zeros((nz,ny,nx))
        um = np.zeros((nz,ny,nx))
        vm = np.zeros((nz,ny,nx))
        hagum = np.zeros((nz,ny,nx))
        hxm = np.zeros((nz,ny,nx))
        huwbm = np.zeros((nz,ny,nx))
        huuxpTm = np.zeros((nz,ny,nx))
        huvymTm = np.zeros((nz,ny,nx))
        ugmm = np.zeros((nz,ny,nx))
        
        
        for i in range(nt):
            print(i/nt*100)
            (u,v,h,e,el) = rdp.getuvhe(fil,i)
            (frhatu,frhatv) = rdp.gethatuv(fil,i)
            hm += h/nt
            em += e/nt
            elm += el/nt
            um += u/nt
            vm += v/nt
            h_cu = frhatu*D
            h_cv = frhatv*D
            hm_cu += h_cu/nt
            hm_cv += h_cv/nt
        
            (dudt,cau,pfu,dudtvisc,diffu,dudtdia,gkeu,rvxv) = rdp.getmomxterms(fil,i)
            (dhdt,uh,vh,wd,uhgm,vhgm) = rdp.getcontterms(fil,i)

            hagu = h_cu*(cau - gkeu - rvxv + pfu)
            hagum += hagu/nt
             
            hx = h_cu*(diffu + dudtvisc)
            hxm += hx/nt
             
            dwd = (wd[1:,:,:] - wd[0:-1,:,:])/2;
            huwb = u*(dwd + np.roll(dwd,-1,axis=2))/2 - h_cu*dudtdia
            huwbm += huwb/nt
        
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(np.concatenate((uh[:,:,-1:],uh),axis=2),axis = 2)/ah
            huuxpT = u*(uhx + np.roll(uhx,-1,axis=2))/2 - h_cu*gkeu     
            huuxpTm += huuxpT/nt
        
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(np.concatenate((vh[:,-1:,:],vh),axis=1),axis = 1)/ah
            huvymT = u*(vhy + np.roll(vhy,-1,axis=2))/2 - h_cu*rvxv     
            huvymTm += huvymT/nt
        
            uhgm = np.ma.filled(uhgm.astype(float), 0)
            uhgmx = np.diff(np.dstack((uhgm[:,:,-1:],uhgm)),axis = 2)/ah
            vhgm = np.ma.filled(vhgm.astype(float), 0)
            vhgmy = np.diff(np.hstack((vhgm[:,-1:,:],vhgm)),axis = 1)/ah
            gm = uhgmx + vhgmy
            ugm = u*(gm + np.roll(gm,-1,axis=2))/2
            ugmm += ugm/nt
        
        
        termsx = np.concatenate((hagum[:,:,:,np.newaxis],
                                   hxm[:,:,:,np.newaxis],
                                -huwbm[:,:,:,np.newaxis],
                              -huuxpTm[:,:,:,np.newaxis],
                              -huvymTm[:,:,:,np.newaxis],
                                 -ugmm[:,:,:,np.newaxis]),axis=3)
    
        np.savez('twamomx', termsx=termsx, elm=elm, em=em, vm=vm, um=um)
    else:
        npzfile = np.load('twamomx.npz')
        termsx = npzfile['termsx']
        elm = npzfile['elm']
        em = npzfile['em']
    
    res = np.sum(termsx,axis=3)

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
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(termsx[:,:,:,0],elm,xstart,xend,
                                                  ystart,yend,zs,ze,xq,yh,zl,meanax)
    vmn = -np.amax(np.absolute(np.mean(termsx[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    vmx = np.amax(np.absolute(np.mean(termsx[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    Vctr = np.linspace(vmn,vmx,num=10,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    print(Vctr,Vcbar)
    plt.figure()
    for i in np.arange(6):
        ax = plt.subplot(3,2,i+1)
        (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(termsx[:,:,:,i],elm,xstart,xend,
                                                      ystart,yend,zs,ze,xq,yh,zl,meanax)
        im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
        im2 = ax.contour(X,Y,epl,12,colors='k')
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
    
    if savfil1:
        plt.savefig(savfil1+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
        
    termsxbetter = np.concatenate((termsx[:,:,:,[0]],
                                   termsx[:,:,:,[1]],
                                   termsx[:,:,:,[2]],
                                   termsx[:,:,:,[3]]+termsx[:,:,:,[4]],
                                   termsx[:,:,:,[5]]),axis=3)
 
    vmn = -np.amax(np.absolute(np.mean(termsxbetter[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    vmx = np.amax(np.absolute(np.mean(termsxbetter[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    Vctr = np.linspace(vmn,vmx,num=10,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    print(Vctr,Vcbar)
    plt.figure()
    for i in np.arange(5):
        ax = plt.subplot(3,2,i+1)
        (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(termsxbetter[:,:,:,i],elm,xstart,xend,
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
    
   
    #plt.figure()
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
    if savfil2:
        plt.savefig(savfil2+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()


def mom_twa_y(firstrun,geofil,fil,xstart,xend,ystart,yend,zs,ze,meanax,savfil1=None,savfil2=None):
    
    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp.getgeom(geofil)
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)
    
    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)
    
    if firstrun == 1:
        hm_cu = np.zeros((nz,ny,nx))
        hm_cv = np.zeros((nz,ny,nx))
        hm = np.zeros((nz,ny,nx))
        em = np.zeros((nzp1,ny,nx))
        elm = np.zeros((nz,ny,nx))
        um = np.zeros((nz,ny,nx))
        vm = np.zeros((nz,ny,nx))
        hagvm = np.zeros((nz,ny,nx))
        hym = np.zeros((nz,ny,nx))
        hvwbm = np.zeros((nz,ny,nx))
        huvxpTm = np.zeros((nz,ny,nx))
        hvvymTm = np.zeros((nz,ny,nx))
        vgmm = np.zeros((nz,ny,nx))
        
        
        for i in range(nt):
            print(i/nt*100)
            (u,v,h,e,el) = rdp.getuvhe(fil,i)
            (frhatu,frhatv) = rdp.gethatuv(fil,i)
            hm += h/nt
            em += e/nt
            elm += el/nt
            um += u/nt
            vm += v/nt
            h_cu = frhatu*D
            h_cv = frhatv*D
            hm_cu += h_cu/nt
            hm_cv += h_cv/nt
        
            (dvdt,cav,pfv,dvdtvisc,diffv,dvdtdia,gkev,rvxu) = rdp.getmomyterms(fil,i)
            (dhdt,uh,vh,wd,uhgm,vhgm) = rdp.getcontterms(fil,i)

            hagv = h_cv*(cav - gkev - rvxu + pfv)
            hagvm += hagv/nt
             
            hy = h_cv*(diffv + dvdtvisc)
            hym += hy/nt
             
            dwd = (wd[1:,:,:] - wd[0:-1,:,:])/2;
            hvwb = v*(dwd + np.roll(dwd,-1,axis=1))/2 - h_cv*dvdtdia
            hvwbm += hvwb/nt
        
            uh = np.ma.filled(uh.astype(float), 0)
            uhx = np.diff(np.concatenate((uh[:,:,-1:],uh),axis=2),axis = 2)/ah
            huvxpT = v*(uhx + np.roll(uhx,-1,axis=1))/2 - h_cv*rvxu     
            huvxpTm += huvxpT/nt
        
            vh = np.ma.filled(vh.astype(float), 0)
            vhy = np.diff(np.concatenate((vh[:,-1:,:],vh),axis=1),axis = 1)/ah
            hvvymT = v*(vhy + np.roll(vhy,-1,axis=1))/2 - h_cv*gkev     
            hvvymTm += hvvymT/nt
        
            uhgm = np.ma.filled(uhgm.astype(float), 0)
            uhgmx = np.diff(np.dstack((uhgm[:,:,-1:],uhgm)),axis = 2)/ah
            vhgm = np.ma.filled(vhgm.astype(float), 0)
            vhgmy = np.diff(np.hstack((vhgm[:,-1:,:],vhgm)),axis = 1)/ah
            gm = uhgmx + vhgmy
            vgm = v*(gm + np.roll(gm,-1,axis=1))/2
            vgmm += vgm/nt
        
        
        termsy = np.concatenate((hagvm[:,:,:,np.newaxis],
                                   hym[:,:,:,np.newaxis],
                                -hvwbm[:,:,:,np.newaxis],
                              -huvxpTm[:,:,:,np.newaxis],
                              -hvvymTm[:,:,:,np.newaxis],
                                 -vgmm[:,:,:,np.newaxis]),axis=3)
        
        np.savez('twamomy', termsy=termsy, elm=elm, em=em, vm=vm, um=um)
    
    else:
        npzfile = np.load('twamomy.npz')
        termsy = npzfile['termsy']
        elm = npzfile['elm']
        em = npzfile['em']
    
    res = np.sum(termsy,axis=3)
    
    titl = ['hagum','hym','-huwbm','-huuxpTm','-huvymTm','-ugmm']
    figa = ['(a)','(b)','(c)','(d)','(e)','(f)']
#    zs = 0
#    ze = nz
#    xstart = -1
#    xend = 0
#    ystart = 30
#    yend = 40
    (X,Y,epl,eplmn,eplmx),(xs,xe),(ys,ye) = getisopyc(em,xstart,xend,
                                              ystart,yend,zs,ze,xh,yh,zi,meanax)
    (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(termsy[:,:,:,0],elm,xstart,xend,
                                                  ystart,yend,zs,ze,xh,yh,zl,meanax)

    vmn = -np.amax(np.absolute(np.mean(termsy[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    vmx = np.amax(np.absolute(np.mean(termsy[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    Vctr = np.linspace(vmn,vmx,num=10,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    plt.figure()
    for i in np.arange(6):
        ax = plt.subplot(3,2,i+1)
        (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(termsy[:,:,:,i],elm,xstart,xend,
                                                  ystart,yend,zs,ze,xh,yh,zl,meanax)
        im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
        im2 = ax.contour(X,Y,epl,12,colors='k')
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
    
    if savfil1:
        plt.savefig(savfil1+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
    
    #plt.savefig('twamomy.eps', dpi=300, facecolor='w', edgecolor='w', format='eps', transparent=False, bbox_inches='tight')
    termsybetter = np.concatenate((termsy[:,:,:,[0]],
                                   termsy[:,:,:,[1]],
                                   termsy[:,:,:,[2]],
                                   termsy[:,:,:,[3]]+termsy[:,:,:,[4]],
                                   termsy[:,:,:,[5]]),axis=3)
 
    vmn = -np.amax(np.absolute(np.mean(termsybetter[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    vmx = np.amax(np.absolute(np.mean(termsybetter[zs:ze,ys:ye,xs:xe,:], axis=meanax)))
    Vctr = np.linspace(vmn,vmx,num=10,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    print(Vctr,Vcbar)
    plt.figure()
    for i in np.arange(5):
        ax = plt.subplot(3,2,i+1)
        (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye) = sliceDomain(termsybetter[:,:,:,i],elm,xstart,xend,
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
    
   
    #plt.figure()
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
    if savfil2:
        plt.savefig(savfil2+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

