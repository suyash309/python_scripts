import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
import matplotlib as mpl
import readparams as rdp

mpl.rcParams.update({'font.weight':'bold'})

def pvbudget(firstrun,geofil,fil,savfil=None):
#def pvbudget(firstrun,savfil=None):
#    geofil = ("/home/sbire/MOM6-examples/ocean_only/buoy_forced_gyre_highfluxc"
#              "_2km_fromut_kvkd1em4_kh1e3_khth1e3_lowbz_smd_veryhires_3yr/ocean_geometry.nc")
#    fil = ("/home/sbire/MOM6-examples/ocean_only/buoy_forced_gyre_highfluxc"
#              "_2km_fromut_kvkd1em4_kh1e3_khth1e3_lowbz_smd_veryhires_3yr/output__0031_07.nc")
    
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
        msigpiuxm = np.zeros((nz,ny,nx))
        Yxm = np.zeros((nz,ny,nx))
        wvbxm = np.zeros((nz,ny,nx))
        sigpivym = np.zeros((nz,ny,nx))
        Xym = np.zeros((nz,ny,nx))
        wubym = np.zeros((nz,ny,nx))
        #nt = 2
        
        fh = dset(fil, mode='r')
        for i in range(nt):
            e = fh.variables['e'][i,:,:,:] 
            el = (e[0:-1,:,:] + e[1:,:,:])/2;
            u = fh.variables['u'][i,:,:,:]
            v = fh.variables['v'][i,:,:,:]
            frhatv = fh.variables['frhatv'][i,:,:,:] 
            elm += el/nt
            um += u/nt
            vm += v/nt
        
            cav = fh.variables['CAv'][i,:,:,:]
            gkev = fh.variables['gKEv'][i,:,:,:]
            msigpiu = cav - gkev
            msigpiu = np.ma.filled(msigpiu.astype(float), 0)
            msigpiux = np.diff(np.concatenate((msigpiu,msigpiu[:,:,[-1]]),axis=2),axis = 2)/dxBu
        
            Y = fh.variables['diffv'][i,:,:,:] + fh.variables['dv_dt_visc'][i,:,:,:]
            Y = np.ma.filled(Y.astype(float), 0)
            Yx = np.diff(np.concatenate((Y,Y[:,:,[-1]]),axis=2),axis = 2)/dxBu
           
            wvb = fh.variables['dvdt_dia'][i,:,:,:]
            wvb = np.ma.filled(wvb.astype(float), 0)
            wvbx = np.diff(np.concatenate((wvb,wvb[:,:,[-1]]),axis=2),axis = 2)/dxBu
        
            cau = fh.variables['CAu'][i,:,:,:]
            gkeu = fh.variables['gKEu'][i,:,:,:]
            sigpiv = cau - gkeu
            sigpiv = np.ma.filled(sigpiv.astype(float), 0)
            sigpivy = np.diff(np.concatenate((sigpiv,sigpiv[:,[-1],:]),axis=1),axis = 1)/dyBu
        
            X = fh.variables['diffu'][i,:,:,:] + fh.variables['du_dt_visc'][i,:,:,:]
            X = np.ma.filled(X.astype(float), 0)
            Xy = np.diff(np.concatenate((X,X[:,[-1],:]),axis=1),axis = 1)/dyBu
           
            wub = fh.variables['dudt_dia'][i,:,:,:]
            wub = np.ma.filled(wub.astype(float), 0)
            wuby = np.diff(np.concatenate((wub,wub[:,[-1],:]),axis=1),axis = 1)/dyBu
        
            msigpiuxm += msigpiux/nt
            Yxm += Yx/nt
            wvbxm += wvbx/nt
            sigpivym += sigpivy/nt
            Xym += Xy/nt
            wubym += wuby/nt
        
            print((i+1)/nt*100)
        
        fh.close()
        
        terms = np.concatenate((msigpiuxm[:,:,:,np.newaxis],
                                      Yxm[:,:,:,np.newaxis],
                                    wvbxm[:,:,:,np.newaxis],
                                -sigpivym[:,:,:,np.newaxis],
                                     -Xym[:,:,:,np.newaxis],
                                   -wubym[:,:,:,np.newaxis]),axis=3)
        
        np.savez('PVterms', terms=terms, elm=elm, em=em, vm=vm, um=um)
    else:
        npzfile = np.load('PVterms.npz')
        terms = npzfile['terms']
        elm = npzfile['elm']
    
    res = np.sum(terms,axis=3)
    
    xlen = 10
    ystart = 30
    yend = 40
    figa = ['(a)','(b)','(c)','(d)','(e)','(f)']
    xs = nx-xlen
    xe = nx
    ys = [i for i in range(ny) if yq[i] >= ystart and yq[i] <= yend][1] 
    ye = [i for i in range(ny) if yq[i] >= ystart and yq[i] <= yend][-1]
    zs = 0
    ze = nz
    s = 6400*np.cos(0.5*(ystart+yend)*np.pi/180)*xq*np.pi/180
    X, Y = np.meshgrid(s[xs:xe],zl[zs:ze])
    Y_dummy = np.mean(elm[zs:ze,ys:ye,xs:xe], axis=1)
    
    #termsy *= 1e4
    #res *= 1e4
    vmn = -np.amax(np.absolute(np.mean(terms[zs:ze,ys:ye,xs:xe,:], axis=1)))
    vmx = np.amax(np.absolute(np.mean(terms[zs:ze,ys:ye,xs:xe,:], axis=1)))
    #Vctr = np.linspace(np.floor(vmn),np.ceil(vmx),num=12,endpoint=True)
    #Vctr =  np.linspace(-5e-4,5e-4,num=12,endpoint=True)
    Vctr = np.linspace(vmn,vmx,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    #Vcbar = np.arange(Vctr[1],Vctr[-1],1,'int32')
    print(Vctr,Vcbar)
    plt.figure()
    for i in np.arange(6):
        ax = plt.subplot(3,2,i+1)
        P = np.mean(terms[zs:ze,ys:ye,xs:xe,i], axis=1)
        im = ax.contourf(X, Y_dummy, P, Vctr, cmap=plt.cm.RdBu_r)
        cbar = plt.colorbar(im, ticks=Vcbar)
        ax.text(-61,-300,figa[i])
        cbar.formatter.set_powerlimits((-1, 1))
        cbar.update_ticks()
        if (i+1) % 2 == 1:
            plt.ylabel('$z (m)$') 
        else:
            plt.yticks([0,-500,-1000,-1500,-2000],[])
        if np.in1d(i+1,[5,6]):
            plt.xlabel('$x (km)$')
            plt.xticks([-50,-30,-10],[-50,-30,-10])
        else:
            plt.xticks([-50,-30,-10],[-50,-30,-10])
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()
    
    #plt.savefig('twamomy.eps', dpi=300, facecolor='w', edgecolor='w', format='eps', transparent=False, bbox_inches='tight')
    
    plt.figure()
    im = plt.contourf(X, Y_dummy, np.mean(res[zs:ze,ys:ye,xs:xe], axis=1), Vctr, cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-1, 1))
    cbar.update_ticks()
    plt.show()
