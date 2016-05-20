import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset
import matplotlib as mpl
import readparams as rdp
from ploteb import *
import matplotlib.animation as animation

mpl.rcParams.update({'font.weight':'bold'})
def plotoceanstats(fil,savfil=None):
    (layer,interface,time), (en,ape,ke), (maxcfltrans,maxcfllin) = rdp.getoceanstats(fil)
    time /= 365
    ax =plt.subplot(3,1,1)
    im1 = plt.plot(time,maxcfltrans)
    ax.set_ylabel('Max CFL')
    plt.tick_params(axis='x',labelbottom='off')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    ax = plt.subplot(3,1,2)
    im1 = plt.plot(time,np.mean(ape,axis=1))
    ax.set_ylabel('APE (J)')
    plt.tick_params(axis='x',labelbottom='off')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    ax = plt.subplot(3,1,3)
    im1 = plt.plot(time,np.mean(ke,axis=1))
    plt.xlabel('Time (years)')
    ax.set_ylabel('KE (J)')
    plt.grid()
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()

def plotvel(geofil,fil,savfil=None):

    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp.getgeom(geofil)
    
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)
    
    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)
    
    hm = np.zeros((nz,ny,nx))
    em = np.zeros((nzp1,ny,nx))
    elm = np.zeros((nz,ny,nx))
    um = np.zeros((nz,ny,nx))
    vm = np.zeros((nz,ny,nx))
    
    for i in range(nt):
        (u,v,h,e,el) = rdp.getuvhe(fil,i)
        um += u/nt
        vm += v/nt
        em += e/nt
        elm += el/nt
        print((i+1)/nt*100)
    
    
    xlim = [-1,0]     # actual values of x in degrees
    ylim = [30,40]    # actual values of y in degrees
    zlim = [0,10]     # indices of zl or zi
    
    plt.figure()
    
    ax = plt.subplot(1,2,1)
    X,Y,P,Pmn,Pmx = plotrange(um,xlim[0],xlim[-1],
                                 ylim[0],ylim[-1],
                                 zlim[0],zlim[-1],xq,yh,zl,elm,0)
    Vctr = np.linspace(Pmn,Pmx,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
    
    ax = plt.subplot(1,2,2)
    X,Y,P,Pmn,Pmx = plotrange(vm,xlim[0],xlim[-1],
                                 ylim[0],ylim[-1],
                                 zlim[0],zlim[-1],xq,yh,zl,elm,0)
    Vctr = np.linspace(Pmn,Pmx,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(im, ticks=Vcbar)
    cbar.formatter.set_powerlimits((-3, 4))
    cbar.update_ticks()
    
    if savfil:
        plt.savefig(savfil+'.eps', dpi=300, facecolor='w', edgecolor='w', 
                    format='eps', transparent=False, bbox_inches='tight')
    else:
        plt.show()


def plotsshanim(geofil,fil,desfps,savfil=None):
       
    D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt) = rdp.getgeom(geofil)
    
    (xh,yh), (xq,yq), (zi,zl), time = rdp.getdims(fil)
    
    print(len(time))

    nx = len(xh)
    ny = len(yh)
    nz = len(zl)
    nzp1 = len(zi)
    nt = len(time)
 
    fh = mfdset(fil)
    e1 = fh.variables['e'][:,0,:,:]
    emax = np.ceil(np.amax(np.abs(e1)))
    e = fh.variables['e'][0,:,:,:] 
    fh.close()
     
    xlim = [-25,0]     # actual values of x in degrees
    ylim = [10,50]    # actual values of y in degrees
    zlim = [0,1]     # indices of zl or zi
  
    X,Y,P,Pmn,Pmx = plotrange(e,xlim[0],xlim[-1],
                                ylim[0],ylim[-1],
                                zlim[0],zlim[-1],xh,yh,zi,e,0)
    fig = plt.figure()
    ax = plt.axes()
    Vctr = np.linspace(-emax,emax,num=12,endpoint=True)
    Vcbar = (Vctr[1:] + Vctr[:-1])/2
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar(im, ticks=Vcbar)
    ani = animation.FuncAnimation(fig,update_contour_plot,frames=range(nt),
            fargs=(fil,ax,fig,xlim,ylim,zlim,xh,yh,zi,0))
    if savfil:
        ani.save(savfil+'.mp4', writer="avconv", fps=desfps, 
                extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()
    plt.close(fig)

def update_contour_plot(i,fil,ax,fig,xlim,ylim,zlim,x,y,z,meanax):
    fh = mfdset(fil)
    var = fh.variables['e'][i,:,:,:] 
    fh.close()
    X,Y,P,Pmn,Pmx = plotrange(var,xlim[0],xlim[-1],
                                  ylim[0],ylim[-1],
                                  zlim[0],zlim[-1],x,y,z,var,meanax)
    ax.cla()
    Vctr = np.linspace(-0.6,0.6,num=12,endpoint=True)
    im = ax.contourf(X, Y, P, Vctr, cmap=plt.cm.RdBu_r)
    plt.title(str(i))
    print(i)
    return im,
