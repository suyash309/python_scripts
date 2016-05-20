import numpy as np
from netCDF4 import Dataset as dset
from netCDF4 import MFDataset as mfdset

def getgeom(filename):
    # This fuction returns the depth of the domain D, cell areas at 
    # T and Bu points ah and aq, and grid spacing at Cu, Cv, Bu and T points
    fhgeo = dset(filename, mode='r')
    D = fhgeo.variables['D'][:]
    ah = fhgeo.variables['Ah'][:]
    aq = fhgeo.variables['Aq'][:]
    dxcu = fhgeo.variables['dxCu'][:]
    dycu = fhgeo.variables['dyCu'][:]
    dxcv = fhgeo.variables['dxCv'][:]
    dycv = fhgeo.variables['dyCv'][:]
    dxbu = fhgeo.variables['dxBu'][:]
    dybu = fhgeo.variables['dyBu'][:]
    dxt = fhgeo.variables['dxT'][:]
    dyt = fhgeo.variables['dyT'][:]
    fhgeo.close()
    return D, (ah,aq), (dxcu,dycu,dxcv,dycv,dxbu,dybu,dxt,dyt)

def getdims(filename):
    fh = mfdset(filename)
    xq = fh.variables['xq'][:]
    yq = fh.variables['yq'][:]
    time = fh.variables['Time'][:]
    xh = fh.variables['xh'][:]
    yh = fh.variables['yh'][:]
    zi = fh.variables['zi'][:]
    zl = fh.variables['zl'][:]
    fh.close()
    return (xh,yh), (xq,yq), (zi,zl), time

def getuvhe(filename,i):
    fh = mfdset(filename)
    h = fh.variables['h'][i,:,:,:] 
    e = fh.variables['e'][i,:,:,:] 
    el = (e[0:-1,:,:] + e[1:,:,:])/2;
    u = fh.variables['u'][i,:,:,:]
    v = fh.variables['v'][i,:,:,:]
    fh.close()
    return (u,v,h,e,el)

def gethatuv(filename,i):
    fh = mfdset(filename)
    frhatu = fh.variables['frhatu'][i,:,:,:] 
    frhatv = fh.variables['frhatv'][i,:,:,:] 
    fh.close()
    return (frhatu,frhatv)

def getPVRV(filename,i):
    fh = mfdset(filename)
    PV = fh.variables['PV'][i,:,:,:]
    RV = fh.variables['RV'][i,:,:,:]
    fh.close()
    return (PV,RV)   

def getmomxterms(filename,i):
    fh = mfdset(filename)
    dudt = fh.variables['dudt'][i,:,:,:]
    cau = fh.variables['CAu'][i,:,:,:]
    pfu = fh.variables['PFu'][i,:,:,:]
    dudtvisc = fh.variables['du_dt_visc'][i,:,:,:]
    diffu = fh.variables['diffu'][i,:,:,:]
    dudtdia = fh.variables['dudt_dia'][i,:,:,:]
    gkeu = fh.variables['gKEu'][i,:,:,:]
    rvxv = fh.variables['rvxv'][i,:,:,:]
    fh.close()
    return (dudt,cau,pfu,dudtvisc,diffu,dudtdia,gkeu,rvxv)

def getmomyterms(filename,i):
    fh = mfdset(filename)
    dvdt = fh.variables['dvdt'][i,:,:,:]
    cav = fh.variables['CAv'][i,:,:,:]
    pfv = fh.variables['PFv'][i,:,:,:]
    dvdtvisc = fh.variables['dv_dt_visc'][i,:,:,:]
    diffv = fh.variables['diffv'][i,:,:,:]
    dvdtdia = fh.variables['dvdt_dia'][i,:,:,:]
    gkev = fh.variables['gKEv'][i,:,:,:]
    rvxu = fh.variables['rvxu'][i,:,:,:]
    fh.close()
    return (dvdt,cav,pfv,dvdtvisc,diffv,dvdtdia,gkev,rvxu)   

def getcontterms(filename,i):
    fh = mfdset(filename)
    dhdt = fh.variables['dhdt'][i,:,:,:]
    uh = fh.variables['uh'][i,:,:,:]
    vh = fh.variables['vh'][i,:,:,:]
    wd = fh.variables['wd'][i,:,:,:]
    uhgm = fh.variables['uhGM'][i,:,:,:]
    vhgm = fh.variables['vhGM'][i,:,:,:]
    fh.close()
    return (dhdt,uh,vh,wd,uhgm,vhgm)   

def getoceanstats(filename):
    fh = mfdset(filename)
    layer = fh.variables['Layer'][:]
    interface = fh.variables['Interface'][:]
    time = fh.variables['Time'][:]
    en = fh.variables['En'][:]
    ape = fh.variables['APE'][:]
    ke = fh.variables['KE'][:]
    maxcfltrans = fh.variables['max_CFL_trans'][:]
    maxcfllin = fh.variables['max_CFL_lin'][:]
    return (layer,interface,time), (en,ape,ke), (maxcfltrans,maxcfllin)
