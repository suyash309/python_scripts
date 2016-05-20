import numpy as np

def plotrange(var,xstart,xend,ystart,yend,zstart,zend,x,y,z,elm,meanax):
    nx = len(x)
    ny = len(y)
    nz = len(z)
    xs = [i for i in range(nx) if x[i] >= xstart and x[i] <= xend][1] 
    xe = [i for i in range(nx) if x[i] >= xstart and x[i] <= xend][-1]
    ys = [i for i in range(ny) if y[i] >= ystart and y[i] <= yend][1] 
    ye = [i for i in range(ny) if y[i] >= ystart and y[i] <= yend][-1]
    zs = zstart
    ze = zend
    if meanax == 0:
            sx = 6400*np.cos(0.5*(ystart+yend)*np.pi/180)*x*np.pi/180
            sy = 6400*y*np.pi/180
            X, Y = np.meshgrid(sx[xs:xe],sy[ys:ye])
    elif meanax == 1:
            sx = 6400*np.cos(0.5*(ystart+yend)*np.pi/180)*x*np.pi/180
            X, Y = np.meshgrid(sx[xs:xe],z[zs:ze])
            Y = np.mean(elm[zs:ze,ys:ye,xs:xe], axis=meanax)
    else:
            sy = 6400*y*np.pi/180
            X, Y = np.meshgrid(sy[ys:ye],z[zs:ze])
            Y = np.mean(elm[zs:ze,ys:ye,xs:xe], axis=meanax)

    P = np.mean(var[zs:ze,ys:ye,xs:xe], axis=meanax)
    Pmn = -np.amax(np.absolute(P))
    Pmx = np.amax(np.absolute(P))
    return X,Y,P,Pmn,Pmx


def sliceDomain(var,elm,xstart,xend,ystart,yend,zstart,zend,x,y,z,meanax):
    nx = len(x)
    ny = len(y)
    nz = len(z)
    xs = [i for i in range(nx) if x[i] >= xstart and x[i] <= xend][1] 
    xe = [i for i in range(nx) if x[i] >= xstart and x[i] <= xend][-1]
    ys = [i for i in range(ny) if y[i] >= ystart and y[i] <= yend][1] 
    ye = [i for i in range(ny) if y[i] >= ystart and y[i] <= yend][-1]
    zs = zstart
    ze = zend
    if meanax == 0:
            sx = 6400*np.cos(0.5*(ystart+yend)*np.pi/180)*x*np.pi/180
            sy = 6400*y*np.pi/180
            X, Y = np.meshgrid(sx[xs:xe],sy[ys:ye])
    elif meanax == 1:
            sx = 6400*np.cos(0.5*(ystart+yend)*np.pi/180)*x*np.pi/180
            X, Y = np.meshgrid(sx[xs:xe],z[zs:ze])
            Y = np.mean(elm[zs:ze,ys:ye,xs:xe], axis=meanax)
    else:
            sy = 6400*y*np.pi/180
            X, Y = np.meshgrid(sy[ys:ye],z[zs:ze])
            Y = np.mean(elm[zs:ze,ys:ye,xs:xe], axis=meanax)

    P = np.mean(var[zs:ze,ys:ye,xs:xe], axis=meanax)
    Pmn = -np.amax(np.absolute(P))
    Pmx = np.amax(np.absolute(P))
    return (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye)


def getisopyc(e,xstart,xend,ystart,yend,zstart,zend,x,y,z,meanax):
    nx = len(x)
    ny = len(y)
    nz = len(z)
    xs = [i for i in range(nx) if x[i] >= xstart and x[i] <= xend][1] 
    xe = [i for i in range(nx) if x[i] >= xstart and x[i] <= xend][-1]
    ys = [i for i in range(ny) if y[i] >= ystart and y[i] <= yend][1] 
    ye = [i for i in range(ny) if y[i] >= ystart and y[i] <= yend][-1]
    zs = zstart
    ze = zend
    if meanax == 1:
            sx = 6400*np.cos(0.5*(ystart+yend)*np.pi/180)*x*np.pi/180
            X, Y = np.meshgrid(sx[xs:xe],z[zs:ze])
            Y = np.mean(e[zs:ze,ys:ye,xs:xe], axis=meanax)
            XX, P = np.meshgrid(sx[xs:xe],z[zs:ze])
    else:
            sy = 6400*y*np.pi/180
            X, Y = np.meshgrid(sy[ys:ye],z[zs:ze])
            Y = np.mean(e[zs:ze,ys:ye,xs:xe], axis=meanax)
            XX, P = np.meshgrid(sy[ys:ye],z[zs:ze])

    Pmn = -np.amax(np.absolute(P))
    Pmx = np.amax(np.absolute(P))
    return (X,Y,P,Pmn,Pmx),(xs,xe),(ys,ye)
