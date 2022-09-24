import numpy as np
from scipy.interpolate import griddata
import datetime
from matplotlib import pyplot as plt
from numpy.linalg import svd
from copy import copy
from functools import partial
from sklearn.neighbors import KernelDensity as KDE
from scipy.stats import chi2, poisson
import pandas as pd
from time import time
from scipy.special import lambertw
from scipy.optimize import differential_evolution

MC=1.0          # minimum magnitude to use
RMIN=1.e-8

# key input files
CATALOG_FILE=r'KNMI eventcatalogue 10 feb 2022.xlsx'
PRESSURE_FILE=r'XY_PRF_Avg_OS2_CY.csv'
FAULT_FILE=r'faults_grl.txt'  

class Catalog(object):                      # CLASS for earthquake catalog
    def __init__(self):
        self.t=None
        self.x=None
        self.y=None
        self.m=None
    def has_events(self, t, x=None, y=None, m=None):
        ''' Associate events with the catalog.
        '''
        self.t=t
        self.x=x
        self.y=y
        self.m=m
class Pressure(object):						# CLASS for pressure data
    def __init__(self, file, parent=None):							# initialize object
        self.parent=parent
        self.file=file
        self.read()
    def read(self):										
        ''' reads a prsesure file
        '''
        fp=open(self.file,'r')
        ln=fp.readline().strip()
        fp.close()
        v=ln.split(',')[2:]
        self.t=np.array([int(float(vi.split('_')[0][1:])) for vi in v])
        
        # load in raw data file
        dat=np.genfromtxt(self.file, skip_header=1, delimiter=',')
        self.x2=dat[:,0]/1.e3
        self.y2=dat[:,1]/1.e3
        self.x=np.unique(self.x2)
        self.nx=len(self.x)
        self.y=np.unique(self.y2)
        self.ny=len(self.y)
        self.xx,self.yy=np.meshgrid(self.x,self.y,indexing='ij')
        self.p=34.*np.ones((self.t.shape[0], self.nx, self.ny))
        
        # loop through all rows, assigning to indices
        k=0 	# row
        for i in range(self.nx):
            xi=self.x[i]
            for j in range(self.ny):
                yj=self.y[j]
                if abs(xi-self.x2[k])<1.e-2 and abs(yj-self.y2[k])<1.e-2:
                    self.p[:,i,j]=dat[k,2:]/10. # convert bar to MPa
                    k += 1
                if k == dat.shape[0]:
                    break
            if k == dat.shape[0]:
                    break            
        self.dpdt=finite_difference(self.p)
        self.dp=self.p-np.broadcast_to(self.p[0,:,:], self.p.shape)
        self.p0=self.p[0,:,:]
    def projectPressure(self):
        nt=len(self.t)
        points=np.array([(self.xx).flatten(), (self.yy).flatten()]).T

        self.load=self.dp
        values=[self.load[i,:,:].flatten() for i in range(self.t.shape[0])]        
        
        # construct large vector of fault coordinates (for single interpolation)
        Nx=0
        for flt in self.parent.faults:
            Nx += flt.nx
            # empty load vector that will be populated during interpolation
            flt.dp=np.zeros((nt,flt.nx))
            flt.t=copy(self.t)
        ptsi=np.zeros((Nx,2))
        i=0
        for flt in self.parent.faults:
            ptsi[i:i+flt.nx,0]=flt.xv
            ptsi[i:i+flt.nx,1]=flt.yv
            i += flt.nx
        # interpolate faults for each year
        for i in range(nt):
            # perform interpolation using values data from that year
            dpi=griddata(points, values[i], ptsi, method='nearest')
            # assign interpolated values to fault objects
            j=0
            for flt in self.parent.faults:
                flt.dp[i,:]=dpi[j:j+flt.nx]
                j += flt.nx
        
        # set initial fault pressure
        p0i=griddata(points, self.p0.flatten(), ptsi, method='nearest')
        j=0
        for flt in self.parent.faults:
            flt.p0=p0i[j:j+flt.nx]
            j += flt.nx
class Tectonic(object):						# CLASS for tectonic stress state
    """
        Parameters
        ----------
        principal : list
            three item list containing the principal stresses (MPa)
        theta : float
            angle (degrees) between the maximum horizontal principal stress and North 
            (rotated clockwise)
        s1_vertical : bool, optional
            flag indicating the the vertical stress is the maximum principal
        s3_vertical : bool, optional
            flag indicating the the vertical stress is the minimum principal
        
        Notes
        -----
        It is assumed that the positive y-axis is aligned with North
    """
    def __init__(self, principal, theta, s_vertical, parent=None):
        self.parent=parent
        assert len(principal) == 3
        self.principal=np.sort(principal)
        self.s3,self.s2,self.s1=self.principal  # absolute stresses
        self.s0=np.zeros((3,3))
        
        if s_vertical == 1:
            self.s0[2,2]=self.s1
            self.sv, self.sH, self.sh=self.s1, self.s2, self.s3
        elif s_vertical == 3:
            self.s0[2,2]=self.s3
            self.sv, self.sH, self.sh=self.s3, self.s1, self.s2
        elif s_vertical == 2:
            self.s0[2,2]=self.s2
            self.sv, self.sH, self.sh=self.s2, self.s1, self.s3
        else:
            raise TypeError("unrecognized value '%i' for s_vertical: must be 1, 2 or 3"%s_vertical)
            
        # compute and rotate horizontal components
        # theta=rotation of rift axis clockwise from x-axis
        ct=np.cos(theta/180.*np.pi)
        st=np.sin(theta/180.*np.pi)
        self.R=np.array([[ct, -st],[st, ct]])
        self.s0[0,0]=self.sh
        self.s0[1,1]=self.sH
        self.s0[:2,:2]=np.matmul(self.s0[:2,:2], self.R)
        self.s0[:2,:2]=np.matmul(self.R.T, self.s0[:2,:2])
        dip=self.parent.faults[0].dip/180.*np.pi
        self.n=np.array([
            np.sin(dip)*self.R[0,0],
            np.sin(dip)*self.R[1,0],
            np.cos(dip)])
    def resolve(self, n):
        # resolve stresses on fault with normal n
        assert len(n) == 3
        n=np.array(n)
        n=n / np.sqrt(np.dot(n,n))
        nx,ny,nz=n
        sxx=self.s0[0,0]
        syy=self.s0[1,1]
        szz=self.s0[2,2]
        sxy=self.s0[0,1]
        syz=self.s0[1,2]
        sxz=self.s0[0,2]
        
        tx=nx*sxx+ny*sxy+nz*sxz
        ty=nx*sxy+ny*syy+nz*syz
        tz=nx*sxz+ny*syz+nz*szz
        tm=np.sqrt(tx**2+ty**2+tz**2)
        
        sn0=nx*tx+ny*ty+nz*tz
        
        if abs(sn0-tm)< 1.e-6: 
            tau0=0.
        else:
            tau0=np.sqrt(tm**2-sn0**2)
        
        return sn0, tau0
    def resolve_changes(self, dp, n, A):
        # resolve stresses on fault with normal n
        assert len(n) == 3
        n=np.array(n)
        n=n/np.sqrt(np.dot(n,n))
        nx,ny,nz=n
        
        # for the vector dp, compute the stress state
        sxx=self.s0[0,0]+A*dp
        syy=self.s0[1,1]+A*dp
        szz=self.s0[2,2]+A*dp*0.
        sxy=self.s0[0,1]+A*dp*0.
        syz=self.s0[1,2]+A*dp*0.
        sxz=self.s0[0,2]+A*dp*0.
        
        # compute traction on the fault
        tx=nx*sxx+ny*sxy+nz*sxz
        ty=nx*sxy+ny*syy+nz*syz
        tz=nx*sxz+ny*syz+nz*szz
        tm=np.sqrt(tx**2+ty**2+tz**2)
        
        # compute normal stress on the fault
        sn=nx*tx+ny*ty+nz*tz
        
        # compute shear stress on the fault
        tau=np.sqrt(tm**2-sn**2)
        tau[np.where(abs(sn-tm)< 1.e-6)]=0.
        
        return sn, tau
    def projectStress(self):
        # resolve stress conditions on a dictionary of fault objects
        self.sn0, self.tau0=self.resolve(self.n)
        for flt in self.parent.faults:
            flt.sn0, flt.tau0=self.resolve(flt.n)
            flt.fr=self.parent.f
            flt.hsr=self.parent.H
    def computeLoads(self):
        # compute loads and loading rates for optimally oriented fault across domain            
        self.sn, self.tau=self.resolve_changes(self.parent.pressure.load, self.n, self.parent.A)
        self.dsn=self.sn-self.sn0
        self.dtau=self.tau-self.tau0
        self.dsndt=finite_difference(self.sn)
        self.dtaudt=finite_difference(self.tau)
        
        # assign interpolated values to fault objects
        for flt in self.parent.faults:
            # output total normal stress and shear stress for dp
            flt.sn, flt.tau=self.resolve_changes(flt.dp, flt.n, self.parent.A)
            # change in total normal stress=final-initial
            flt.dsn=flt.sn-flt.sn0
            flt.dtau=flt.tau-flt.tau0
class Fault(object):                        # CLASS for fault structure
    def __init__(self, x, y, dip, dx):
        assert len(x) == len(y)
        self.x=x 					# x position coord
        self.y=y					# y position coord
        self.dip=dip				# dip angle
        self.compute_geometry()		# appproximate the fault trace by a straight line
        self.nx=int(self.l0/dx)
        self.xv=np.linspace(*self.x, self.nx)
        self.yv=np.linspace(*self.y, self.nx)
    def __repr__(self):
        return "flt:"+self.name
    def compute_geometry(self):
        # compute centroid and normal vector)
        self.c,self.n2=planeFit([self.x,self.y])
        self.N2=np.array([self.n2[1], -self.n2[0]]) 	# along-strike unit vector
        self.n=np.array([np.sin(self.dip/180.*np.pi)*self.n2[0],np.sin(self.dip/180.*np.pi)*self.n2[1],np.cos(self.dip/180.*np.pi)])
        
        # find approximate length of fault
        ps=np.array([self.x-self.c[0], self.y-self.c[1]]).T 	# points relative to centroid		
        p1s=np.array([np.dot(p, self.N2) for p in ps])     # cpt of point in strike of plane
        self.cmax=np.max(p1s)
        self.cmin=np.min(p1s)
        self.l0=self.cmax-self.cmin
        # save angular normal of fault
            # anticlockwise angle between normal and +x
        self.strike=np.arctan2(self.n2[1],self.n2[0])/np.pi*180.
            # clockwise angle between normal and +y
        self.strike=90.-self.strike
            # clockwise angle between in plane and +y 
        self.strike += 90.
        # adjust strike so in range [0,, 360]
        if self.strike < 0.: 
            self.strike += 360.
        if self.strike > 360.: 
            self.strike -= 360.
def planeFit(points):
    """
    p, n=planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points=np.reshape(points, (np.shape(points)[0], -1)) # Collapse trailing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr=points.mean(axis=1)
    x=points-ctr[:,np.newaxis]
    M=np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]

class Model(object):                        # CLASS for seismicity model
    def __init__(self):
        # groningen coordinate limits
        self.x0,self.x1=[228,270]
        self.y0,self.y1=[566,615.5]
    def load_catalog(self, fl):
        ''' loads in earthquake data from file
        '''
        self.catalog_file=fl
        if fl.endswith('.xlsx'):
            eqs=pd.read_excel(self.catalog_file)[['Easting','Northing','Datum','Magnitude']]
            eqs.columns=[c.replace('Datum','time') for c in eqs.columns]
            eqs['x']=eqs['Easting']/1.e3
            eqs['y']=eqs['Northing']/1.e3
        elif fl.endswith('.csv'):
            eqs=pd.read_csv(self.catalog_file, parse_dates=[0], infer_datetime_format=True,sep=';')[['RDX_EDT','RDY_EDT','#Origin_time','magnitude']]
            eqs.columns=[c.replace('#Origin_time','time').replace('magnitude','Magnitude') for c in eqs.columns]
            eqs['x']=eqs['RDX_EDT']/1.e3
            eqs['y']=eqs['RDY_EDT']/1.e3

        eqs=eqs[(eqs['x']>self.x0)&(eqs['x']<self.x1)&(eqs['y']>self.y0)&(eqs['y']<self.y1)]
        eqs=eqs[eqs['Magnitude']>=MC]
        self.catalog=Catalog()
        self.catalog.has_events(year_fraction(eqs['time'].values), eqs['x'].values, eqs['y'].values, eqs['Magnitude'].values)
    def load_pressure(self, fl):
        ''' loads in pressure data from file
        '''
        self.pressure_file=fl
        self.pressure=Pressure(self.pressure_file, parent=self)
    def load_faults(self, fl, dx=1.):
        ''' load in fault data from file
        '''
        self.fault_file=fl
        with open(self.fault_file, 'r') as fp:
            self.faults=[]
            for ln in fp.readlines()[1:]:
                x0,y0,x1,y1=[float(lni) for lni in ln.strip().split(',')]
                self.faults.append(Fault([x0,x1],[y0,y1],70.,dx))
    def set_parameters(self, H, f, A, h, theta, use_thickness, pseudoprospective=False):
        ''' fix model parameters
        '''
        self.H=H
        self.f=f
        self.A=A
        self.h=h
        self.theta=theta
        self.use_thickness=use_thickness
        self.pseudoprospective=pseudoprospective
        if self.pseudoprospective:
            self.TT=2017+1./12
        else:
            self.TT=2021
    def compute_loading(self, overburden_density=2500., initial_pressure=40., depth=3.e3):
        ''' apply loading to faults
        '''
        self.overburden_density=overburden_density
        self.initial_pressure=initial_pressure
        self.depth=depth
        self.setup_stress()
        self.stress_state.projectStress()
        self.pressure.projectPressure()
        self.stress_state.computeLoads()
    def setup_stress(self):
        ''' configure initial tectonic stress
        '''
        sV=abs(self.depth)*self.overburden_density*9.81/1.e6 	
        # compute minimium horizontal principal (absolute) stress at criticality
        sh=(sV-self.initial_pressure)/((np.sqrt(self.f**2+1)+self.f)**2)+self.initial_pressure
        # offset stress for non-criticality
        sh += self.h*(sV-sh)
        # compute maximum horizontal principal (absolute) stress using horizontal stress ratio
        sH=self.H*sV + (1-self.H)*sh
        # create the stress state object
        self.stress_state=Tectonic(principal=[sh, sH, sV], theta=-self.theta*90., s_vertical=1, parent=self)
    def rate_model(self, bandwidth=0.5):   
        ''' assemble rate model
        '''
        ts=self.pressure.t*1.
        rs=np.array([[] for i in range(len(ts))]).T
        xs, ys=np.array([]), np.array([])
        # for each seismogenic fault
        for flt in self.faults:
            # compute stressing components 
            flt.dpdt=finite_difference(flt.dp)
            flt.dsndt=finite_difference(flt.dsn)
            flt.dtaudt=finite_difference(flt.dtau)
            flt.CFS=flt.tau-flt.fr*(flt.sn-(flt.p0+flt.dp))      # CFS
            flt.dCFSdt=flt.dtaudt-flt.fr*(flt.dsndt-flt.dpdt)               # CFS rate
            # compute seismicity rate
            flt.r=flt.dCFSdt*(flt.CFS>0.)*(flt.dCFSdt>0.)*flt.l0/flt.nx+RMIN
            xs=np.concatenate([xs, flt.xv])
            ys=np.concatenate([ys, flt.yv])
            rs=np.concatenate([rs, flt.r.T], axis=0)

        # compute a normalized KDE at each time
        kdes=[]
        for i in range(rs.shape[1]):
            kdes.append(KDE(kernel='gaussian', bandwidth=bandwidth).fit(np.array([xs,ys]).T, sample_weight=rs[:,i]))
        
        # return KDE interpolator function
        if self.use_thickness:
            thickness_weight=thickness
        else:
            thickness_weight=no_thickness
        self.rf_=partial(kde_rf,kdes,ts,rs.sum(axis=0),thickness_weight)
        self.rf=vectorize(self.rf_)
        return self.rf
    def score_rate_model(self, N=21):
        ''' compute modified log-likelihood
        '''
        ## MATCH TIME PROFILE
        t_mod,r_mod=self.rf_.args[1],self.rf_.args[2]  #time and rate model
        ti_obs = self.catalog.t[self.catalog.t<self.TT] # observations
        i=np.argmin(abs(t_mod-self.TT))   # truncate at calibration period
        r_mod=r_mod/r_mod[:i].sum()*len(ti_obs)   # scale seismicity rate model to number of obs.

        r_obs,e=np.histogram(ti_obs, bins=t_mod[:i])  # calculate annual rate observation
        t_obs=0.5*(e[:-1]+e[1:])   # corresponding times of rate obs. (bin midpoints)

        # interpolate model at rate observations
        ri_mod=np.interp(t_obs,t_mod,r_mod)

        # calculate LLK for temporal match
        j=np.argmin(abs(t_obs-2012.5))   # truncate at calibration period
        unc=np.array([np.max([2*ri,1]) for ri in r_obs])
        LLKt1 = -0.5*np.sum((r_obs[:j]-ri_mod[:j])**2/unc[:j])
        LLKt2 = -0.5*np.sum((r_obs[j:]-ri_mod[j:])**2/unc[j:])

        ## MATCH HORIZONTAL PROFILES
        # model grid
        x0,x1=self.pressure.x[0],self.pressure.x[-1]
        y0,y1=self.pressure.y[0],self.pressure.y[-1]
        x_grid,y_grid=np.linspace(x0,x1,N),np.linspace(y0,y1,N)
        xx,yy=np.meshgrid(x_grid,y_grid,indexing='xy')

        # model seismicity density grid at end of calibration
        rr_mod=0*xx
        for t in self.rf_.args[1]:
            if t > self.TT: break
            rr_mod += self.rf_(t,xx.flatten(),yy.flatten()).reshape(xx.shape)     

        # collapse along y direction
        rx_mod=rr_mod.sum(axis=0)
        rx_mod=rx_mod/rx_mod.sum()*len(ti_obs)
        
        xi_obs = self.catalog.x[self.catalog.t<self.TT]    # observed x locations
        densx_obs,e=np.histogram(xi_obs, bins=x_grid)     # observed x density
        x_obs=0.5*(e[:-1]+e[1:])   # corresponding loc of density obs. (bin midpoints)

        # interpolate model at density observations
        rxi_mod=np.interp(x_obs,x_grid,rx_mod)

        # calculate LLK for temporal match
        unc=np.array([np.max([2*ri,1]) for ri in densx_obs])
        LLKx = -0.5*np.sum((densx_obs-rxi_mod)**2/unc)/N

        # collapse along x direction
        ry_mod=rr_mod.sum(axis=1)
        ry_mod=ry_mod/ry_mod.sum()*len(ti_obs)    
        
        yi_obs = self.catalog.y[self.catalog.t<self.TT]    # observed y locations
        densy_obs,e=np.histogram(yi_obs, bins=y_grid)     # observed y density
        y_obs=0.5*(e[:-1]+e[1:])   # corresponding loc of density obs. (bin midpoints)

        # interpolate model at density observations
        ryi_mod=np.interp(y_obs,y_grid,ry_mod)

        # calculate LLK for temporal match
        unc=np.array([np.max([2*ri,1]) for ri in densy_obs])
        LLKy = -0.5*np.sum((densy_obs-ryi_mod)**2/unc)

        # WEIGHTED LLK
        omega = 0.75          # fraction of weighting on time (1 = full time weighting)
        t_wt1 = omega/(2011-1960)   # number of years in pre-peak calibration period
        t_wt2 = omega/(self.TT-2012)   # number of years in post-peak calibration period
        xy_wt = (1.-omega)/(N-1)

        return t_wt1*LLKt1+t_wt2*LLKt2+xy_wt*(LLKx+LLKy)
    
# helper functions
def finite_difference(v):
    ''' calculate finite difference of matrix
    '''
    dv=0.*v
    if len(dv.shape)==3:
        dv[1:-1,:,:]=(v[2:,:,:]-v[:-2,:,:])/2.
        dv[-1,:,:]=v[-1,:,:]-v[-2,:,:]
        dv[0,:,:]=v[1,:,:]-v[0,:,:]
    elif len(dv.shape)==2:
        dv[1:-1,:]=(v[2:,:]-v[:-2,:])/2.
        dv[-1,:]=v[-1,:]-v[-2,:]
        dv[0,:]=v[1,:]-v[0,:]
    return dv
def isiterable(x):
    try:
        [_ for _ in x]
        return True
    except TypeError:
        return False
def vectorize(f):
    return partial(vector_f, f)
def vector_f(f, t, x, y):
    return np.array([f(*p) for p in zip(t,x,y)])
def year_fraction(date):
    if isiterable(date):
        return np.array([year_fraction(d) for d in date])
    else:
        date=pd.to_datetime(date)
        start=datetime.date(date.year, 1, 1).toordinal()
        year_length=datetime.date(date.year+1, 1, 1).toordinal()-start
        return date.year + float(date.toordinal()-start) / year_length

# rate functions
def hf(x,y):
    ''' Groningen thickness function
    '''
    return -3.0409293334119023*x+4.113225196077208*y-1449.5997490388474
def thickness(x,y):
    return hf(x,y)/hf(250.,592.)
def no_thickness(x,y):
    return 1.
def kde_rf(kdes, ts, rs, h, t, x, y):
    ''' return interpolated kde value 
    '''
    xy=np.array([x,y]).T
    if len(xy.shape) == 1: 
        xy=xy.reshape(1,-1)
    
    # edge cases
    if t <= ts[0]:
        return np.exp(kdes[0].score_samples(xy))*rs[0]
    elif t >= ts[-1]:
        return np.exp(kdes[-1].score_samples(xy))*rs[-1]
    # find bounding KDEs for given time
    idx=np.searchsorted(ts,t,'left')-1
    t0,t1=ts[idx],ts[idx+1]
    dt0,dt1=t-t0, t1-t
    # interpolate-normalized KDEs multiplied by the total rate for that year
    on_fault=np.exp(kdes[idx].score_samples(xy))*dt1*rs[idx]+np.exp(kdes[idx+1].score_samples(xy))*dt0*rs[idx+1]
    return (on_fault+RMIN)*h(x,y)
def generate_times_opt(t, sr):
    ''' generate earthquake catalog event times from rate function
    '''
    ti = np.linspace(t[0],t[-1],101)
    si = np.interp(ti,t,sr)
    dt = ti[1]-ti[0]
    si = 0.5*(si[:-1]+si[1:])		
    N = 0
    ts=[]
    for sii,t0,t1 in zip(si,ti[:-1], ti[1:]):
        N = np.random.poisson(sii*dt)
        ts += list(np.random.rand(N)*(t1-t0)+t0)
    return np.sort(ts)

# workflow functions
def prospective_forecast():
    # initialise
    dat=Model()
    dat.load_catalog(CATALOG_FILE)
    dat.load_pressure(PRESSURE_FILE)
    dat.load_faults(FAULT_FILE)
    
    # compute rate model
    H, f, A, h, theta = [0.38093657, 0.50235846, 0.99799661, 0.19119866, 0.51830452]
    dat.set_parameters(
        H=H, 
        f=f,
        A=A,
        h=h,
        theta=theta,
        use_thickness=True,
        pseudoprospective=False
        )
    dat.compute_loading(
        overburden_density=2500., 
        initial_pressure=35., 
        depth=3.e3
        )
    dat.rate_model()

    # get unscaled seismicity rate
    ts,rs=dat.rf_.args[1],dat.rf_.args[2]    
    tv=np.linspace(ts[0],ts[-1],1000)
    rv=np.interp(tv,ts,rs)
    iv=np.argmin(abs(tv-dat.TT))
    jv=np.argmin(abs(tv-2030))                    

    # calculate total unscaled seismicity
    Nh=int(np.trapz(rv[:iv],tv[:iv]))

    # scale to catalog
    rs=rs/Nh*np.sum(dat.catalog.t<dat.TT)
    rv=np.interp(tv,ts,rs)
    Nh=int(np.trapz(rv[:iv],tv[:iv]))
    Nf=int(np.trapz(rv[iv:jv],tv[iv:jv]))       # estimate events in forecast

    # prospective N-2.5 and Mmax exceedance curves
    N_catalogs=100000         # number of synthetic catalogs to generate
    beta,eta=[0.6, 2.1e-4]      # tapered GR parameters
    eta0=eta/1.e8               # unbounded GR
    Mm=10**(9.1+1.5*MC)         
    Nx=poisson.rvs(Nf,size=N_catalogs)  # random catalog sizes
    
    # generate catalogs and save outputs
    N25s=[]; N25s0=[]; Mmaxs=[]; Mmaxs0=[]
    for i,Nxi in enumerate(Nx):
        # tapered magnitudes
        x=np.random.rand(int(Nxi))
        M=np.real(Mm*beta*lambertw(eta/beta*(np.exp(-eta)*x)**(-1./beta))/eta)
        m=(np.log10(M)-9.1)/1.5
        N25s.append(np.sum(m>=2.5))
        Mmaxs.append(np.max(m))
        
        # untapered magnitudes
        M=np.real(Mm*beta*lambertw(eta0/beta*(np.exp(-eta0)*x)**(-1./beta))/eta0)
        m=(np.log10(M)-9.1)/1.5
        N25s0.append(np.sum(m>=2.5))
        Mmaxs0.append(np.max(m))

    # generate exceedance curves
    for v,lbl in zip([N25s,N25s0,Mmaxs,Mmaxs0],['N2.5 taper','N2.5 unbounded','Mmax taper','Mmax unbounded']):
        h,e=np.histogram(v,bins=31)
        h=h/h.sum()/(e[1]-e[0])
        e=0.5*(e[:-1]+e[1:])
        c=1.-np.cumsum(h)*(e[1]-e[0])
        
        # print output
        print('{:s} exceedance: 50, 10, 1, 0.1 %'.format(lbl))
        print(np.interp([0.5,0.1,0.01,0.001], c[::-1], e[::-1]))

def global_optimization():
    ''' runs the genetic algorithm
    '''
    # file for saving output
    name='root'    
    fl='{:s}_parameter_sweep.txt'.format(name)
    with open(fl, 'w') as fp:
        fp.write('H, f, A, h, theta, LLK\n')
    
    # optimization function
    opt_func=partial(run_one_model, name)

    # run the genetic algorithm
    differential_evolution(opt_func, bounds=[[0,1],[0.5,0.8],[0.8,1],[0,0.5],[0,1]], workers=4)
def run_one_model(name, x):
    ''' runs one simulation of the genetic algorithm
    '''
    fl='{:s}_parameter_sweep.txt'.format(name)
    H, f, A, h, theta=x

    # initialise the model
    dat=Model()
    dat.load_catalog(CATALOG_FILE)
    dat.load_pressure(PRESSURE_FILE)
    dat.load_faults(FAULT_FILE)
    dat.set_parameters(
        H=H, 
        f=f,
        A=A,
        h=h,
        theta=theta,
        use_thickness=True,
        pseudoprospective=True
        )

    try:
        # compute the model
        t0=time()
        dat.compute_loading(overburden_density=2500., initial_pressure=35., depth=3.e3)
        dat.rate_model()
        score=dat.score_rate_model()
        t1=time()
        
        # save results and print to screen
        print((7*'{:3.2f}, ').format(H, f, A, h, theta, score, t1-t0)[:-2])
        with open(fl, 'a') as fp:
            fp.write('{:8.7e}, {:8.7e}, {:8.7e}, {:8.7e}, {:8.7e}, {:8.7e}\n'.format(H, f, A, h, theta, score))

        return -score
    except:
        with open(fl, 'a') as fp:
            fp.write('{:8.7e}, {:8.7e}, {:8.7e}, {:8.7e}, {:8.7e}, {:8.7e}\n'.format(H, f, A, h, theta, -9999999999999.))

        print((7*'{:3.2f}, ').format(H, f, A, h, theta, -9999999999999, -9999999999999)[:-2])
        return 1.e32
def test_plot(dat):
    ''' produce a basic plot of the model
    '''    
    f,axs=plt.subplot_mosaic([['A','A','A'],['A','A','A'],['B','C','D']])
    ax=axs['A']
    ax1=axs['B']
    ax2=axs['C']
    ax3=axs['D']

    # get unscaled seismicity rate
    ts,rs=dat.rf_.args[1],dat.rf_.args[2]    
    tv=np.linspace(ts[0],ts[-1],1000)
    rv=np.interp(tv,ts,rs)
    iv=np.argmin(abs(tv-dat.TT))
    jv=np.argmin(abs(tv-2022.5))                    

    # calculate total unscaled seismicity
    Nh=int(np.trapz(rv[:iv],tv[:iv]))
    Nf=int(np.trapz(rv[iv:jv],tv[iv:jv]))

    # scale to catalog
    i=np.argmin(abs(ts-dat.TT))
    rs=rs/Nh*np.sum(dat.catalog.t<dat.TT)
    rv=np.interp(tv,ts,rs)
    Nh=int(np.trapz(rv[:iv],tv[:iv]))
    Nf=int(np.trapz(rv[iv:jv],tv[iv:jv]))       # estimate events in forecast

    # calculate 90% confidence interval on model rate
    alpha=0.1
    rlo=[chi2.ppf(alpha/2., r) for r in rs]
    rhi=[chi2.ppf(1.-alpha/2., r) for r in rs]

    # plot seismicity
    ax.plot(ts,rs,'k-',lw=0.5,zorder=3, label='model')    
    ax.fill_between(ts, rlo, rhi, color='k', alpha=0.2, linewidth=0., zorder=2, label='90% envelope')
    i=np.argmin(abs(ts-dat.TT))
    ax.axvline(dat.TT, color='k', linestyle=':', label='train/test split')
    h,e=np.histogram(dat.catalog.t, bins=ts[:i+1])
    ax.plot(0.5*(e[:-1]+e[1:]), h, 'k.', label='training data')
    h,e=np.histogram(dat.catalog.t, bins=ts[i:])
    h=[hi if hi>0 else None for hi in h]
    ax.plot(0.5*(e[:-1]+e[1:]), h, 'wo', mec='k', ms=5, label='test data')
    ax.plot(2022.5, 4., 'wo', mec='k', ms=5)
    ax.plot(2022.5, 8., 'wo', mec='k', ms=5, alpha=0.5)
    ax.set_xlim([1985,2030])
    ax.set_ylim([None, 110])
    ax.legend()
    ax.set_xlabel('year')
    ax.set_ylabel(r'annual earthquakes [M$\geq$1.0]')

    # pseudo-prospective L-test    
    N_catalogs=10000         # number of synthetic catalogs to generate
    LLKS=[]
    for i in range(N_catalogs):
        tis=generate_times_opt(tv[iv:jv], rv[iv:jv])  # catalog times
        LLK=np.sum(np.log(np.interp(tis, tv[iv:jv], rv[iv:jv])))-Nf   # compute likelihood
        LLKS.append(LLK)
    
    # observed catalog
    tos=dat.catalog.t[np.where((dat.catalog.t>dat.TT)&(dat.catalog.t<2022.5))]
    # observed likelihood
    LLKo=np.sum(np.log(np.interp(tos, tv[iv:jv], rv[iv:jv])))-Nf
    
    # compute exceedance curve
    h,e=np.histogram(LLKS,bins=31)
    h=h/h.sum()/(e[1]-e[0])
    e=0.5*(e[:-1]+e[1:])
    c=1.-np.cumsum(h)*(e[1]-e[0])
    ax1.plot(e,c,'k-')

    ax1.set_yscale('log')
    ax1.set_ylim([0.01,1.1])        
    ax1.axvline(LLKo,color='k',linestyle=':')
    ax1.set_title('L-test: p={:3.2f}'.format(np.interp(LLKo, e, c)))
    
    # pseudo-prospective N-test and Mmax     
    beta,eta=[0.6, 2.1e-4]      # tapered GR parameters
    eta0=eta/1.e6               # unbounded GR
    Mm=10**(9.1+1.5*MC)         
    Nx=poisson.rvs(Nf,size=N_catalogs)  # random catalog sizes
    
    # generate catalogs and save outputs
    N25s=[]; N25s0=[]; Mmaxs=[]; Mmaxs0=[]
    for i,Nxi in enumerate(Nx):
        # tapered magnitudes
        x=np.random.rand(int(Nxi))
        M=np.real(Mm*beta*lambertw(eta/beta*(np.exp(-eta)*x)**(-1./beta))/eta)
        m=(np.log10(M)-9.1)/1.5
        N25s.append(np.sum(m>=2.5))
        Mmaxs.append(np.max(m))
        
        # untapered magnitudes
        M=np.real(Mm*beta*lambertw(eta0/beta*(np.exp(-eta0)*x)**(-1./beta))/eta0)
        m=(np.log10(M)-9.1)/1.5
        N25s0.append(np.sum(m>=2.5))
        Mmaxs0.append(np.max(m))

    # generate exceedance curves
    nps=[]; mps=[]
    for v, col, ax in zip([N25s,N25s0,Mmaxs,Mmaxs0],['-','--','-','--'],[ax2,ax2,ax3,ax3]):
        h,e=np.histogram(v,bins=31)
        h=h/h.sum()/(e[1]-e[0])
        e=0.5*(e[:-1]+e[1:])
        c=1.-np.cumsum(h)*(e[1]-e[0])
        ax.plot(e,c,'k'+col)
        if ax==ax2:
            nps.append(np.interp(10, e, c))
        else:
            mps.append(np.interp(3.4, e, c))

        ax.set_yscale('log')
        ax.set_ylim([0.01,1.1])
    ax2.set_xlim([0,15])
    ax3.set_xlim([2.5,6])
    ax3.axvline(3.4,color='k',linestyle=':', label='observed')
    ax3.plot([],[],'k-',label='tapered')
    ax3.plot([],[],'k--',label='unbounded')
    
    ax2.axvline(10,color='k',linestyle=':')
    ax3.legend()

    ax2.set_title('N-test: p={:3.2f} ({:3.2f})'.format(*nps))
    ax3.set_title('Mmax: p={:3.2f} ({:3.2f})'.format(*mps))
    ax1.set_xlabel('likelihood')
    ax2.set_xlabel('$N_{2.5}$')
    ax3.set_xlabel('$M_{max}$')
    ax1.set_ylabel('probability')
    ax2.set_ylabel('probability')
    ax3.set_ylabel('probability')

    plt.tight_layout()
    plt.show()
def test_model():
    ''' test model function working
    '''
    # initialise
    dat=Model()
    dat.load_catalog(CATALOG_FILE)
    dat.load_pressure(PRESSURE_FILE)
    dat.load_faults(FAULT_FILE)
    
    # compute rate model
    H, f, A, h, theta = [0.36301536, 0.50064199, 0.99788937, 0.19481503, 0.51096218]
    dat.set_parameters(
        H=H, 
        f=f,
        A=A,
        h=h,
        theta=theta,
        use_thickness=True,
        pseudoprospective=True
        )
    dat.compute_loading(
        overburden_density=2500., 
        initial_pressure=35., 
        depth=3.e3
        )
    dat.rate_model()
    
    # plot output
    test_plot(dat)
def test():
    test_model()

if __name__=="__main__":
    test()
    prospective_forecast()
    global_optimization()
