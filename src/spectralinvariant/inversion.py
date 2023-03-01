"""
Inversion of leaf reflectance spectral invariant theory (p-theory)

Classes
--------
- PROSPECT_D : PROSPECT radiative transfer model version D class

Functions
---------
- pc_fast : least squares calculation of the spectral invariants rho, p, and c.
- rectifier: rectification of input value
- rss_function: residual sum of squares for the p-theory forward model
- minimize_cab: inversion of leaf chlorophyll a+b content using `scipy.minimize` methods
- golden_cab: inversion of leaf chlorophyll a+b content using `scipy.golden` golden section search method
"""

import numpy as np
from scipy.special import exp1
from scipy.optimize import minimize, golden, brent
from spectralinvariant.prospect import PD_refractive, PD_k_Cab, PD_k_Cw, PD_k_Cm, PD_k_Car, PD_k_Brown, PD_k_Anth, PD_tav90n, PD_t12
from math import sqrt, exp

class PROSPECT_D:
    """PROSPECT radiative transfer model version D class

    Attributes
    ----------
    lambd : ndarray
        Wavelengths used by the PROSPECT model. The values have to be between 400 and 2500 nm.
    nlambd : int
        Number of wavelengths
    n : ndarray
        Refractive index
    kCab : ndarray
        Specific absorption coefficient of chlorophylls a+b
    kw : ndarray
        Specific absorption coefficient of the leaf water content
    km : ndarray
        Specific absorption coefficient of the dry mass per area
    kCar : ndarray
        Specific absorption coefficient of carotenoids
    kbrown : ndarray
        Specific absorption coefficient of fraction of brown leaves
    kanth : ndarray
        Specific absorption coefficient of anthocyanins
    kp : ndarray
        Specific absorption coefficient of proteins
    kcl : ndarray
        Specific absorption coefficient of cellulose and lignin
    tav90n : ndarray
        XXX
    t12 : ndarray
        XXX


    Methods
    -------
    PROSPECT(N=None, Cab=None, Cw=None, Cm=None, Car=None, Cbrown=None, Canth=None, Cp=None, Ccl=None)
    
    To do
    -----
    - make sure 400 < lambd < 2500

    """
    def __init__(self, N=2., Cab=48.6, Cw=0.0115, Cm=0.0045, Car=10.5, Cbrown=0., Canth=7.8, Cp=0., Ccl=0.):
        """
        Féret J.B., Gitelson A.A., Noble S.D., & Jacquemoud S. (2017), PROSPECT-D: towards modeling leaf optical properties through a complete lifecycle, Remote Sensing of Environment, 193:204-21

        Parameters
        ----------
        N : float
            Leaf structure parameter. Defaults to 2.0.
        Cab : float
            Leaf Chlorophyll a+b content [ug/cm2]. Defaults to 48.6.
        Cw : float
            Leaf Equivalent Water content [cm]. Defaults to 0.0115
        Cm : float
            Leaf dry Mass per Area [g/cm2]. Defaults to 0.0045
        Car : float 
            Leaf Carotenoids content [ug/cm2]. Defaults to 10.5.
        Cbrown : float
            Fraction of brown leaves. Defaults to 0.
        Anth : float
            Leaf Anthocyanins content [ug/cm2] Defaults to 0.
        Cp : float
            Leaf protein content [g/cm2]. Defaults to 0. (not used in PROSPECT-D).
        Ccl : float
            Leaf cellulose, hemicellulose, and lignin content [g/cm2]. Defaults to 0 (not used in PROSPECT-D)
        """
        self.N = N
        self.Cab = Cab
        self.Cw = Cw
        self.Cm = Cm
        self.Car = Car
        self.Cbrown = Cbrown
        self.Canth = Canth
        self.Cp = Cp
        self.Ccl = Ccl

        self.lambd = np.arange(400, 2501, dtype=float)
        self.nlambd = len(self.lambd)

        self.n = PD_refractive
        self.kCab = PD_k_Cab
        self.kw = PD_k_Cw
        self.km = PD_k_Cm
        self.kCar = PD_k_Car
        self.kbrown = PD_k_Brown
        self.kanth = PD_k_Anth
        self.kp = 0.
        self.kcl = 0.
        self.tav90n = PD_tav90n
        self.t12 = PD_t12
        
    def subset(self, lambd_new):
        """Select subset wavelengths and interpolate the specific absorption coefficients to the subset range

        Parameters
        ----------
        lambd_new : ndarray
            New wavelengths. Must be between 400 and 2500 nm
        """

        self.n = np.interp(lambd_new, self.lambd, self.n)
        self.kCab = np.interp(lambd_new, self.lambd, self.kCab)
        self.kw = np.interp(lambd_new, self.lambd, self.kw)
        self.km = np.interp(lambd_new, self.lambd, self.km)
        self.kCar = np.interp(lambd_new, self.lambd, self.kCar)
        self.kbrown = np.interp(lambd_new, self.lambd, self.kbrown)
        self.kanth = np.interp(lambd_new, self.lambd, self.kanth)
        self.tav90n = np.interp(lambd_new, self.lambd, self.tav90n)
        self.t12 = np.interp(lambd_new, self.lambd, self.t12)
        self.lambd = lambd_new
        self.nlambd = len(self.lambd)


    def reset_attributes(self):
        """Resets the wavelengths and the specific absorption coefficients to the original range
        """
        self.lambd = np.arange(400, 2501, dtype=float)
        self.nlambd = len(self.lambd)

        self.n = PD_refractive
        self.kCab = PD_k_Cab
        self.kw = PD_k_Cw
        self.km = PD_k_Cm
        self.kCar = PD_k_Car
        self.kbrown = PD_k_Brown
        self.kanth = PD_k_Anth
        self.kp = 0.
        self.kcl = 0.
        self.tav90n = PD_tav90n
        self.t12 = PD_t12


    def set_params(self, N=None, Cab=None, Cw=None, Cm=None, Car=None, Cbrown=None, Canth=None, Cp=None, Ccl=None):
        """Sets PROSPECT input parameters

        Parameters
        ----------
        N : float
            Leaf structure parameter. Defaults to None.
        Cab : float
            Leaf Chlorophyll a+b content [ug/cm2]. Defaults to None.
        Cw : float
            Leaf Equivalent Water content [cm]. Defaults to None
        Cm : float
            Leaf dry Mass per Area [g/cm2]. Defaults to None
        Car : float 
            Leaf Carotenoids content [ug/cm2]. Defaults to None
        Cbrown : float
            Fraction of brown leaves. Defaults to None.
        Anth : float
            Leaf Anthocyanins content [ug/cm2] Defaults to None.
        Cp : float
            Leaf protein content [g/cm2]. Defaults to None. (not used in PROSPECT-D).
        Ccl : float
            Leaf cellulose, hemicellulose, and lignin content [g/cm2]. Defaults to None (not used in PROSPECT-D)

        """
        if N is not None: self.N=N
        if Cab is not None: self.Cab=Cab
        if Cw is not None: self.Cw=Cw
        if Cm is not None: self.Cm=Cm
        if Car is not None: self.Car=Car
        if Cbrown is not None: self.Cbrown=sCbrown
        if Canth is not None: self.Canth=Canth
        if Cp is not None: self.Cp=Cp
        if Ccl is not None: self.Ccl=Ccl

        
    # def PROSPECT(self, N=1.5, Cab=50, Cw=0.001, Cm=0.001, Car=1.0, Cbrown=0.001, Canth=0.0, Cp=0.0, Ccl=0.0):
    def PROSPECT(self, N=None, Cab=None, Cw=None, Cm=None, Car=None, Cbrown=None, Canth=None, Cp=None, Ccl=None):
        """Computes leaf hemispherical-directional reflectance between 400 and 2500 nm using the PROSPECT-D model
        Féret J.B., Gitelson A.A., Noble S.D., & Jacquemoud S. (2017), PROSPECT-D: towards modeling leaf optical properties through a complete lifecycle, Remote Sensing of Environment, 193:204-21

        Note that this implementation of the PROSPECT model does not compute the transmittance factor T, as it is not needed in the inversion.
        If later needed, the computation of T can be included by uncommenting two lines at the end of this function and modifying the output

        Parameters
        ----------
        N : float
            Leaf structure parameter. Defaults to 2.0.
        Cab : float
            Leaf Chlorophyll a+b content [ug/cm2]. Defaults to 48.6.
        Cw : float
            Leaf Equivalent Water content [cm]. Defaults to 0.0115
        Cm : float
            Leaf dry Mass per Area [g/cm2]. Defaults to 0.0045
        Car : float 
            Leaf Carotenoids content [ug/cm2]. Defaults to 10.5.
        Cbrown : float
            Fraction of brown leaves. Defaults to 0.
        Anth : float
            Leaf Anthocyanins content [ug/cm2] Defaults to 0.
        Cp : float
            Leaf protein content [g/cm2]. Defaults to 0. (not used in PROSPECT-D).
        Ccl : float
            Leaf cellulose, hemicellulose, and lignin content [g/cm2]. Defaults to 0 (not used in PROSPECT-D)

        Returns
        -------
        RN : ndarray
           leaf hemispherical-directional reflectance factor of the N layers

        To do
        -----
        - Write a C extension of the Swamee and Ohija approximation for exp1 to improve computational speed
        """
        if N is None: N=self.N
        if Cab is None: Cab=self.Cab
        if Cw is None: Cw=self.Cw
        if Cm is None: Cm=self.Cm
        if Car is None: Car=self.Car
        if Cbrown is None: Cbrown=self.Cbrown
        if Canth is None: Canth=self.Canth
        if Cp is None: Cp=self.Cp
        if Ccl is None: Ccl=self.Ccl
        k = (Cab*self.kCab + Car*self.kCar + Canth*self.kanth + Cbrown*self.kbrown + Cw*self.kw + Cm*self.km) / N
        
        # ind_k0 = np.where(k==0)
        # if not len(ind_k0[0])==0: k[ind_k0] = np.finfo(float).eps
        trans = (1. - k)*np.exp(-k) + (k*k)*exp1(k) # The exp1 function takes most of the computation time in this function...
        
        trans2 = trans*trans

        # t12 is tav(4 0,n); tav90n is tav(90,n)
        t21 = self.tav90n/(self.n*self.n)
        r12 = 1. - self.t12
        r21 = 1. - t21
        r21_2 = r21 * r21
        x = self.t12 / self.tav90n
        y = x * (self.tav90n - 1.) + 1. - self.t12

        # reflectance and transmittance of the elementary layer N = 1
        ra = r12 + (self.t12 * t21 * r21 * trans2) / (1. - (r21_2)*(trans2))
        ta = (self.t12 * t21 * trans) / (1. - (r21_2)*(trans2))
        r90 = (ra - y) / x
        t90 = ta / x

        # reflectance and transmittance of N layers
        t90_2 = t90*t90
        r90_2 = r90*r90
        
        b = (t90_2 - r90_2 - 1.)
        delta = np.sqrt(b*b - 4.*r90_2)
        beta = (1. + r90_2 - t90_2 - delta)/(2. * r90)
        va = (1. + r90_2 - t90_2 + delta) / (2. * r90)

        # vb = np.zeros(self.nlambd)
        # ind_vb_le = np.where(va*(beta - r90) <= 1e-14)
        # ind_vb_gt = np.where(va*(beta - r90) > 1e-14)
        # vb[ind_vb_le] = np.sqrt(beta[ind_vb_le]*(va[ind_vb_le] - r90[ind_vb_le])/(1e-14))
        # vb[ind_vb_gt] = np.sqrt(beta[ind_vb_gt]*(va[ind_vb_gt] - r90[ind_vb_gt]) / (va[ind_vb_gt]*(beta[ind_vb_gt] - r90[ind_vb_gt])))
        vb = np.sqrt(beta*(va - r90) / (va * (beta - r90))) # This will sometimes produce invalid values (negative square root or division by zero)
        
        vbNN = vb**(N-1)
        vbNNinv = 1. / vbNN
        vainv = 1. / va
        s1 = ta * t90 * (vbNN - vbNNinv)
        # s2 = ta * (va - vainv)
        s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

        RN = ra + s1 / s3
        # TN = s2 / s3
        return RN



def pc_fast(hypdata, refspectrum):
    """Solves three spectral invariant parameters p, rho, and C using linear regression for a single pixel

     Parameters
    ----------
    hypdata : ndarray
        Hyperspectral reflectance data
    refspectrum : ndarray
        The leaf reference spectrum

    Returns
    -------
        beta : ndarray
            The estimated values of rho, p, and c, respectively
    """
    n = len(refspectrum)
    X = np.array([np.ones(n), hypdata, 1./refspectrum]).T
    Y = hypdata / refspectrum
    XTX = (X.T).dot(X)
    XTY = (X.T).dot(Y)
    try:
        beta = np.linalg.inv(XTX).dot(XTY)
    except: # If the matrix XTX is singular, use the Moore-Penrose pseudoinverse
        beta = np.linalg.pinv(XTX).dot(XTY)
    return beta


def rectifier(x, a=0, b=1.1):
    """Returns 0 if x is between [a, b] and a exp(x^2) if x is outside of the range.

     Parameters
    ----------
    x : float
        Value to be rectified
    a : float
        Lower bound of the rectification range
    b : float
        Upper bound of the rectification range

    Returns
    -------
    out : float
        Output of the rectu
    """
    out = 0.
    if (x<a) or (x>b):
        out = exp(x*x) # Using math.exp(x) is faster than np.exp(x) when x is not an ndarray
    return out


def rss_function(u, prospect_instance, hypdata, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0):
    """Calculates the (regularized) residual sum of squares between measured and modelled top-of-canopy reflectance
    
     Parameters
    ----------
    cab : float
        Leaf Chlorophyll content
    prospect_instance : object
        Instance of the PROSPECT_D class
    hypdata : ndarray
        Hyperspectral reflectance data
    gamma : float
        Regularization parameter. If gamma<=0, regularization is not used 
    p_lower : float
        Lower bound for rectifying the values of p
    p_upper : float
        Upper bound for rectifying the values of p
    rho_lower : float
        Lower bound for rectifying the values of rho
    rho_upper : float
        Upper bound for rectifying the values of p

    Returns
    -------
        beta : ndarray
            The estimated values of rho, p, and c, respectively
    """
    # Generate leaf reflectance specturm using PROSPECT
    refspectrum = prospect_instance.PROSPECT(N=1.5, Cab=u, Cw=0., Cm=0., Car=0., Cbrown=0.0, Canth=0., Cp=0., Ccl=0.)

    # Compute the spectral invariants with least squares
    rho, p, C = pc_fast(hypdata, refspectrum) # p, rho, C

    # Compute the forward model, residual, and residual sum of squares
    prediction = rho + p*hypdata + C/refspectrum # The forward model divided by leaf reflectance seems to work better in the inversion than just the forward model
    residual = hypdata/refspectrum - prediction
    rss = residual.dot(residual)

    # Regularize
    if gamma > 0.:
        rho = rectifier(rho, a=rho_lower, b=rho_upper)
        p = rectifier(p, a=p_lower, b=p_upper)
        rss += (rho*rho + p*p + C*C)*gamma
    return rss


# methods = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr']
def minimize_cab(prospect_instance, hypdata, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, initial_guess=30., bounds=[(1., 100.)], method='Powell', **kwargs):
    """Invert the leaf chlorophyll a+b content using p-theory with `scipy.minimize` function

    This is basically a wrapper for the `scipy.minimize` function which supports several optimization.

    Parameters
    ----------
    prospect_instance : object
        PROSPECT-D object.
    hypdata : ndarray
        Hyperspectral reflectance data.
    gamma : float
        Regularization parameter. If gamma<=0, regularization is not used .
    p_lower : float
        Lower bound for rectifying the values of p.
    p_upper : float
        Upper bound for rectifying the values of p.
    rho_lower : float
        Lower bound for rectifying the values of rho.
    rho_upper : float
        Upper bound for rectifying the values of p.
    initial_guess : float
        Initial guess for the chlorophyll content. 
    bounds : sequence
        Upper and lower bounds of the chlorophyll a+b content.
    method : str
        Type of solver. Should be one of
        - 'Nelder-Mead'
        - 'L-BFGS-B'
        - 'TNC'
        - 'SLSQP'
        - 'Powell'
        - 'trust-constr'
    **kwargs : dict
        Keyword arguments for the `scipy.minimize` function

    Returns
    -------
    ans : float
        Result of the inversion
    """
    try:
        ans = minimize(rss_function, x0=initial_guess, args=(prospect_instance, hypdata, gamma, p_lower, p_upper, rho_lower, rho_upper), bounds=bounds, method=method, tol=1e-4, **kwargs).x[0]
    except:
        ans = 1
    return ans


def golden_cab(prospect_instance, hypdata, gamma=1.0, p_lower=0.0, p_upper=1.0, rho_lower=0.0, rho_upper=2.0, bounds=(1., 100.), **kwargs):
    """Invert the leaf chlorophyll a+b content using p-theory using the golden section search method

    This is basically a wrapper for the `scipy.golden` function.

    Parameters
    ----------
    prospect_instance : object
        PROSPECT-D object.
    hypdata : ndarray
        Hyperspectral reflectance data.
    gamma : float
        Regularization parameter. If gamma<=0, regularization is not used .
    p_lower : float
        Lower bound for rectifying the values of p.
    p_upper : float
        Upper bound for rectifying the values of p.
    rho_lower : float
        Lower bound for rectifying the values of rho.
    rho_upper : float
        Upper bound for rectifying the values of p.
    bounds : sequence
        Upper and lower brackets for searching the chlorophyll a+b content.
    **kwargs : dict
        Keyword arguments for the `scipy.golden` function

    Returns
    -------
    ans : float
        Result of the inversion
    """
    try:
        ans = golden(rss_function, args=(prospect_instance, hypdata, gamma, p_lower, p_upper, rho_lower, rho_upper), brack=bounds, tol=1e-4, maxiter=10, **kwargs)
    except:
        ans = 1.0
    return ans

