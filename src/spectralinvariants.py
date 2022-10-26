"""
Copyright (C) 2017,2018  Matti Mõttus
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
Functions for applying the spectral invariants theory to hyper- and multispectral data
Utilized by e.g. hypdatatools_algorithms
"""
import numpy as np
import spectralinvariant.prospect as prospect
from numpy.linalg import solve


def p_forpixel(hypdata, refspectrum, p_values):
    """ the actual calculation of p (fitting a line).
        DEPRECATED, please use the similar newer function by Olli instead!

    Implementation from scratch, designed to fill in a 3D raster of p-values given as function argument

    Args:
        hypdata: a ndarray with spectrum
        refspectrum: the reference spectrum with the same wavelengths
        p_values: output, ndarray of length 4
            0:slope 1:intercept 2: DASF 3:R

    Returns:
        nothing, use p() to get something back.
    """
    # possible implementation (in Scilab notation)
    #   Sxx       = sum(x^2)-sum(x)^2/n
    #   Syy       = sum(y^2)-sum(y)^2/n
    #   Sxy       = sum(x.*y)-sum(x)*sum(y)/n
    #   slope     = Sxy/Sxx
    #   intercept = mean(y) - slope*mean(x)
    #   rxy       = Sxy/sqrt(Sxx*Syy) # Correlation coefficient

    y_DASF = hypdata / refspectrum

    n = hypdata.shape[0]
    Sx = hypdata.sum()
    Sxx = (hypdata * hypdata).sum() - Sx * Sx / n
    Sy = y_DASF.sum()
    Syy = (y_DASF * y_DASF).sum() - Sy * Sy / n
    Sxy = (hypdata * y_DASF).sum() - Sx * Sy / n
    p_values[0] = Sxy / Sxx  # p = slope
    p_values[1] = (Sy - p_values[0] * Sx) / n  # rho = intercept
    p_values[2] = p_values[1] / (1 - p_values[0])  # DASF
    p_values[3] = Sxy / np.sqrt(Sxx * Syy)  # R = Pearson's correlation coefficient

def p_forimage(hypdata, refspectrum, i_ref=None ):
    """ the actual calculation of p (fitting a line).
        DEPRECATED, please use the similar newer function by Olli instead!

    Implementation from scratch, designed to fill in a 3D raster of p-values given as function argument.
    Input is a spectral image (np.ndarray), ouput 3-dim np.ndarray

    Args:
        hypdata: a spectral image, np.ndarray with spectrum along dimension #2
        refspectrum: the reference spectrum with the same wavelengths
        i_ref: the index along the spectral direction (#2) for susetting the fitting interval
            if None, whole reference spectrum is used

    Returns:
        p_values: output, ndarray with dimension #2 having length 4
            0:slope 1:intercept 2: DASF 3:R
    """
    if i_ref is None:
        i_ref = range(len(refspectrum))
    hd = hypdata[:,:,i_ref]
    y_DASF = hd / refspectrum[i_ref]

    n = hd.shape[2]
    Sx = hd.sum(axis=2)
    Sxx = (hd * hd).sum(axis=2) - Sx * Sx / n
    Sy = y_DASF.sum(axis=2)
    Syy = (y_DASF * y_DASF).sum(axis=2) - Sy * Sy / n
    Sxy = (hd * y_DASF).sum(axis=2) - Sx * Sy / n

    p_values = np.empty( (hd.shape[0], hd.shape[1], 4) )
    p_values[:,:,0] = Sxy / Sxx  # p = slope
    p_values[:,:,1] = (Sy - p_values[:,:,0] * Sx) / n  # rho = intercept
    p_values[:,:,2] = p_values[:,:,1] / (1 - p_values[:,:,0])  # DASF
    p_values[:,:,3] = Sxy / np.sqrt(Sxx * Syy)  # R = Pearson's correlation coefficient

def p_forpixel_old(hypdata, refspectrum, p_values):
    """ the actual calculation of p (fitting a line).
        DEPRECATED, please use the similar newer function by Olli instead!

    Several options possible, based on different functions available in numpy.
    All include some overhead (computation of unneeded quantities)

    Args:
        hypdata: a ndarray with spectrum
        refspectrum: the reference spectrum with the same wavelengths
        p_values: output, ndarray of length 4
            0:slope 1:intercept 2: DASF 3:R
    """
    y_DASF = hypdata / refspectrum

    # linear regression with scipy
    # p_model = stats.linregress( hypdata, y_DASF )
    # p_values[0] = p_model.slope
    # p_values[1] = p_model.intercept
    # p_values[3] = p_model.rvalue

    # linear regression with numpy
    iph = np.ones_like(hypdata)  # placeholders for linalg.lstsq
    p_values[0], p_values[1] = np.linalg.lstsq(np.vstack([hypdata, np.ones_like(hypdata)]).T, y_DASF.T)[0]
    p_values[3] = np.corrcoef(hypdata, y_DASF)[0][1]
    p_values[2] = p_values[1] / (1 - p_values[0])  # DASF


def p(hypdata, refspectrum):
    """ Calculate p (by fitting a line).
        DEPRECATED, please use the similar newer function by Olli instead!

    Based on p_forpixel()
    Assumes that the input data is already spectrally subset

    Args:
        hypdata: hyperspectral reflectance data as np.array
        refspectrum: the reference spectrum, has to be same length as hypdata

    Returns:
        ndarray of length 4: 0:slope 1:intercept 2: DASF 3:R
    """
    if len(hypdata.shape) == 1:
        axis=0
    elif len(hypdata.shape) == 2:
        axis=1
    elif len(hypdata.shape) == 3:
        axis=2
    else:
        raise Exception('The length of hypdata.shape must be less than 3!')
    y_DASF = hypdata / refspectrum

    n = len(refspectrum)
    Sx = hypdata.sum(axis=axis)
    Sxx = (hypdata * hypdata).sum(axis=axis) - Sx * Sx / n
    Sy = y_DASF.sum(axis=axis)
    Syy = (y_DASF * y_DASF).sum(axis=axis) - Sy * Sy / n
    Sxy = (hypdata * y_DASF).sum(axis=axis) - Sx * Sy / n
    p_out = Sxy / Sxx  # p = slope
    rho_out = (Sy - p_out * Sx) / n  # rho = intercept
    DASF_out = rho_out / (1 - p_out)  # DASF
    R_out = Sxy / np.sqrt(Sxx * Syy)  # R = Pearson's correlation coefficient

    return (p_out, rho_out, DASF_out, R_out)


def pC_forpixel(hypdata, refspectrum):
    """Calculate rho, p and c using ordinary linear regression for a single pixel.

    Uses ordinary least squares to minimize the residual sum of squares for the equation :math:`y = X \beta + \epsilon`.
    Assumes that the input data is already spectrally subset.

    Args:
        hypdata: hyperspectral reflectance data as np.array
        refspectrum: the reference spectrum, has to be same length as hypdata

    Returns:
        ndarray of length 4: 0:p 1:rho 2: c 3:RSS
    """
    X = np.array([np.ones(len(refspectrum)), hypdata, 1./refspectrum]) # Matrix of independent variables
    y = hypdata / refspectrum # Vector of dependent variables
    C = X.dot(X.T) # Correlation matrix for the regressors
    r = X.dot(y) # Vector of correlations for y
    
    try:
        beta = np.linalg.inv(C).dot(r)
    except: # If the matrix C is singular, use the Moore-Penrose pseudoinverse
        beta = np.linalg.pinv(C).dot(r)
    
    y_pred = X.T.dot(beta)
    residual = y - y_pred
    RSS = residual.dot(residual)
    return (beta[1], beta[0], beta[2], RSS)


def pC(hypdata, refspectrum):
    """ Calculate rho, p and c.

    Estimates parameters `rho`, `p`, and `c` for :math:`R/S = \rho + p R c/S` by finding the roots of the residual sum of squares (RSS) analytically.
    Assumes that the input data is already spectrally subset

    Args:
        hypdata: hyperspectral reflectance data as np.array
        refspectrum: the reference spectrum, has to be same length as hypdata

    Returns:
        ndarray of length 4: 0:p 1:rho 2: c 3:RSS
    """

    if len(hypdata.shape) == 1:
        axis=0
    elif len(hypdata.shape) == 2:
        axis=1
    elif len(hypdata.shape) == 3:
        axis=2
    else:
        raise Exception('The length of hypdata.shape must be less than 3!')
    y_DASF = hypdata / refspectrum

    # Calculating the means takes a while... (>150 microseconds on my PC)
    y_DASF = hypdata / refspectrum
    
    YM = np.mean(y_DASF, axis=axis)
    RM = np.mean(hypdata, axis=axis)
    YRM = np.mean(y_DASF*hypdata, axis=axis)
    RM2 = np.mean(hypdata*hypdata, axis=axis)
    AM = np.mean(1./refspectrum, axis=0)
    AM2 = np.mean(1./(refspectrum*refspectrum), axis=0)
    YAM = np.mean(y_DASF/refspectrum, axis=axis)

    # Everything from below here is fast (<10 microseconds on my PC)
    numeC = RM*RM - RM2
    C1 = (YM * RM - YRM) / numeC
    C3 = RM * AM - YM
    C2 = -C3 / numeC

    c_out = (YM * AM - YAM - C1*C3) / (C2*C3 + AM * AM - AM2)
    p_out = C1 + C2*c_out
    rho_out = YM - p_out*RM - c_out*AM

    if len(hypdata.shape) == 1:
            estimate = rho_out + p_out*hypdata + c_out/refspectrum
    elif len(hypdata.shape) == 2:
        estimate = rho_out[:,None] + p_out[:,None]*hypdata + c_out[:,None]/refspectrum
    else:
        estimate = rho_out[:,:,None] + p_out[:,:,None]*hypdata + c_out[:,:,None]/refspectrum
        
    residual = y_DASF - estimate
    RSS = np.sum(residual*residual, axis=axis) # Residual sum of squares
    return (p_out, rho_out, c_out, RSS)


def pC_old(BRF,w, wl=None, wl_fit=(670,710,790), verbose=False):
    """ Fit the p-equation with a constant reflectance component.
        DEPRECATED, please use the similar newer function by Olli instead!

    Fits to the data the equation BRF/w=pBRF+rho+C/w by solving a system of linear equations.
    Three equations are needed for the three parmaters, selected as the first, the last, and the
    location of the minimum BRF/w. The function assumes that BRF and w are appropriate subsets,
    i.e., correspond to the wavelngths used in fitting, and are at least of length 3.


    Args:
        BRF: hypdata, hyperspectral reflectance data as np.array
        w: refspectrum, the reference spectrum, has to be same length as BRF
        wl: wavelengths of the data in BRF and w. If given, used for selecting wavelengths [nm]
        wl_fit: three wavelengths used for fitting (BRF-C)/w [nm]
        verbose: whether to print output on band selection

    Returns:
        ndarray of length 5: 0:slope 1:intercept 2: DASF 3:R 4:C
    """

    y_DASF = BRF / w

    # if wavelength info given, use standard wavelengths
    # otherwise, use first, last and sth from the middle (e.g., with minimum BRF/w)
    #   alternatively, selection based on w could be considered in the future
    if wl is not None:
        i_1 = np.argmin( np.abs(wl-wl_fit[0]) )
        i_2 = np.argmin( np.abs(wl-wl_fit[1]) )
        i_3 = np.argmin( np.abs(wl-wl_fit[2]) )
        if i_2 < 1:
            i_2 = 1
        if i_1 == i_2:
            i_1 = 0
        if i_3 == i_2:
            i_3 = i_3 +1
            if i_3 == len(wl):
                i_3 = i_3 - 1
                i_2 = i_2 - 1
                if i_1 == i_2:
                    i_1 = i_1 - 1
    else:
        i_1 = 0
        i_3 = len(wl)-1
        i_2 = argmin( y_DASF )
        if i_2==i_1 or i_2==i_3:
            i_2 = int( len(BRF)/2 )

    if verbose:
        print("pC(): using ref. albedos {:5.2f},{:5.2f},{:5.2f}"
            .format(w[i_1],w[i_2],w[i_3]), end="")
        if wl is not None:
            print(", wl={:6.1f},{:6.1f},{:6.1f}"
            .format(wl[i_1],wl[i_2],wl[i_3]))
        else:
            print("\n")
    # the coefficients for the three equations
    # x[0]=p, x[1]=rho_primed, x[2]=C
    #  rho_primed = rho-pC
    eq1_lhs = np.array([ BRF[i_1], 1, 1/w[i_1] ])
    eq2_lhs = np.array([ BRF[i_2], 1, 1/w[i_2] ])
    eq3_lhs = np.array([ BRF[i_3], 1, 1/w[i_3] ])
    # the system of equations: lhs*x=rhs
    lhs = np.stack( [eq1_lhs, eq2_lhs, eq3_lhs] )
    rhs = np.array( [y_DASF[i_1], y_DASF[i_2], y_DASF[i_3] ] )

    x = np.linalg.solve( lhs, rhs )
    p = x[0]
    C = x[2]
    rho = x[1] + p*C
    DASF = rho/(1-p)
    # correlation measures the linearity after subtracting C from BRF
    R = np.corrcoef( (BRF-C)/w, (BRF-C) )[0,1]
    return np.array([ p, rho, DASF, R, C ])

def referencealbedo_prospectparams():
    """Returns the PROSPECT parameters required for creating the  reference albedo.

    Based on Knyazikhin et al. (2013), PNAS, section "SI Text 4" (Supporting Information)
    all prospect versions in prospect have four positional arguments N,Cab,Cw,Cm

    Returns:
        list of 4 values corresponding to the prospect positional parameters
    """
    return (1.2, 16, 0.005, 0.002)

def referencealbedo( wl=None, model="PROSPECTCp" ):
    """ Generate the NON-TRANSFORMED reference leaf albedo using prospect5.

    See also the TRANSFORMED albedo generated by referencealbedo_transformed()
    See Knyazikhin et al. (2013), PNAS, SI Text 4 for model parameters

    Args:
        wl: np.array, input wavelengths. If not set, the whole PROSPECT range is used
        model: str, model name, defaults to "PROSPECT5"

    Returns:
        np.array, leaf albedo (refl+trans)
    """
    # possible options in prospect.py
    # prospect_Cp(N,Cab,Cw,Cm,Car,Cbrown,Anth,Cp,Ccl), the old PROSPECT (1996)
    # prospect_D (N,Cab,Cw,Cm,Car,Cbrown,Anth)
    # prospect_4 (N,Cab,Cw,Cm)
    # prospect_5 (N,Cab,Cw,Cm,Car)
    # prospect_5B(N,Cab,Cw,Cm,Car,Cbrown)
    # reference albedo calculated with default N, Cab=16, Cw=0.005, Cm=0.002, others zero.
    #    actually, N should not matter as it mostly changes leaf reflectance/transmittance ratio (Lewis & Disney 2007)
    # note -- all prospect versions in prospect have four positional arguments N,Cab,Cw,Cm
    prospect_in = referencealbedo_prospectparams()

    if model == "PROSPECT5":
        M = prospect.prospect_5(*prospect_in)
    elif model == "PROSPECT4":
        M = prospect.prospect_4(*prospect_in)
    elif model == "PROSPECTD":
        M = prospect.prospect_D(*prospect_in)
    elif model == "PROSPECT5B":
        M = prospect.prospect_5B(*prospect_in)
    else:
        M = prospect.prospect_Cp(*prospect_in)

    if wl is None:
        refalbedo = M[:,1] + M[:,2]
    else:
        # wl is given to the function
        # interpolate to the values in wl
        R_interp = np.interp(wl, M[:, 0], M[:, 1])
        T_interp = np.interp(wl, M[:, 0], M[:, 2])
        refalbedo = R_interp + T_interp
    return refalbedo

def referencealbedo_transformed( wl=None, model="PROSPECTCp"):
    """Returns the *TRANSFORMED* reference albedo based on the selected PROSPECT model.

    Transformation is based on the paper by Lewis & Disney (2007) in RSE

    Args:
        wl: the wavelengths used. If None, output with RPOSPECT wavelengths
        model: str with the flavor of PROSPECT to be used; see referencealbedo() for details

    Returns:
        ndarray with the albedo values
    """

    # start by getting the non-transformed albedo for teh whole wavelength range
    refalbedo = referencealbedo( model=model )

    # transform it using Eq. (11) by Lewis and Disney (2007)
    # we need w_inf; and for this we need the wax refractive index n
    if model == "PROSPECT5":
        n = prospect.P5_refractive
    elif model == "PROSPECT4":
        n = prospect.P4_refractive
    elif model == "PROSPECTD":
        n = prospect.PD_refractive
    elif model == "PROSPECT5B":
        n = prospect.P5_refractive
    else:
        n = prospect.Pcp_refractive

    w_inf = -0.0492 -0.00618*n + 0.04836*n**2 # fraction of scattering from leaf surface only
    refalbedo_tr = ( refalbedo - w_inf ) / ( 1 - w_inf )

    if wl is not None:
        # wl is given to the function
        # we need to interpolate to the desired wavelengths
        # no sanity checks made
        refalbedo_tr = np.interp(wl, reference_wavelengths(), refalbedo_tr)
    return refalbedo_tr

def leafalbedo_LD(Cab,Cw,Cm,Car=0,Cbrown=0,Canth=0,Cp=0,Ccl=0, model="PROSPECTCp", transformed=False, correctFresnel=True ):
    """ Implements the Lewis and Disney (2007, RSE) p-based approximation to leaf spectral albedo.

    The default value is the PROSPECT published in 1996 cited by Lewis & Disney (2007)
    p varies somewhat with wavelength (or, technically, the leaf wax refractive index)
    input params are the same as for PROSPECT

    Args:
        Cab: Leaf Chlorophyll a+b content [ug/cm2],
        Cw: Leaf Equivalent Water content [cm] (or [g/cm2] for fresh leaves)
        Cm: Leaf dry Mass per Area [g/cm2]
        Car: Leaf Carotenoids content [ug/cm2]
        Cbrown: Fraction of brown leaves
        Canth: Leaf Anthocyanins content [ug/cm2]
        Cp: Leaf protein content [g/cm2]
        Ccl: Leaf cellulose and lignin content [g/cm2]
        model: the PROPSECT flavor to use, see code for options
        transformed: whether to return the "transformed" albedo
            transformed albedo is defined by Lewis & Disney as the probability of a photon being
            scattered from the leaf given that it interacts with internal leaf constituents
        correctFresnel: Whether to use a more accurate expression for the surface reflection
            than the approximation by Lewis and Disney

    Returns:
        np.array of leaf albedo (length: full prospect range)
    """

    # load the model component spectra and refractive index
    if model == "PROSPECT5":
        n = prospect.P5_refractive
        kCab = prospect.P5_k_Cab
        kw = prospect.P5_k_Cw
        km = prospect.P5_k_Cm
        kCar = prospect.P5_k_Car
        kbrown = prospect.P5_k_Brown
        kanth = 0
        kp = 0
        kcl = 0
    elif model == "PROSPECT4":
        n = prospect.P4_refractive
        kCab = prospect.P4_k_Cab
        kw = prospect.P4_k_Cw
        km = prospect.P4_k_Cm
        kCar = 0
        kbrown = 0
        kanth = 0
        kp = 0
        kcl = 0
    elif model == "PROSPECTD":
        n = prospect.PD_refractive
        kCab = prospect.PD_k_Cab
        kw = prospect.PD_k_Cw
        km = prospect.PD_k_Cm
        kCar = prospect.PD_k_Car
        kbrown = prospect.PD_k_Brown
        kanth = prospect.PD_k_Anth
        kp = 0
        kcl = 0
    elif model == "PROSPECT5B":
        n = prospect.P5_refractive
        kCab = prospect.P5_k_Cab
        kw = prospect.P5_k_Cw
        km = prospect.P5_k_Cm
        kCar = prospect.P5_k_Car
        kbrown = prospect.P5_k_Brown
        kanth = 0
        kp = 0
        kcl = 0
    else:
        n = prospect.Pcp_refractive
        kCab = prospect.Pcp_k_Cab
        kw = prospect.Pcp_k_Cw
        km = prospect.Pcp_k_Cm
        kCar = prospect.Pcp_k_Car
        kbrown = prospect.Pcp_k_Brown
        kanth = prospect.Pcp_k_Anth
        kp = prospect.Pcp_k_Cp # do not use the _orig version as the latter requires a weight in absorption calculations
        kcl = prospect.Pcp_k_Ccl # do not use the _orig version as the latter requires a weight in absorption calculations
           # the decision no to use _orig was based on a quick inspection of the code from EnMapBox

    if correctFresnel:
        # use Fresnel reflectance for normal incidence
        w_inf = fresnel_normal(n)
        # an alternative would be to assume 30 deg normal incidence, but this makes little difference
        #   and is slower
        # w_inf = fresnel_general(n, 30 * np.pi / 180)
    else:
        w_inf = -0.0492 -0.00618*n + 0.04836*n**2 # eq. 12, approximation for the fraction of scattering from leaf surface only
        # Note: it seems that eq. 12 is an overestimation, it can be larger than prospect-predicted reflectance in blue (at large n).
    A = Cab*kCab + Cw*kw + Cm*km + Car*kCar + Cbrown*kbrown + Canth*kanth + Cp*kp + Ccl*kcl # eq. 10
    a = 1.3618 - 0.02294*n + 0.01299*n**2 # eq. 13c
    W_leaf = np.exp( -a*A ) # eq. 13b
    p_leaf = -1.2523 + 2.2307*n - 0.6094*n**2 # eq. 15
    w_transformed = (1-p_leaf)*W_leaf/(1-p_leaf*W_leaf) # eq. 13a
    if transformed:
        return w_transformed
    else:
        # solve w_transformed for w, total leaf albedo
        w = w_transformed*(1-w_inf) + w_inf
        return w

def fresnel_general( n_wax, theta, polratio=1):
    """ Calculate the Fresnel (specular) reflectance at a surface with a given spectrally variable refractive index

    Args:
        n_wax: np.array of wax refraction index
        theta: floating-point, incidence angle in radians
        polratio: the ratio of parallel to cross polarization

    Returns:
        np.array of reflectance factor, same length as n_wax
    """

    if theta==0:
        # for normal incidence, the equations below generate nan (0/0)
        #  use the much simpler separate function for this special case
        return fresnel_normal( n_wax )
    # the angle at which light travels in the material after being transmitted through the surface
    theta_t = np.arcsin( np.sin(theta)/n_wax )
    # the fractions of irradiance of different polarization
    crossfraction = 1.0 / (1+polratio)
    paralfraction = 1.0 - crossfraction
    # reflectance is calculated separately for the two polarizations and averaged
    r_paral = np.tan(theta - theta_t) ** 2 / np.tan(theta + theta_t) ** 2
    r_cross = np.sin(theta - theta_t) ** 2 / np.sin(theta + theta_t) ** 2
    r = r_paral*paralfraction + r_cross*crossfraction
    return r

def fresnel_normal( n_wax ):
    """ Calculate the Fresnel (specular) reflectance at normal incidence.

    Args:
        n_wax: np.array of wax refraction index

    Returns:
        np.array of reflectance factor, same length as n_wax
    """
    return ( (1-n_wax)/(1+n_wax) )**2


def reference_wavelengths():
    """ Returns the full wavelength range used for calculating the reference albedo.

    That is, the wavelength range for the PROSPECT model.
    A wrapper for prospect_wavelengths() function to allow the user avoid importing the prospect module directly

    Returns:
        ndarray of wavelengths in nanometers
    """
    return prospect.prospect_wavelengths()

def transform_albedo( albedo, refalbedo, wl=None ):
    """Transforms the albedo, i.e., forces it to cross referencealbedo at referencealbedo=1.

    Details:
    M.A. Schull et al. / Journal of Quantitative Spectroscopy & Radiative Transfer 112 (2011) 736–750
        see Appendix A

    Args:
        albedo: TBW
        refalbedo: TBW

    Returns:
        np.array of the transformed albedo
    """

    # perform a very simple linear regression

    n = hypdata.shape[0]
    Sx = hypdata.sum()
    Sxx = (hypdata * hypdata).sum() - Sx * Sx / n
    Sy = y_DASF.sum()
    Syy = (y_DASF * y_DASF).sum() - Sy * Sy / n
    Sxy = (hypdata * y_DASF).sum() - Sx * Sy / n
    p_values[0] = Sxy / Sxx  # p = slope
    p_values[1] = (Sy - p_values[0] * Sx) / n  # rho = intercept