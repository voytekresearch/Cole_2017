"""
util.py
Some functions for analyzing data in this repo
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy import io
import os


def loadmeta():
    """Load meta data for analysis of PD data.

    Returns
    -------
    Fs : float
        Sampling rate (Hz)
    t : numpy array
        time array corresponding to the ecog signals
    Spd : int
        Number of PD subjects
    Sdy : int
        Number of dystonia subjects (unused in this analysis)
    flo : 2-element tuple
        frequency limits of the beta range (Hz)
    fhi : 2-element tuple
        frequency limits for the high gamma range (Hz)
    """

    Fs = 1000.  # Sampling rate (Hz)
    t = np.arange(0, 30, 1 / Fs)  # Time series (seconds)
    Spd = 23
    Sdy = 9
    flo = (13, 30)
    fhi = (50, 200)
    return Fs, t, Spd, Sdy, flo, fhi


def loadPD(filepath='data.mat'):
    '''
    Load the data after the following preprocessing:
    1. Low-pass filter at 200Hz
    2. Notch filter at high frequency peaks for each subject

    Parameters
    ----------
    filepath : string
        path to saved data

    Returns
    -------
    ecog : dict
        Pre-processed voltage traces
        'B' : subject-by-time array for PD patients pre-DBS
        'D' : subject-by-time array for PD patients on DBS
    '''
    data = io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    ecog = _blankecog()
    ecog['B'] = data['B']
    ecog['D'] = data['D']
    return ecog


def _lowpass200_all(ecog):
    """
    Apply a 200Hz low-pass filter to all data
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    # Define low pass filter
    def lowpass200(x, Fs=1000, Fc=200, Ntaps=250):
        taps = sp.signal.firwin(Ntaps, Fc / (Fs / 2.))
        return np.convolve(taps, x, 'same')

    # Apply low pass filter to all data
    ecoglp = _blankecog()
    for s in range(S):
        ecoglp['B'][s] = lowpass200(ecog['B'][s])
        ecoglp['D'][s] = lowpass200(ecog['D'][s])

    return ecoglp


def _remove_hifreqpeaks_all(ecog, order=3):
    """
    Apply notch filters to remove high frequency peaks in all data
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    # Define notch filter method
    def hifreqnotch(x, cf, bw, Fs, order):
        '''
        Notch Filter the time series x with a butterworth with center frequency cf
        and bandwidth bw
        '''
        nyq_rate = Fs / 2.
        f_range = [cf - bw / 2., cf + bw / 2.]
        Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
        b, a = sp.signal.butter(order, Wn, 'bandstop')
        return sp.signal.filtfilt(b, a, x)

    def multiplenotch(x, cfs, bws, Fs, order):
        """
        Perform all notch filters for a given piece of data
        """
        Nfilters = len(cfs)
        for f in range(Nfilters):
            x = hifreqnotch(x, cfs[f], bws[f], Fs, order)
        return x

    # Define notch filter for each subject
    hicfPD, hibwPD = _hifreqparams()

    # Apply notch filter to all data
    ecoghf = _blankecog()
    for s in range(S):
        ecoghf['B'][s] = multiplenotch(
            ecog['B'][s], hicfPD[s], hibwPD[s], Fs=Fs, order=order)
        ecoghf['D'][s] = multiplenotch(
            ecog['D'][s], hicfPD[s], hibwPD[s], Fs=Fs, order=order)

    return ecoghf


def _hifreqparams():
    """
    Return the parameters of the notch filters for the data
    These parameters were obtained by visual inspection of the data
    """
    hicfPD = [[118.8],  # S0
              [164.8, 179.7],  # S1
              [69.1, 117, 138.2, 165.2, 166.3, 186.2, 207],  # S2
              [119.8, 170],  # S3
              [106.9, 213.7],  # S4
              [119.8, 161.7, 166.6, 180, 192.8, 211],  # S5
              [79.5, 113.6, 115, 116.1, 119.8, 125.5, 151, 152.3,
                  153.3, 168, 183.2, 185, 212, 213.8, 215],  # S6
              [119.8, 146.7, 179.7],  # S7
              [127.6, 148.5, 176.1, 214],  # S8
              [119.8, 143, 179.7, 189.2],  # S9
              [54, 79, 118.8, 119.8, 145.5, 172.6, 175.7, 177.5, 179.8],  # S10
              [144.3],  # S11
              [140.9, 179.8, 186.6],  # S12
              [119.8, 147.8, 151.8, 153.6, 179.7],  # S13
              [140.5, 174.3],  # S14
              [155.3, 215],  # S15
              [119.8, 121.6, 129.7, 161.8, 179.7, 194.3],  # S16
              [106, 159, 167.4, 178.4, 189.5, 212.4],  # S17
              [112.1, 128.4],  # S18
              [119.8, 132.6],  # S19
              [168],  # S20
              [119.8, 120.3],  # S21
              [92.5, 112.3, 129.6, 148, 160.8, 167,
               168.5, 176.1, 185, 204]  # S22
              ]
    hibwPD = [[0.5],  # S0
              [1, 0.5],  # S1
              [0.5, 3, 0.5, 0.5, 0.5, 2, 8],  # S2.
              [0.5, 3],  # S3
              [0.5, 0.5],  # S4
              [0.5, 1, 0.5, 4, 0.5, 1],  # S5
              [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,
                  1, 4, 0.5, 2, 0.5, 0.5, 1],  # S6
              [0.5, 0.5, 0.5],  # S7
              [0.5, 0.5, 0.5, 1],  # S8
              [0.5, 0.5, 0.5, 0.5],  # S9
              [1, 0.5, 0.5, 0.5, 2, 1, 1, 1, .5],  # S10
              [1],  # S11
              [2, 0.5, 1],  # S12
              [0.5, 0.5, 1, 1, 0.5],  # S13
              [0.5, 1],  # S14
              [0.5, 2],  # S15
              [0.5, 1.5, 1, 1, 0.5, 1],  # S16
              [8, 8, 0.5, 0.5, 0.5, 1],  # S17
              [0.5, 0.5],  # S18
              [0.5, 1],  # S19
              [4],  # S20
              [0.5, 0.5],  # S21
              [1, 0.5, 1, 4, 0.5, 3, 0.5, 0.5, 5, 6]  # S22
              ]
    return hicfPD, hibwPD


def _blankecog(S=23, dtype=object):
    ecog = {}
    ecog['B'] = np.zeros(S, dtype=dtype)
    ecog['D'] = np.zeros(S, dtype=dtype)
    return ecog


def normalize_signal_power(ecog):
    Fs, t, S, Sdy, flo, fhi = loadmeta()
    for s in range(S):
        ecog['B'][s] = ecog['B'][s] / np.sqrt(np.sum(ecog['B'][s]**2))
        ecog['D'][s] = ecog['D'][s] / np.sqrt(np.sum(ecog['D'][s]**2))
    return ecog


def measure_shape(ecog, boundaryS=100, ampPC=0, widthS=5, esrmethod='aggregate'):
    """This function calculates the shape measures calculated for analysis
    of the PD data set

    1. Peak and trough times
    2. Peak and trough sharpness
    3. Sharpness ratio
    4. Steepness ratio
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()
    from shape import findpt, ex_sharp, esr, rd_steep, rdsr
    pks = _blankecog()
    trs = _blankecog()
    pksharp = _blankecog()
    trsharp = _blankecog()
    esrs = _blankecog(dtype=float)
    peaktotrough = _blankecog(dtype=float)
    risteep = _blankecog()
    desteep = _blankecog()
    rdsrs = _blankecog(dtype=float)
    risetodecay = _blankecog(dtype=float)

    for s in range(S):
        pks['B'][s], trs['B'][s] = findpt(
            ecog['B'][s], flo, Fs=Fs, boundary=boundaryS)
        pksharp['B'][s] = ex_sharp(
            ecog['B'][s], pks['B'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        trsharp['B'][s] = ex_sharp(
            ecog['B'][s], trs['B'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        esrs['B'][s] = np.log10(esr(
            ecog['B'][s], pks['B'][s], trs['B'][s], widthS, esrmethod=esrmethod, ampPC=ampPC))
        peaktotrough['B'][s] = np.mean(
            pksharp['B'][s]) / np.mean(trsharp['B'][s])
        risteep['B'][s], desteep['B'][s] = rd_steep(
            ecog['B'][s], pks['B'][s], trs['B'][s])
        rdsrs['B'][s] = np.log10(rdsr(risteep['B'][s], desteep['B'][s]))
        risetodecay['B'][s] = np.mean(
            risteep['B'][s]) / np.mean(desteep['B'][s])

        pks['D'][s], trs['D'][s] = findpt(
            ecog['D'][s], flo, Fs=Fs, boundary=boundaryS)
        pksharp['D'][s] = ex_sharp(
            ecog['D'][s], pks['D'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        trsharp['D'][s] = ex_sharp(
            ecog['D'][s], trs['D'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        esrs['D'][s] = np.log10(esr(
            ecog['D'][s], pks['D'][s], trs['D'][s], widthS, esrmethod=esrmethod, ampPC=ampPC))
        peaktotrough['D'][s] = np.mean(
            pksharp['D'][s]) / np.mean(trsharp['D'][s])
        risteep['D'][s], desteep['D'][s] = rd_steep(
            ecog['D'][s], pks['D'][s], trs['D'][s])
        rdsrs['D'][s] = np.log10(rdsr(risteep['D'][s], desteep['D'][s]))
        risetodecay['D'][s] = np.mean(
            risteep['D'][s]) / np.mean(desteep['D'][s])

    return pks, trs, pksharp, trsharp, esrs, peaktotrough, risteep, desteep, rdsrs, risetodecay


def measure_pac(ecog, flo, fhi, Fs=1000, Nlo=231, Nhi=240):
    """This function esimates PAC on the PD data
    """
    # Calculate PAC
    import pac
    S = len(ecog['B'])

    pacs = _blankecog(dtype=float)
    for s in range(S):
        pacs['B'][s] = pac.ozkurt(ecog['B'][s], ecog['B'][s], flo, fhi, fs=Fs, filter_kwargslo={
                                  'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})
        pacs['D'][s] = pac.ozkurt(ecog['D'][s], ecog['D'][s], flo, fhi, fs=Fs, filter_kwargslo={
                                  'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})

    return pacs


def measure_psd(ecog, Hzmed=1):
    """This function calculates the PSD for all signals
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    from tools.spec import fftmed

    psd = _blankecog(dtype=object)
    for s in range(S):
        f, psd['B'][s] = fftmed(ecog['B'][s], Fs=Fs, Hzmed=Hzmed)
        _, psd['D'][s] = fftmed(ecog['D'][s], Fs=Fs, Hzmed=Hzmed)
    return f, psd


def measure_power(ecog):
    """This function calculates the beta and high gamma power for all signals
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    # Calculate PSD
    from tools.spec import fftmed
    Hzmed = 0
    psd = _blankecog(dtype=object)
    for s in range(S):
        f, psd['B'][s] = fftmed(ecog['B'][s], Fs=Fs, Hzmed=Hzmed)
        _, psd['D'][s] = fftmed(ecog['D'][s], Fs=Fs, Hzmed=Hzmed)

    # Calculate beta power
    from tools.spec import calcpow
    bp = _blankecog(dtype=float)
    for s in range(S):
        bp['B'][s] = np.log10(calcpow(f, psd['B'][s], flo))
        bp['D'][s] = np.log10(calcpow(f, psd['D'][s], flo))

    # Calculate high gamma power
    hgp = _blankecog(dtype=float)
    for s in range(S):
        hgp['B'][s] = np.log10(calcpow(f, psd['B'][s], fhi))
        hgp['D'][s] = np.log10(calcpow(f, psd['D'][s], fhi))

    # Calculate total power
    tp = _blankecog(dtype=float)
    for s in range(S):
        tp['B'][s] = np.log10(np.sum(ecog['B'][s]**2))
        tp['D'][s] = np.log10(np.sum(ecog['D'][s]**2))

    return bp, hgp, tp


def measure_rigid():
    # Rigidity data
    rigidB = np.array([2, 1, 99, 2, 1, 0, 0, 2, 2, 1, 3,
                       99, 0, 1, 3, 1, 99, 0, 0, 2, 2, 2, 2])
    rigidD = np.array([0, 0, 99, 1, 0, 0, 0, 1, 1, 0, 2,
                       99, 0, 0, 2, 0, 99, 0, 0, 1, 1, 1, 0])
    return rigidB, rigidD


def calculate_comodulogramPAC(ecog, comodkwargs=None):
    """ Calculate the PAC measures for all signals in PD data using comodulogram method
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()
    if comodkwargs is None:
        comodkwargs = {}

    cpac = _blankecog(dtype=float)
    for s in range(S):
        cpac['B'][s] = _comodPAC(ecog['B'][s], flo, fhi, Fs, **comodkwargs)
        cpac['D'][s] = _comodPAC(ecog['D'][s], flo, fhi, Fs, **comodkwargs)
    return cpac


def _comodPAC(x, flo, fhi, Fs, dp=2, da=4, w_lo=7, w_hi=7, pac_method='mi_tort'):

    # Calculate comodulogram
    # Filter order was based off presuming deHemptinne 2015 used the default FIR1 filter order
    # using eegfilt:
    # https://sccn.ucsd.edu/svn/software/eeglab/functions/sigprocfunc/eegfilt.m
    from pac import comodulogram
    comod = comodulogram(x, x, flo, fhi, dp, da, w_lo=w_lo, w_hi=w_hi, fs=Fs, pac_method='mi_tort')
    return np.mean(comod)


def firf(x, f_range, fs=1000, w=3, rmvedge=True):
    """
    Filter signal with an FIR filter
    *Like fir1 in MATLAB
    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles
        of the oscillation whose frequency is the low cutoff of the
        bandpass filter
    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    nyq = np.float(fs / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. '
            'Provide more data or a shorter filter.')

    # Perform filtering
    taps = sp.signal.firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = sp.signal.filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    if rmvedge:
        return _remove_edge(x_filt, Ntaps)
    else:
        return x_filt


def _remove_edge(x, N):
    """
    Calculate the number of points to remove for edge artifacts

    x : array
        time series to remove edge artifacts from
    N : int
        length of filter
    """
    N = int(N)
    return x[N:-N]


def morletT(x, f0s, Fs, w=7, s=.5):
    """
    Calculate the time-frequency representation of the signal 'x' over the
    frequencies in 'f0s' using morlet wavelets
    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    s : float
        Scaling factor
    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x
    """
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')

    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F, T], dtype=complex)
    for f in range(F):
        mwt[f] = morletf(x, f0s[f], Fs, w=w, s=s)

    return mwt


def morletf(x, f0, Fs, w=7, s=.5, M=None, norm='sss'):
    """
    Convolve a signal with a complex wavelet
    The real part is the filtered signal
    Taking np.abs() of output gives the analytic amplitude
    Taking np.angle() of output gives the analytic phase
    x : array
        Time series to filter
    f0 : float
        Center frequency of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2
    Returns
    -------
    x_trans : array
        Complex time series
    """
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')

    if M is None:
        M = w * Fs / f0

    morlet_f = sp.signal.morlet(M, w=w, s=s)
    morlet_f = morlet_f

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(x, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode='same')

    return mwt_real + 1j*mwt_imag


def simphase(T, flo, w=3, dt=.001, randseed=0, returnwave=False):
    """ Simulate the phase of an oscillation
    The first and last second of the oscillation are simulated and taken out
    in order to avoid edge artifacts in the simulated phase

    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    flo : 2-element array (lo,hi)
        frequency range of simulated oscillation
    dt : float
        time step of simulated oscillation
    returnwave : boolean
        option to return the simulated oscillation
    """
    from tools.spec import bandpass_default
    np.random.seed(randseed)
    whitenoise = np.random.randn(int((T+2)/dt))
    theta, _ = bandpass_default(whitenoise, flo, 1/dt, rmv_edge=False, w=w)

    if returnwave:
        return np.angle(sp.signal.hilbert(theta[int(1/dt):int((T+1)/dt)])), theta[int(1/dt):int((T+1)/dt)]
    else:
        return np.angle(sp.signal.hilbert(theta[int(1/dt):int((T+1)/dt)]))


def simfiltonef(T, f_range, Fs, N, samp_buffer=10000):
    """ Simulate a band-pass filtered signal with brown noise
    Input suggestions: f_range=(2,None), Fs=1000, N=1000

    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    Fs : float
        oscillation sampling rate
    f_range : 2-element array (lo,hi)
        frequency range of simulated data
        if None: do not filter
    N : int
        order of filter
    """

    if f_range is None:
        # Do not filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs))
        return brownN
    elif f_range[1] is None:
        # High pass filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs+N*2))
        # Filter
        nyq = Fs / 2.
        if N % 2 == 0:
            print('NOTE: Increased high-pass filter order by 1 in order to be odd')
            N += 1

        taps = sp.signal.firwin(N, f_range[0] / nyq, pass_zero=False)
        brownNf = sp.signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]

    else:
        # Bandpass filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs+N*2))
        # Filter
        nyq = Fs / 2.
        taps = sp.signal.firwin(N, np.array(f_range) / nyq, pass_zero=False)
        brownNf = sp.signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]


def simbrown(N):
    """Simulate a brown noise signal (power law distribution 1/f^2)
    with N samples"""
    wn = np.random.randn(N)
    return np.cumsum(wn)
