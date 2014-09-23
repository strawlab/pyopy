# coding=utf-8
from pyopy.matlab_utils import MatlabSequence
from oscail.common.config import Configurable


def HCTSA_CO_AddNoise(eng, x, tau=1, meth='quantiles', nbins=20):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes changes in the automutual information function with the addition of
    % noise to the input time series.
    % Adds Gaussian-distributed noise to the time series with increasing standard
    % deviation, eta, across the range eta = 0, 0.1, ..., 2, and measures the
    % mutual information at each point using histograms with
    % nbins bins (implemented using CO_HistogramAMI).
    % 
    % The output is a set of statistics on the resulting set of automutual
    % information estimates, including a fit to an exponential decay, since the
    % automutual information decreases with the added white noise.
    % 
    % Can calculate these statistics for time delays 'tau', and for a number 'nbins'
    % bins.
    % 
    % This algorithm is quite different, but was based on the idea of 'noise
    % titration' presented in: "Titration of chaos with added noise", Chi-Sang Poon
    % and Mauricio Barahona P. Natl. Acad. Sci. USA, 98(13) 7107 (2001)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac2',
                                                      'fitexpa',
                                                      'fitexpadjr2',
                                                      'fitexpb',
                                                      'fitexpr2',
                                                      'fitexprmse',
                                                      'fitlina',
                                                      'fitlinb',
                                                      'meanch',
                                                      'mse',
                                                      'pcrossmean',
                                                      'pdec']}
    if tau is None:
        out = eng.run_function(1, 'CO_AddNoise', x, )
    elif meth is None:
        out = eng.run_function(1, 'CO_AddNoise', x, tau)
    elif nbins is None:
        out = eng.run_function(1, 'CO_AddNoise', x, tau, meth)
    else:
        out = eng.run_function(1, 'CO_AddNoise', x, tau, meth, nbins)
    return outfunc(out)


class CO_AddNoise(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes changes in the automutual information function with the addition of
    % noise to the input time series.
    % Adds Gaussian-distributed noise to the time series with increasing standard
    % deviation, eta, across the range eta = 0, 0.1, ..., 2, and measures the
    % mutual information at each point using histograms with
    % nbins bins (implemented using CO_HistogramAMI).
    % 
    % The output is a set of statistics on the resulting set of automutual
    % information estimates, including a fit to an exponential decay, since the
    % automutual information decreases with the added white noise.
    % 
    % Can calculate these statistics for time delays 'tau', and for a number 'nbins'
    % bins.
    % 
    % This algorithm is quite different, but was based on the idea of 'noise
    % titration' presented in: "Titration of chaos with added noise", Chi-Sang Poon
    % and Mauricio Barahona P. Natl. Acad. Sci. USA, 98(13) 7107 (2001)
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac2',
                'fitexpa',
                'fitexpadjr2',
                'fitexpb',
                'fitexpr2',
                'fitexprmse',
                'fitlina',
                'fitlinb',
                'meanch',
                'mse',
                'pcrossmean',
                'pdec')

    def __init__(self, tau=1, meth='quantiles', nbins=20):
        super(CO_AddNoise, self).__init__(add_descriptors=False)
        self.tau = tau
        self.meth = meth
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_CO_AddNoise(engine,
                                 x,
                                 tau=self.tau,
                                 meth=self.meth,
                                 nbins=self.nbins)


def HCTSA_CO_AutoCorr(eng, x, tau=32, WhatMethod='Fourier'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the autocorrelation of an input time series, y, at a time-lag, tau
    % 
    %---INPUTS:
    % y, a scalar time series column vector.
    % tau, the time-delay. If tau is a scalar, returns autocorrelation for y at that
    %       lag. If tau is a vector, returns autocorrelations for y at that set of
    %       lags.
    % WhatMethod, the method of computing the autocorrelation: 'Fourier',
    %             'TimeDomainStat', or 'TimeDomain'.
    %       
    %---OUTPUT: the autocorrelation at the given time-lag.
    %
    %---HISTORY:
    % Ben Fulcher, 2014-03-24. Added multiple definitions for computing the
    %       autocorrelation Computing mean/std across the full time series makes a
    %       significant difference for short time series, but can produce values
    %       outside [-1,+1]. The filtering-based method used by Matlab's autocorr,
    %       is probably the best for short time series, and is now implemented here.
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if tau is None:
        out = eng.run_function(1, 'CO_AutoCorr', x, )
    elif WhatMethod is None:
        out = eng.run_function(1, 'CO_AutoCorr', x, tau)
    else:
        out = eng.run_function(1, 'CO_AutoCorr', x, tau, WhatMethod)
    return outfunc(out)


class CO_AutoCorr(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the autocorrelation of an input time series, y, at a time-lag, tau
    % 
    %---INPUTS:
    % y, a scalar time series column vector.
    % tau, the time-delay. If tau is a scalar, returns autocorrelation for y at that
    %       lag. If tau is a vector, returns autocorrelations for y at that set of
    %       lags.
    % WhatMethod, the method of computing the autocorrelation: 'Fourier',
    %             'TimeDomainStat', or 'TimeDomain'.
    %       
    %---OUTPUT: the autocorrelation at the given time-lag.
    %
    %---HISTORY:
    % Ben Fulcher, 2014-03-24. Added multiple definitions for computing the
    %       autocorrelation Computing mean/std across the full time series makes a
    %       significant difference for short time series, but can produce values
    %       outside [-1,+1]. The filtering-based method used by Matlab's autocorr,
    %       is probably the best for short time series, and is now implemented here.
    %
    ----------------------------------------
    """

    def __init__(self, tau=32, WhatMethod='Fourier'):
        super(CO_AutoCorr, self).__init__(add_descriptors=False)
        self.tau = tau
        self.WhatMethod = WhatMethod

    def eval(self, engine, x):
        return HCTSA_CO_AutoCorr(engine,
                                 x,
                                 tau=self.tau,
                                 WhatMethod=self.WhatMethod)


def HCTSA_CO_AutoCorrShape(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Outputs a set of statistics summarizing how the autocorrelation function
    % changes with the time lag, tau.
    % Outputs include the number of peaks, and autocorrelation in the
    % autocorrelation function itself.
    % 
    % INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['Nac',
                                                      'ac1',
                                                      'ac1maxima',
                                                      'ac1minima',
                                                      'ac2',
                                                      'ac3',
                                                      'actau',
                                                      'fexpabsacf_a',
                                                      'fexpabsacf_adjr2',
                                                      'fexpabsacf_b',
                                                      'fexpabsacf_r2',
                                                      'fexpabsacf_rmse',
                                                      'fexpabsacf_varres',
                                                      'flinlmxacf_a',
                                                      'flinlmxacf_adjr2',
                                                      'flinlmxacf_b',
                                                      'flinlmxacf_r2',
                                                      'flinlmxacf_rmse',
                                                      'maximaspread',
                                                      'meanabsacf',
                                                      'meanacf',
                                                      'meanmaxima',
                                                      'meanminima',
                                                      'nextrema',
                                                      'nmaxima',
                                                      'nminima',
                                                      'pextrema']}
    out = eng.run_function(1, 'CO_AutoCorrShape', x, )
    return outfunc(out)


class CO_AutoCorrShape(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Outputs a set of statistics summarizing how the autocorrelation function
    % changes with the time lag, tau.
    % Outputs include the number of peaks, and autocorrelation in the
    % autocorrelation function itself.
    % 
    % INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """

    outnames = ('Nac',
                'ac1',
                'ac1maxima',
                'ac1minima',
                'ac2',
                'ac3',
                'actau',
                'fexpabsacf_a',
                'fexpabsacf_adjr2',
                'fexpabsacf_b',
                'fexpabsacf_r2',
                'fexpabsacf_rmse',
                'fexpabsacf_varres',
                'flinlmxacf_a',
                'flinlmxacf_adjr2',
                'flinlmxacf_b',
                'flinlmxacf_r2',
                'flinlmxacf_rmse',
                'maximaspread',
                'meanabsacf',
                'meanacf',
                'meanmaxima',
                'meanminima',
                'nextrema',
                'nmaxima',
                'nminima',
                'pextrema')

    def __init__(self, ):
        super(CO_AutoCorrShape, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_CO_AutoCorrShape(engine, x)


def HCTSA_CO_CompareMinAMI(eng, x, meth='std2', nbins=MatlabSequence('2:80')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds the first minimum of the automutual information by various different
    % estimation methods, and sees how this varies over different coarse-grainings
    % of the time series.
    % 
    % The function returns a set of statistics on the set of first minimums of the
    % automutual information function obtained over a range of the number of bins
    % used in the histogram estimation, when specifying 'nbins' as a vector
    % 
    % INPUTS:
    % y, the input time series
    % 
    % meth, the method for estimating mutual information (input to CO_HistogramAMI)
    % 
    % nbins, the number of bins for the AMI estimation to compare over (can be a
    %           scalar or vector)
    % 
    % Outputs include the minimum, maximum, range, number of unique values, and the
    % position and periodicity of peaks in the set of automutual information
    % minimums.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['conv4',
                                                      'iqr',
                                                      'max',
                                                      'maxp',
                                                      'mean',
                                                      'median',
                                                      'min',
                                                      'mode',
                                                      'modef',
                                                      'nlocmax',
                                                      'nunique',
                                                      'range',
                                                      'std']}
    if meth is None:
        out = eng.run_function(1, 'CO_CompareMinAMI', x, )
    elif nbins is None:
        out = eng.run_function(1, 'CO_CompareMinAMI', x, meth)
    else:
        out = eng.run_function(1, 'CO_CompareMinAMI', x, meth, nbins)
    return outfunc(out)


class CO_CompareMinAMI(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds the first minimum of the automutual information by various different
    % estimation methods, and sees how this varies over different coarse-grainings
    % of the time series.
    % 
    % The function returns a set of statistics on the set of first minimums of the
    % automutual information function obtained over a range of the number of bins
    % used in the histogram estimation, when specifying 'nbins' as a vector
    % 
    % INPUTS:
    % y, the input time series
    % 
    % meth, the method for estimating mutual information (input to CO_HistogramAMI)
    % 
    % nbins, the number of bins for the AMI estimation to compare over (can be a
    %           scalar or vector)
    % 
    % Outputs include the minimum, maximum, range, number of unique values, and the
    % position and periodicity of peaks in the set of automutual information
    % minimums.
    % 
    ----------------------------------------
    """

    outnames = ('conv4',
                'iqr',
                'max',
                'maxp',
                'mean',
                'median',
                'min',
                'mode',
                'modef',
                'nlocmax',
                'nunique',
                'range',
                'std')

    def __init__(self, meth='std2', nbins=MatlabSequence('2:80')):
        super(CO_CompareMinAMI, self).__init__(add_descriptors=False)
        self.meth = meth
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_CO_CompareMinAMI(engine,
                                      x,
                                      meth=self.meth,
                                      nbins=self.nbins)


def HCTSA_CO_Embed2(eng, x, tau='tau'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Embeds the z-scored time series in a two-dimensional time-delay
    % embedding space with a given time-delay, tau, and outputs a set of
    % statistics about the structure in this space, including angular 
    % distribution, etc.
    % 
    %---INPUTS:
    % y, the column-vector time series
    % tau, the time-delay (can be 'tau' for first zero-crossing of ACF)
    % 
    %---OUTPUTS: include the distribution of angles between successive points in the
    % space, stationarity of this angular distribution, euclidean distances from the
    % origin, and statistics on outliers.
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['arearat',
                                                      'areas_50',
                                                      'areas_all',
                                                      'eucdm1',
                                                      'eucdm2',
                                                      'eucdm3',
                                                      'eucdm4',
                                                      'eucdm5',
                                                      'eucds1',
                                                      'eucds2',
                                                      'eucds3',
                                                      'eucds4',
                                                      'eucds5',
                                                      'hist10std',
                                                      'histent',
                                                      'mean_eucdm',
                                                      'mean_eucds',
                                                      'meanspana',
                                                      'std_eucdm',
                                                      'std_eucds',
                                                      'stdb1',
                                                      'stdb2',
                                                      'stdb3',
                                                      'stdb4',
                                                      'stdspana',
                                                      'theta_ac1',
                                                      'theta_ac2',
                                                      'theta_ac3',
                                                      'theta_mean',
                                                      'theta_std']}
    if tau is None:
        out = eng.run_function(1, 'CO_Embed2', x, )
    else:
        out = eng.run_function(1, 'CO_Embed2', x, tau)
    return outfunc(out)


class CO_Embed2(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Embeds the z-scored time series in a two-dimensional time-delay
    % embedding space with a given time-delay, tau, and outputs a set of
    % statistics about the structure in this space, including angular 
    % distribution, etc.
    % 
    %---INPUTS:
    % y, the column-vector time series
    % tau, the time-delay (can be 'tau' for first zero-crossing of ACF)
    % 
    %---OUTPUTS: include the distribution of angles between successive points in the
    % space, stationarity of this angular distribution, euclidean distances from the
    % origin, and statistics on outliers.
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """

    outnames = ('arearat',
                'areas_50',
                'areas_all',
                'eucdm1',
                'eucdm2',
                'eucdm3',
                'eucdm4',
                'eucdm5',
                'eucds1',
                'eucds2',
                'eucds3',
                'eucds4',
                'eucds5',
                'hist10std',
                'histent',
                'mean_eucdm',
                'mean_eucds',
                'meanspana',
                'std_eucdm',
                'std_eucds',
                'stdb1',
                'stdb2',
                'stdb3',
                'stdb4',
                'stdspana',
                'theta_ac1',
                'theta_ac2',
                'theta_ac3',
                'theta_mean',
                'theta_std')

    def __init__(self, tau='tau'):
        super(CO_Embed2, self).__init__(add_descriptors=False)
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_Embed2(engine,
                               x,
                               tau=self.tau)


def HCTSA_CO_Embed2_AngleTau(eng, x, maxtau=50):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Investigates how the autocorrelation of angles between successive points in
    % the two-dimensional time-series embedding change as tau varies from
    % tau = 1, 2, ..., maxtau.
    % 
    % INPUTS:
    % y, a column vector time series
    % maxtau, the maximum time lag to consider
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1_thetaac1',
                                                      'ac1_thetaac2',
                                                      'ac1_thetaac3',
                                                      'diff_thetaac12',
                                                      'max_thetaac1',
                                                      'max_thetaac2',
                                                      'max_thetaac3',
                                                      'mean_thetaac1',
                                                      'mean_thetaac2',
                                                      'mean_thetaac3',
                                                      'meanrat_thetaac12',
                                                      'min_thetaac1',
                                                      'min_thetaac2',
                                                      'min_thetaac3']}
    if maxtau is None:
        out = eng.run_function(1, 'CO_Embed2_AngleTau', x, )
    else:
        out = eng.run_function(1, 'CO_Embed2_AngleTau', x, maxtau)
    return outfunc(out)


class CO_Embed2_AngleTau(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Investigates how the autocorrelation of angles between successive points in
    % the two-dimensional time-series embedding change as tau varies from
    % tau = 1, 2, ..., maxtau.
    % 
    % INPUTS:
    % y, a column vector time series
    % maxtau, the maximum time lag to consider
    % 
    ----------------------------------------
    """

    outnames = ('ac1_thetaac1',
                'ac1_thetaac2',
                'ac1_thetaac3',
                'diff_thetaac12',
                'max_thetaac1',
                'max_thetaac2',
                'max_thetaac3',
                'mean_thetaac1',
                'mean_thetaac2',
                'mean_thetaac3',
                'meanrat_thetaac12',
                'min_thetaac1',
                'min_thetaac2',
                'min_thetaac3')

    def __init__(self, maxtau=50):
        super(CO_Embed2_AngleTau, self).__init__(add_descriptors=False)
        self.maxtau = maxtau

    def eval(self, engine, x):
        return HCTSA_CO_Embed2_AngleTau(engine,
                                        x,
                                        maxtau=self.maxtau)


def HCTSA_CO_Embed2_Basic(eng, x, tau=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Obtains a set of measures of point density in a plot of y_i against y_{i-tau}.
    %
    % INPUTS:
    % y, the input time series
    % 
    % tau, the time lag (can be set to 'tau' to set the time lag the first zero
    %                       crossing of the autocorrelation function)
    % 
    % Outputs include the number of points near the diagonal, and similarly, the
    % number of points that are close to certain geometric shapes in the y_{i-tau}, 
    % y_{tau} plot, including parabolas, rings, and circles.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['downdiag01',
                                                      'downdiag05',
                                                      'incircle_01',
                                                      'incircle_02',
                                                      'incircle_05',
                                                      'incircle_1',
                                                      'incircle_2',
                                                      'incircle_3',
                                                      'medianincircle',
                                                      'parabdown01',
                                                      'parabdown01_1',
                                                      'parabdown01_n1',
                                                      'parabdown05',
                                                      'parabdown05_1',
                                                      'parabdown05_n1',
                                                      'parabup01',
                                                      'parabup01_1',
                                                      'parabup01_n1',
                                                      'parabup05',
                                                      'parabup05_1',
                                                      'parabup05_n1',
                                                      'ratdiag01',
                                                      'ratdiag05',
                                                      'ring1_01',
                                                      'ring1_02',
                                                      'ring1_05',
                                                      'stdincircle',
                                                      'updiag01',
                                                      'updiag05']}
    if tau is None:
        out = eng.run_function(1, 'CO_Embed2_Basic', x, )
    else:
        out = eng.run_function(1, 'CO_Embed2_Basic', x, tau)
    return outfunc(out)


class CO_Embed2_Basic(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Obtains a set of measures of point density in a plot of y_i against y_{i-tau}.
    %
    % INPUTS:
    % y, the input time series
    % 
    % tau, the time lag (can be set to 'tau' to set the time lag the first zero
    %                       crossing of the autocorrelation function)
    % 
    % Outputs include the number of points near the diagonal, and similarly, the
    % number of points that are close to certain geometric shapes in the y_{i-tau}, 
    % y_{tau} plot, including parabolas, rings, and circles.
    % 
    ----------------------------------------
    """

    outnames = ('downdiag01',
                'downdiag05',
                'incircle_01',
                'incircle_02',
                'incircle_05',
                'incircle_1',
                'incircle_2',
                'incircle_3',
                'medianincircle',
                'parabdown01',
                'parabdown01_1',
                'parabdown01_n1',
                'parabdown05',
                'parabdown05_1',
                'parabdown05_n1',
                'parabup01',
                'parabup01_1',
                'parabup01_n1',
                'parabup05',
                'parabup05_1',
                'parabup05_n1',
                'ratdiag01',
                'ratdiag05',
                'ring1_01',
                'ring1_02',
                'ring1_05',
                'stdincircle',
                'updiag01',
                'updiag05')

    def __init__(self, tau=1):
        super(CO_Embed2_Basic, self).__init__(add_descriptors=False)
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_Embed2_Basic(engine,
                                     x,
                                     tau=self.tau)


def HCTSA_CO_Embed2_Dist(eng, x, tau='tau'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on the sequence of successive Euclidean distances between
    % points in a two-dimensional time-delay embedding space with a given
    % time-delay, tau.
    % 
    % Outputs include the autocorrelation of distances, the mean distance, the
    % spread of distances, and statistics from an exponential fit to the
    % distribution of distances.
    % 
    % INPUTS:
    % y, a z-scored column vector representing the input time series
    % tau, the time delay.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['d_ac1',
                                                      'd_ac2',
                                                      'd_ac3',
                                                      'd_cv',
                                                      'd_expfit_l',
                                                      'd_expfit_nlogL',
                                                      'd_expfit_sumdiff',
                                                      'd_iqr',
                                                      'd_max',
                                                      'd_mean',
                                                      'd_median',
                                                      'd_min',
                                                      'd_std']}
    if tau is None:
        out = eng.run_function(1, 'CO_Embed2_Dist', x, )
    else:
        out = eng.run_function(1, 'CO_Embed2_Dist', x, tau)
    return outfunc(out)


class CO_Embed2_Dist(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on the sequence of successive Euclidean distances between
    % points in a two-dimensional time-delay embedding space with a given
    % time-delay, tau.
    % 
    % Outputs include the autocorrelation of distances, the mean distance, the
    % spread of distances, and statistics from an exponential fit to the
    % distribution of distances.
    % 
    % INPUTS:
    % y, a z-scored column vector representing the input time series
    % tau, the time delay.
    % 
    ----------------------------------------
    """

    outnames = ('d_ac1',
                'd_ac2',
                'd_ac3',
                'd_cv',
                'd_expfit_l',
                'd_expfit_nlogL',
                'd_expfit_sumdiff',
                'd_iqr',
                'd_max',
                'd_mean',
                'd_median',
                'd_min',
                'd_std')

    def __init__(self, tau='tau'):
        super(CO_Embed2_Dist, self).__init__(add_descriptors=False)
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_Embed2_Dist(engine,
                                    x,
                                    tau=self.tau)


def HCTSA_CO_Embed2_Shapes(eng, x, tau='tau', shape='circle', r=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Takes a shape and places it on each point in the two-dimensional time-delay
    % embedding space sequentially. This function counts the points inside this shape
    % as a function of time, and returns statistics on this extracted time series.
    % 
    % INPUTS:
    % y, the input time-series as a (z-scored) column vector
    % tau, the time-delay
    % shape, has to be 'circle' for now...
    % r, the radius of the circle
    % 
    % Outputs are of the constructed time series of the number of nearby points, and
    % include the autocorrelation, maximum, median, mode, a Poisson fit to the
    % distribution, histogram entropy, and stationarity over fifths of the time
    % series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac2',
                                                      'hist10_ent',
                                                      'iqr',
                                                      'iqronrange',
                                                      'max',
                                                      'median',
                                                      'mode',
                                                      'mode_val',
                                                      'poissfit_absdiff',
                                                      'poissfit_l',
                                                      'poissfit_sqdiff',
                                                      'statav5_m',
                                                      'statav5_s',
                                                      'std',
                                                      'tau']}
    if tau is None:
        out = eng.run_function(1, 'CO_Embed2_Shapes', x, )
    elif shape is None:
        out = eng.run_function(1, 'CO_Embed2_Shapes', x, tau)
    elif r is None:
        out = eng.run_function(1, 'CO_Embed2_Shapes', x, tau, shape)
    else:
        out = eng.run_function(1, 'CO_Embed2_Shapes', x, tau, shape, r)
    return outfunc(out)


class CO_Embed2_Shapes(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Takes a shape and places it on each point in the two-dimensional time-delay
    % embedding space sequentially. This function counts the points inside this shape
    % as a function of time, and returns statistics on this extracted time series.
    % 
    % INPUTS:
    % y, the input time-series as a (z-scored) column vector
    % tau, the time-delay
    % shape, has to be 'circle' for now...
    % r, the radius of the circle
    % 
    % Outputs are of the constructed time series of the number of nearby points, and
    % include the autocorrelation, maximum, median, mode, a Poisson fit to the
    % distribution, histogram entropy, and stationarity over fifths of the time
    % series.
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac2',
                'hist10_ent',
                'iqr',
                'iqronrange',
                'max',
                'median',
                'mode',
                'mode_val',
                'poissfit_absdiff',
                'poissfit_l',
                'poissfit_sqdiff',
                'statav5_m',
                'statav5_s',
                'std',
                'tau')

    def __init__(self, tau='tau', shape='circle', r=1):
        super(CO_Embed2_Shapes, self).__init__(add_descriptors=False)
        self.tau = tau
        self.shape = shape
        self.r = r

    def eval(self, engine, x):
        return HCTSA_CO_Embed2_Shapes(engine,
                                      x,
                                      tau=self.tau,
                                      shape=self.shape,
                                      r=self.r)


def HCTSA_CO_FirstMin(eng, x, MinWhat='mi'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the time at which the first minimum in a given correlation function
    % occurs.
    % 
    %---INPUTS:
    % y, the input time series
    % MinWhat, the type of correlation to minimize: either 'ac' for autocorrelation,
    %           or 'mi' for automutual information
    % 
    % Note that selecting 'ac' is unusual operation: standard operations are the
    % first zero-crossing of the autocorrelation (as in CO_FirstZero), or the first
    % minimum of the mutual information function ('mi').
    %
    % The 'mi' option uses Rudy Moddemeijer's RM_information.m code that may or may
    % not be great...
    % 
    %---HISTORY
    % Ben Fulcher, 2008
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if MinWhat is None:
        out = eng.run_function(1, 'CO_FirstMin', x, )
    else:
        out = eng.run_function(1, 'CO_FirstMin', x, MinWhat)
    return outfunc(out)


class CO_FirstMin(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the time at which the first minimum in a given correlation function
    % occurs.
    % 
    %---INPUTS:
    % y, the input time series
    % MinWhat, the type of correlation to minimize: either 'ac' for autocorrelation,
    %           or 'mi' for automutual information
    % 
    % Note that selecting 'ac' is unusual operation: standard operations are the
    % first zero-crossing of the autocorrelation (as in CO_FirstZero), or the first
    % minimum of the mutual information function ('mi').
    %
    % The 'mi' option uses Rudy Moddemeijer's RM_information.m code that may or may
    % not be great...
    % 
    %---HISTORY
    % Ben Fulcher, 2008
    % 
    ----------------------------------------
    """

    def __init__(self, MinWhat='mi'):
        super(CO_FirstMin, self).__init__(add_descriptors=False)
        self.MinWhat = MinWhat

    def eval(self, engine, x):
        return HCTSA_CO_FirstMin(engine,
                                 x,
                                 MinWhat=self.MinWhat)


def HCTSA_CO_FirstZero(eng, x, corrfn='ac', maxtau=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the first zero-crossing of a given autocorrelation function.
    % 
    % y, the input time series
    % corrfn, the self-correlation function to measure:
    %         (i) 'ac': normal linear autocorrelation function. Uses CO_AutoCorr to
    %                   calculate autocorrelations.
    % maxtau, a maximum time-delay to search up to
    % 
    % In future, could add an option to return the point at which the function
    % crosses the axis, rather than the first integer lag at which it has already
    % crossed (what is currently implemented)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if corrfn is None:
        out = eng.run_function(1, 'CO_FirstZero', x, )
    elif maxtau is None:
        out = eng.run_function(1, 'CO_FirstZero', x, corrfn)
    else:
        out = eng.run_function(1, 'CO_FirstZero', x, corrfn, maxtau)
    return outfunc(out)


class CO_FirstZero(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the first zero-crossing of a given autocorrelation function.
    % 
    % y, the input time series
    % corrfn, the self-correlation function to measure:
    %         (i) 'ac': normal linear autocorrelation function. Uses CO_AutoCorr to
    %                   calculate autocorrelations.
    % maxtau, a maximum time-delay to search up to
    % 
    % In future, could add an option to return the point at which the function
    % crosses the axis, rather than the first integer lag at which it has already
    % crossed (what is currently implemented)
    % 
    ----------------------------------------
    """

    def __init__(self, corrfn='ac', maxtau=None):
        super(CO_FirstZero, self).__init__(add_descriptors=False)
        self.corrfn = corrfn
        self.maxtau = maxtau

    def eval(self, engine, x):
        return HCTSA_CO_FirstZero(engine,
                                  x,
                                  corrfn=self.corrfn,
                                  maxtau=self.maxtau)


def HCTSA_CO_HistogramAMI(eng, x, tau=2, meth='std2', nbins=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the automutual information using histograms, using a given approach
    % to binning the data.
    % 
    % Uses hist2.m function (renamed NK_hist2.m here) by Nedialko Krouchev, obtained
    % from Matlab Central,
    % http://www.mathworks.com/matlabcentral/fileexchange/12346-hist2-for-the-people
    % [[hist2 for the people by Nedialko Krouchev, 20 Sep 2006 (Updated 21 Sep 2006)]]
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % tau, the time-lag (1 by default)
    % 
    % meth, the method of computing automutual information:
    %           (i) 'even': evenly-spaced bins through the range of the time series,
    %           (ii) 'std1', 'std2': bins that extend only up to a multiple of the
    %                                standard deviation from the mean of the time
    %                                series to exclude outliers,
    %           (iii) 'quantiles': equiprobable bins chosen using quantiles.
    % 
    % nbins, the number of bins, required by some methods, meth (see above)
    % 
    %---OUTPUT: the automutual information calculated in this way.
    %
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if tau is None:
        out = eng.run_function(1, 'CO_HistogramAMI', x, )
    elif meth is None:
        out = eng.run_function(1, 'CO_HistogramAMI', x, tau)
    elif nbins is None:
        out = eng.run_function(1, 'CO_HistogramAMI', x, tau, meth)
    else:
        out = eng.run_function(1, 'CO_HistogramAMI', x, tau, meth, nbins)
    return outfunc(out)


class CO_HistogramAMI(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the automutual information using histograms, using a given approach
    % to binning the data.
    % 
    % Uses hist2.m function (renamed NK_hist2.m here) by Nedialko Krouchev, obtained
    % from Matlab Central,
    % http://www.mathworks.com/matlabcentral/fileexchange/12346-hist2-for-the-people
    % [[hist2 for the people by Nedialko Krouchev, 20 Sep 2006 (Updated 21 Sep 2006)]]
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % tau, the time-lag (1 by default)
    % 
    % meth, the method of computing automutual information:
    %           (i) 'even': evenly-spaced bins through the range of the time series,
    %           (ii) 'std1', 'std2': bins that extend only up to a multiple of the
    %                                standard deviation from the mean of the time
    %                                series to exclude outliers,
    %           (iii) 'quantiles': equiprobable bins chosen using quantiles.
    % 
    % nbins, the number of bins, required by some methods, meth (see above)
    % 
    %---OUTPUT: the automutual information calculated in this way.
    %
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """

    def __init__(self, tau=2, meth='std2', nbins=10):
        super(CO_HistogramAMI, self).__init__(add_descriptors=False)
        self.tau = tau
        self.meth = meth
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_CO_HistogramAMI(engine,
                                     x,
                                     tau=self.tau,
                                     meth=self.meth,
                                     nbins=self.nbins)


def HCTSA_CO_NonlinearAutocorr(eng, x, taus=(0, 4, 5), doabs=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes autocorrelations of the input time series of the form
    % <x_i x_{i-\tau_1} x{i-\tau_2}...>
    % The usual two-point autocorrelations are
    % <x_i.x_{i-\tau}>
    % 
    % Assumes that all the taus are much less than the length of the time
    % series, N, so that the means can be approximated as the sample means and the
    % standard deviations approximated as the sample standard deviations and so
    % the z-scored time series can simply be used straight-up.
    % 
    % % INPUTS:
    % y  -- should be the z-scored time series (Nx1 vector)
    % taus -- should be a vector of the time delays as above (mx1 vector)
    %   e.g., [2] computes <x_i x_{i-2}>
    %   e.g., [1,2] computes <x_i x_{i-1} x{i-2}>
    %   e.g., [1,1,3] computes <x_i x_{i-1}^2 x{i-3}>
    % doabs [opt] -- a boolean (0,1) -- if one, takes an absolute value before
    %                taking the final mean -- useful for an odd number of
    %                contributions to the sum. Default is to do this for odd
    %                numbers anyway, if not specified.
    %
    % Note: for odd numbers of regressions (i.e., even number length
    %         taus vectors) the result will be near zero due to fluctuations
    %         below the mean; even for highly-correlated signals. (doabs)
    % Note: doabs = 1 is really a different operation that can't be compared with
    %         the values obtained from taking doabs = 0 (i.e., for odd lengths
    %         of taus)
    % Note: It can be helpful to look at nlac at each iteration.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if taus is None:
        out = eng.run_function(1, 'CO_NonlinearAutocorr', x, )
    elif doabs is None:
        out = eng.run_function(1, 'CO_NonlinearAutocorr', x, taus)
    else:
        out = eng.run_function(1, 'CO_NonlinearAutocorr', x, taus, doabs)
    return outfunc(out)


class CO_NonlinearAutocorr(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes autocorrelations of the input time series of the form
    % <x_i x_{i-	au_1} x{i-	au_2}...>
    % The usual two-point autocorrelations are
    % <x_i.x_{i-	au}>
    % 
    % Assumes that all the taus are much less than the length of the time
    % series, N, so that the means can be approximated as the sample means and the
    % standard deviations approximated as the sample standard deviations and so
    % the z-scored time series can simply be used straight-up.
    % 
    % % INPUTS:
    % y  -- should be the z-scored time series (Nx1 vector)
    % taus -- should be a vector of the time delays as above (mx1 vector)
    %   e.g., [2] computes <x_i x_{i-2}>
    %   e.g., [1,2] computes <x_i x_{i-1} x{i-2}>
    %   e.g., [1,1,3] computes <x_i x_{i-1}^2 x{i-3}>
    % doabs [opt] -- a boolean (0,1) -- if one, takes an absolute value before
    %                taking the final mean -- useful for an odd number of
    %                contributions to the sum. Default is to do this for odd
    %                numbers anyway, if not specified.
    %
    % Note: for odd numbers of regressions (i.e., even number length
    %         taus vectors) the result will be near zero due to fluctuations
    %         below the mean; even for highly-correlated signals. (doabs)
    % Note: doabs = 1 is really a different operation that can't be compared with
    %         the values obtained from taking doabs = 0 (i.e., for odd lengths
    %         of taus)
    % Note: It can be helpful to look at nlac at each iteration.
    % 
    ----------------------------------------
    """

    def __init__(self, taus=(0, 4, 5), doabs=None):
        super(CO_NonlinearAutocorr, self).__init__(add_descriptors=False)
        self.taus = taus
        self.doabs = doabs

    def eval(self, engine, x):
        return HCTSA_CO_NonlinearAutocorr(engine,
                                          x,
                                          taus=self.taus,
                                          doabs=self.doabs)


def HCTSA_CO_RM_AMInformation(eng, x, tau=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Wrapper for Rudy Moddemeijer's information code to calculate automutual
    % information.
    % 
    % INPUTS:
    % y, the input time series
    % tau, the time lag at which to calculate the automutual information
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if tau is None:
        out = eng.run_function(1, 'CO_RM_AMInformation', x, )
    else:
        out = eng.run_function(1, 'CO_RM_AMInformation', x, tau)
    return outfunc(out)


class CO_RM_AMInformation(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Wrapper for Rudy Moddemeijer's information code to calculate automutual
    % information.
    % 
    % INPUTS:
    % y, the input time series
    % tau, the time lag at which to calculate the automutual information
    %
    ----------------------------------------
    """

    def __init__(self, tau=10):
        super(CO_RM_AMInformation, self).__init__(add_descriptors=False)
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_RM_AMInformation(engine,
                                         x,
                                         tau=self.tau)


def HCTSA_CO_StickAngles(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes line-of-sight angles between time-series points where each
    % time-series value is treated as a stick protruding from an opaque baseline
    % level. Statistics are returned on the raw time series, where sticks protrude
    % from the zero-level, and the z-scored time series, where sticks
    % protrude from the mean level of the time series.
    % 
    % INPUTS:
    % y, the input time series
    % 
    % Outputs are returned on the obtained sequence of angles, theta, reflecting the
    % maximum deviation a stick can rotate before hitting a stick representing
    % another time point. Statistics include the mean and spread of theta,
    % the different between positive and negative angles, measures of symmetry of
    % the angles, stationarity, autocorrelation, and measures of the distribution of
    % these stick angles.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1_all',
                                                      'ac1_n',
                                                      'ac1_p',
                                                      'ac2_all',
                                                      'ac2_n',
                                                      'ac2_p',
                                                      'kurtosis_all',
                                                      'kurtosis_n',
                                                      'kurtosis_p',
                                                      'mean',
                                                      'mean_n',
                                                      'mean_p',
                                                      'median',
                                                      'median_n',
                                                      'median_p',
                                                      'pnsumabsdiff',
                                                      'q10_all',
                                                      'q10_n',
                                                      'q10_p',
                                                      'q1_all',
                                                      'q1_n',
                                                      'q1_p',
                                                      'q90_all',
                                                      'q90_n',
                                                      'q90_p',
                                                      'q99_all',
                                                      'q99_n',
                                                      'q99_p',
                                                      'ratmean_n',
                                                      'ratmean_p',
                                                      'skewness_all',
                                                      'skewness_n',
                                                      'skewness_p',
                                                      'statav2_all_m',
                                                      'statav2_all_s',
                                                      'statav2_n_m',
                                                      'statav2_n_s',
                                                      'statav2_p_m',
                                                      'statav2_p_s',
                                                      'statav3_all_m',
                                                      'statav3_all_s',
                                                      'statav3_n_m',
                                                      'statav3_n_s',
                                                      'statav3_p_m',
                                                      'statav3_p_s',
                                                      'statav4_all_m',
                                                      'statav4_all_s',
                                                      'statav4_n_m',
                                                      'statav4_n_s',
                                                      'statav4_p_m',
                                                      'statav4_p_s',
                                                      'statav5_all_m',
                                                      'statav5_all_s',
                                                      'statav5_n_m',
                                                      'statav5_n_s',
                                                      'statav5_p_m',
                                                      'statav5_p_s',
                                                      'std',
                                                      'std_n',
                                                      'std_p',
                                                      'symks_n',
                                                      'symks_p',
                                                      'tau_all',
                                                      'tau_n',
                                                      'tau_p']}
    out = eng.run_function(1, 'CO_StickAngles', x, )
    return outfunc(out)


class CO_StickAngles(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes line-of-sight angles between time-series points where each
    % time-series value is treated as a stick protruding from an opaque baseline
    % level. Statistics are returned on the raw time series, where sticks protrude
    % from the zero-level, and the z-scored time series, where sticks
    % protrude from the mean level of the time series.
    % 
    % INPUTS:
    % y, the input time series
    % 
    % Outputs are returned on the obtained sequence of angles, theta, reflecting the
    % maximum deviation a stick can rotate before hitting a stick representing
    % another time point. Statistics include the mean and spread of theta,
    % the different between positive and negative angles, measures of symmetry of
    % the angles, stationarity, autocorrelation, and measures of the distribution of
    % these stick angles.
    % 
    ----------------------------------------
    """

    outnames = ('ac1_all',
                'ac1_n',
                'ac1_p',
                'ac2_all',
                'ac2_n',
                'ac2_p',
                'kurtosis_all',
                'kurtosis_n',
                'kurtosis_p',
                'mean',
                'mean_n',
                'mean_p',
                'median',
                'median_n',
                'median_p',
                'pnsumabsdiff',
                'q10_all',
                'q10_n',
                'q10_p',
                'q1_all',
                'q1_n',
                'q1_p',
                'q90_all',
                'q90_n',
                'q90_p',
                'q99_all',
                'q99_n',
                'q99_p',
                'ratmean_n',
                'ratmean_p',
                'skewness_all',
                'skewness_n',
                'skewness_p',
                'statav2_all_m',
                'statav2_all_s',
                'statav2_n_m',
                'statav2_n_s',
                'statav2_p_m',
                'statav2_p_s',
                'statav3_all_m',
                'statav3_all_s',
                'statav3_n_m',
                'statav3_n_s',
                'statav3_p_m',
                'statav3_p_s',
                'statav4_all_m',
                'statav4_all_s',
                'statav4_n_m',
                'statav4_n_s',
                'statav4_p_m',
                'statav4_p_s',
                'statav5_all_m',
                'statav5_all_s',
                'statav5_n_m',
                'statav5_n_s',
                'statav5_p_m',
                'statav5_p_s',
                'std',
                'std_n',
                'std_p',
                'symks_n',
                'symks_p',
                'tau_all',
                'tau_n',
                'tau_p')

    def __init__(self, ):
        super(CO_StickAngles, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_CO_StickAngles(engine, x)


def HCTSA_CO_TSTL_AutoCorrMethod(eng, x, maxlag=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the autocorrelation function using a fast Fourier Transform method
    % implemented in TSTOOL and returns the mean square discrepancy between the
    % autocorrelation coefficients obtained in this way from those obtained in the
    % time domain using CO_AutoCorr.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % No real rationale behind this, other than the difference in autocorrelations
    % computed by the two methods may somehow be informative of something about the
    % time series...
    % 
    %---INPUTS:
    % y, the input time series
    % maxlag, the maximum time lag to compute up to -- will compare autocorrelations
    %         up to this value
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if maxlag is None:
        out = eng.run_function(1, 'CO_TSTL_AutoCorrMethod', x, )
    else:
        out = eng.run_function(1, 'CO_TSTL_AutoCorrMethod', x, maxlag)
    return outfunc(out)


class CO_TSTL_AutoCorrMethod(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the autocorrelation function using a fast Fourier Transform method
    % implemented in TSTOOL and returns the mean square discrepancy between the
    % autocorrelation coefficients obtained in this way from those obtained in the
    % time domain using CO_AutoCorr.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % No real rationale behind this, other than the difference in autocorrelations
    % computed by the two methods may somehow be informative of something about the
    % time series...
    % 
    %---INPUTS:
    % y, the input time series
    % maxlag, the maximum time lag to compute up to -- will compare autocorrelations
    %         up to this value
    % 
    ----------------------------------------
    """

    def __init__(self, maxlag=None):
        super(CO_TSTL_AutoCorrMethod, self).__init__(add_descriptors=False)
        self.maxlag = maxlag

    def eval(self, engine, x):
        return HCTSA_CO_TSTL_AutoCorrMethod(engine,
                                            x,
                                            maxlag=self.maxlag)


def HCTSA_CO_TSTL_amutual(eng, x, maxtau=20, nbins=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses amutual code from TSTOOL, which uses a
    % histogram method with n bins to estimate the mutual information of a
    % time series across a range of time-delays, tau.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    % 
    % y, the time series
    % 
    % maxtau, the maximum lag for which to calculate the auto mutual information
    % 
    % nbins, the number of bins for histogram calculation
    % 
    %---OUTPUTS: A number of statistics of the function over the range of tau,
    % including the mean mutual information, its standard deviation, first minimum,
    % proportion of extrema, and measures of periodicity in the positions of local
    % maxima.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ami1',
                                                      'ami10',
                                                      'ami11',
                                                      'ami12',
                                                      'ami13',
                                                      'ami14',
                                                      'ami15',
                                                      'ami16',
                                                      'ami17',
                                                      'ami18',
                                                      'ami19',
                                                      'ami2',
                                                      'ami20',
                                                      'ami21',
                                                      'ami3',
                                                      'ami4',
                                                      'ami5',
                                                      'ami6',
                                                      'ami7',
                                                      'ami8',
                                                      'ami9',
                                                      'fmmi',
                                                      'mami',
                                                      'modeperiodmax',
                                                      'pextrema',
                                                      'pmaxima',
                                                      'pmodeperiodmax',
                                                      'stdami']}
    if maxtau is None:
        out = eng.run_function(1, 'CO_TSTL_amutual', x, )
    elif nbins is None:
        out = eng.run_function(1, 'CO_TSTL_amutual', x, maxtau)
    else:
        out = eng.run_function(1, 'CO_TSTL_amutual', x, maxtau, nbins)
    return outfunc(out)


class CO_TSTL_amutual(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses amutual code from TSTOOL, which uses a
    % histogram method with n bins to estimate the mutual information of a
    % time series across a range of time-delays, tau.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    % 
    % y, the time series
    % 
    % maxtau, the maximum lag for which to calculate the auto mutual information
    % 
    % nbins, the number of bins for histogram calculation
    % 
    %---OUTPUTS: A number of statistics of the function over the range of tau,
    % including the mean mutual information, its standard deviation, first minimum,
    % proportion of extrema, and measures of periodicity in the positions of local
    % maxima.
    % 
    ----------------------------------------
    """

    outnames = ('ami1',
                'ami10',
                'ami11',
                'ami12',
                'ami13',
                'ami14',
                'ami15',
                'ami16',
                'ami17',
                'ami18',
                'ami19',
                'ami2',
                'ami20',
                'ami21',
                'ami3',
                'ami4',
                'ami5',
                'ami6',
                'ami7',
                'ami8',
                'ami9',
                'fmmi',
                'mami',
                'modeperiodmax',
                'pextrema',
                'pmaxima',
                'pmodeperiodmax',
                'stdami')

    def __init__(self, maxtau=20, nbins=10):
        super(CO_TSTL_amutual, self).__init__(add_descriptors=False)
        self.maxtau = maxtau
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_CO_TSTL_amutual(engine,
                                     x,
                                     maxtau=self.maxtau,
                                     nbins=self.nbins)


def HCTSA_CO_TSTL_amutual2(eng, x, maxtau=50):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses amutual2 code from TSTOOL to compute the mutual information up to a given
    % maximum time-delay.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time series data
    % maxtau, maximal lag
    % 
    %---OUTPUTS: Statistics on the output of amutual2 over this range, as for
    % CO_TSTL_amutual.
    % 
    %---HISTORY:
    % Ben Fulcher, 2009
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ami1',
                                                      'ami10',
                                                      'ami11',
                                                      'ami12',
                                                      'ami13',
                                                      'ami14',
                                                      'ami15',
                                                      'ami16',
                                                      'ami17',
                                                      'ami18',
                                                      'ami19',
                                                      'ami2',
                                                      'ami20',
                                                      'ami21',
                                                      'ami22',
                                                      'ami23',
                                                      'ami24',
                                                      'ami25',
                                                      'ami26',
                                                      'ami27',
                                                      'ami28',
                                                      'ami29',
                                                      'ami3',
                                                      'ami30',
                                                      'ami31',
                                                      'ami32',
                                                      'ami33',
                                                      'ami34',
                                                      'ami35',
                                                      'ami36',
                                                      'ami37',
                                                      'ami38',
                                                      'ami39',
                                                      'ami4',
                                                      'ami40',
                                                      'ami41',
                                                      'ami42',
                                                      'ami43',
                                                      'ami44',
                                                      'ami45',
                                                      'ami46',
                                                      'ami47',
                                                      'ami48',
                                                      'ami49',
                                                      'ami5',
                                                      'ami50',
                                                      'ami6',
                                                      'ami7',
                                                      'ami8',
                                                      'ami9',
                                                      'amiac1',
                                                      'fmmi',
                                                      'mami',
                                                      'modeperiodmax',
                                                      'modeperiodmin',
                                                      'pcrossmean',
                                                      'pcrossmedian',
                                                      'pcrossq10',
                                                      'pcrossq90',
                                                      'pextrema',
                                                      'pmaxima',
                                                      'pminima',
                                                      'pmodeperiodmax',
                                                      'pmodeperiodmin',
                                                      'stdami']}
    if maxtau is None:
        out = eng.run_function(1, 'CO_TSTL_amutual2', x, )
    else:
        out = eng.run_function(1, 'CO_TSTL_amutual2', x, maxtau)
    return outfunc(out)


class CO_TSTL_amutual2(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses amutual2 code from TSTOOL to compute the mutual information up to a given
    % maximum time-delay.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time series data
    % maxtau, maximal lag
    % 
    %---OUTPUTS: Statistics on the output of amutual2 over this range, as for
    % CO_TSTL_amutual.
    % 
    %---HISTORY:
    % Ben Fulcher, 2009
    ----------------------------------------
    """

    outnames = ('ami1',
                'ami10',
                'ami11',
                'ami12',
                'ami13',
                'ami14',
                'ami15',
                'ami16',
                'ami17',
                'ami18',
                'ami19',
                'ami2',
                'ami20',
                'ami21',
                'ami22',
                'ami23',
                'ami24',
                'ami25',
                'ami26',
                'ami27',
                'ami28',
                'ami29',
                'ami3',
                'ami30',
                'ami31',
                'ami32',
                'ami33',
                'ami34',
                'ami35',
                'ami36',
                'ami37',
                'ami38',
                'ami39',
                'ami4',
                'ami40',
                'ami41',
                'ami42',
                'ami43',
                'ami44',
                'ami45',
                'ami46',
                'ami47',
                'ami48',
                'ami49',
                'ami5',
                'ami50',
                'ami6',
                'ami7',
                'ami8',
                'ami9',
                'amiac1',
                'fmmi',
                'mami',
                'modeperiodmax',
                'modeperiodmin',
                'pcrossmean',
                'pcrossmedian',
                'pcrossq10',
                'pcrossq90',
                'pextrema',
                'pmaxima',
                'pminima',
                'pmodeperiodmax',
                'pmodeperiodmin',
                'stdami')

    def __init__(self, maxtau=50):
        super(CO_TSTL_amutual2, self).__init__(add_descriptors=False)
        self.maxtau = maxtau

    def eval(self, engine, x):
        return HCTSA_CO_TSTL_amutual2(engine,
                                      x,
                                      maxtau=self.maxtau)


def HCTSA_CO_TranslateShape(eng, x, shape='circle', d=3.5, howtomove='pts'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on the number of data points that reside inside a given
    % geometric shape that is moved around the time series. Inputs specify a shape
    % and its size, and a method for moving this shape through the time domain.
    % 
    % This is usually more informative in an embedding space (CO_Embed2_...), but
    % here we do it just in the temporal domain (_t_).
    % 
    % In the future, could perform a similar analysis with a soft boundary, some
    % decaying force function V(r), or perhaps truncated...?
    %
    % INPUTS:
    % 
    % y, the input time series
    % 
    % shape, the shape to move about the time-domain ('circle')
    % 
    % d, a parameter specifying the size of the shape (e.g., d = 2)
    % 
    % howtomove, a method specifying how to move the shape about, e.g., 'pts'
    %               places the shape on each point in the time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['eights',
                                                      'elevens',
                                                      'fives',
                                                      'fours',
                                                      'max',
                                                      'mode',
                                                      'nines',
                                                      'npatmode',
                                                      'ones',
                                                      'sevens',
                                                      'sixes',
                                                      'statav2_m',
                                                      'statav2_s',
                                                      'statav3_m',
                                                      'statav3_s',
                                                      'statav4_m',
                                                      'statav4_s',
                                                      'std',
                                                      'tens',
                                                      'threes',
                                                      'twos']}
    if shape is None:
        out = eng.run_function(1, 'CO_TranslateShape', x, )
    elif d is None:
        out = eng.run_function(1, 'CO_TranslateShape', x, shape)
    elif howtomove is None:
        out = eng.run_function(1, 'CO_TranslateShape', x, shape, d)
    else:
        out = eng.run_function(1, 'CO_TranslateShape', x, shape, d, howtomove)
    return outfunc(out)


class CO_TranslateShape(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on the number of data points that reside inside a given
    % geometric shape that is moved around the time series. Inputs specify a shape
    % and its size, and a method for moving this shape through the time domain.
    % 
    % This is usually more informative in an embedding space (CO_Embed2_...), but
    % here we do it just in the temporal domain (_t_).
    % 
    % In the future, could perform a similar analysis with a soft boundary, some
    % decaying force function V(r), or perhaps truncated...?
    %
    % INPUTS:
    % 
    % y, the input time series
    % 
    % shape, the shape to move about the time-domain ('circle')
    % 
    % d, a parameter specifying the size of the shape (e.g., d = 2)
    % 
    % howtomove, a method specifying how to move the shape about, e.g., 'pts'
    %               places the shape on each point in the time series
    % 
    ----------------------------------------
    """

    outnames = ('eights',
                'elevens',
                'fives',
                'fours',
                'max',
                'mode',
                'nines',
                'npatmode',
                'ones',
                'sevens',
                'sixes',
                'statav2_m',
                'statav2_s',
                'statav3_m',
                'statav3_s',
                'statav4_m',
                'statav4_s',
                'std',
                'tens',
                'threes',
                'twos')

    def __init__(self, shape='circle', d=3.5, howtomove='pts'):
        super(CO_TranslateShape, self).__init__(add_descriptors=False)
        self.shape = shape
        self.d = d
        self.howtomove = howtomove

    def eval(self, engine, x):
        return HCTSA_CO_TranslateShape(engine,
                                       x,
                                       shape=self.shape,
                                       d=self.d,
                                       howtomove=self.howtomove)


def HCTSA_CO_f1ecac(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds where autocorrelation function first crosses 1/e, the 1/e correlation
    % length
    % 
    % INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'CO_f1ecac', x, )
    return outfunc(out)


class CO_f1ecac(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds where autocorrelation function first crosses 1/e, the 1/e correlation
    % length
    % 
    % INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(CO_f1ecac, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_CO_f1ecac(engine, x)


def HCTSA_CO_fzcglscf(eng, x, alpha=5, beta=10, maxtau=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the first zero-crossing of the generalized self-correlation function
    % introduced in Duarte Queiros and Moyano in Physica A, Vol. 383, pp. 10--15
    % (2007) in the paper "Yet on statistical properties of traded volume:
    % Correlation and mutual information at different value magnitudes"
    % Uses CO_glscf to calculate the generalized self-correlations.
    % Keeps calculating until the function finds a minimum, and returns this lag.
    % 
    % INPUTS:
    % y, the input time series
    % alpha, the parameter alpha
    % beta, the parameter beta
    % maxtau [opt], a maximum time delay to search up to (default is the time-series
    %                length)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if alpha is None:
        out = eng.run_function(1, 'CO_fzcglscf', x, )
    elif beta is None:
        out = eng.run_function(1, 'CO_fzcglscf', x, alpha)
    elif maxtau is None:
        out = eng.run_function(1, 'CO_fzcglscf', x, alpha, beta)
    else:
        out = eng.run_function(1, 'CO_fzcglscf', x, alpha, beta, maxtau)
    return outfunc(out)


class CO_fzcglscf(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the first zero-crossing of the generalized self-correlation function
    % introduced in Duarte Queiros and Moyano in Physica A, Vol. 383, pp. 10--15
    % (2007) in the paper "Yet on statistical properties of traded volume:
    % Correlation and mutual information at different value magnitudes"
    % Uses CO_glscf to calculate the generalized self-correlations.
    % Keeps calculating until the function finds a minimum, and returns this lag.
    % 
    % INPUTS:
    % y, the input time series
    % alpha, the parameter alpha
    % beta, the parameter beta
    % maxtau [opt], a maximum time delay to search up to (default is the time-series
    %                length)
    % 
    ----------------------------------------
    """

    def __init__(self, alpha=5, beta=10, maxtau=None):
        super(CO_fzcglscf, self).__init__(add_descriptors=False)
        self.alpha = alpha
        self.beta = beta
        self.maxtau = maxtau

    def eval(self, engine, x):
        return HCTSA_CO_fzcglscf(engine,
                                 x,
                                 alpha=self.alpha,
                                 beta=self.beta,
                                 maxtau=self.maxtau)


def HCTSA_CO_glscf(eng, x, alpha=2, beta=5, tau=2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the generalized linear self-correlation function of a time series.
    % This function was introduced in Queiros and Moyano in Physica A, Vol. 383, pp.
    % 10--15 (2007) in the paper "Yet on statistical properties of traded volume: 
    % Correlation and mutual information at different value magnitudes"
    % 
    % The function considers magnitude correlations:
    % INPUTS:
    % y, the input time series
    % Parameters alpha, beta are real and nonzero
    % tau is the time-delay (can also be 'tau' to set to first zero-crossing of the ACF)
    % 
    % When alpha = beta estimates how values of the same order of magnitude are
    % related in time
    % When alpha ~= beta, estimates correlations between different magnitudes of the
    % time series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if alpha is None:
        out = eng.run_function(1, 'CO_glscf', x, )
    elif beta is None:
        out = eng.run_function(1, 'CO_glscf', x, alpha)
    elif tau is None:
        out = eng.run_function(1, 'CO_glscf', x, alpha, beta)
    else:
        out = eng.run_function(1, 'CO_glscf', x, alpha, beta, tau)
    return outfunc(out)


class CO_glscf(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the generalized linear self-correlation function of a time series.
    % This function was introduced in Queiros and Moyano in Physica A, Vol. 383, pp.
    % 10--15 (2007) in the paper "Yet on statistical properties of traded volume: 
    % Correlation and mutual information at different value magnitudes"
    % 
    % The function considers magnitude correlations:
    % INPUTS:
    % y, the input time series
    % Parameters alpha, beta are real and nonzero
    % tau is the time-delay (can also be 'tau' to set to first zero-crossing of the ACF)
    % 
    % When alpha = beta estimates how values of the same order of magnitude are
    % related in time
    % When alpha ~= beta, estimates correlations between different magnitudes of the
    % time series.
    % 
    ----------------------------------------
    """

    def __init__(self, alpha=2, beta=5, tau=2):
        super(CO_glscf, self).__init__(add_descriptors=False)
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_glscf(engine,
                              x,
                              alpha=self.alpha,
                              beta=self.beta,
                              tau=self.tau)


def HCTSA_CO_tc3(eng, x, tau=1):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes the tc3 function, a normalized nonlinear autocorrelation, at a
    % given time-delay, tau.
    % Outputs are the raw tc3 expression, its magnitude, the numerator and its magnitude, and
    % the denominator.
    % 
    % INPUTS:
    % y, input time series
    % tau, time lag
    % 
    % See documentation of the TSTOOL package (http://www.physik3.gwdg.de/tstool/)
    % for further details about this function.
    %
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['abs',
                                                      'absnum',
                                                      'denom',
                                                      'num',
                                                      'raw']}
    if tau is None:
        out = eng.run_function(1, 'CO_tc3', x, )
    else:
        out = eng.run_function(1, 'CO_tc3', x, tau)
    return outfunc(out)


class CO_tc3(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    %
    % Computes the tc3 function, a normalized nonlinear autocorrelation, at a
    % given time-delay, tau.
    % Outputs are the raw tc3 expression, its magnitude, the numerator and its magnitude, and
    % the denominator.
    % 
    % INPUTS:
    % y, input time series
    % tau, time lag
    % 
    % See documentation of the TSTOOL package (http://www.physik3.gwdg.de/tstool/)
    % for further details about this function.
    %
    ----------------------------------------
    """

    outnames = ('abs',
                'absnum',
                'denom',
                'num',
                'raw')

    def __init__(self, tau=1):
        super(CO_tc3, self).__init__(add_descriptors=False)
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_tc3(engine,
                            x,
                            tau=self.tau)


def HCTSA_CO_trev(eng, x, tau='ac'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the trev function, a normalized nonlinear autocorrelation,
    % mentioned in the documentation of the TSTOOL nonlinear time-series analysis
    % package (available here: http://www.physik3.gwdg.de/tstool/).
    % 
    % The quantity is often used as a nonlinearity statistic in surrogate data
    % analysis, cf. "Surrogate time series", T. Schreiber and A. Schmitz, Physica D,
    % 142(3-4) 346 (2000).
    % 
    %---INPUTS:
    % 
    % y, time series
    % 
    % tau, time lag (can be 'ac' or 'mi' to set as the first zero-crossing of the
    %       autocorrelation function, or the first minimum of the automutual
    %       information function, respectively)
    % 
    %---OUTPUTS: the raw trev expression, its magnitude, the numerator and its
    % magnitude, and the denominator.
    % 
    %---HISTORY:
    % Ben Fulcher, 15/11/2009
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['abs',
                                                      'absnum',
                                                      'denom',
                                                      'num',
                                                      'raw']}
    if tau is None:
        out = eng.run_function(1, 'CO_trev', x, )
    else:
        out = eng.run_function(1, 'CO_trev', x, tau)
    return outfunc(out)


class CO_trev(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the trev function, a normalized nonlinear autocorrelation,
    % mentioned in the documentation of the TSTOOL nonlinear time-series analysis
    % package (available here: http://www.physik3.gwdg.de/tstool/).
    % 
    % The quantity is often used as a nonlinearity statistic in surrogate data
    % analysis, cf. "Surrogate time series", T. Schreiber and A. Schmitz, Physica D,
    % 142(3-4) 346 (2000).
    % 
    %---INPUTS:
    % 
    % y, time series
    % 
    % tau, time lag (can be 'ac' or 'mi' to set as the first zero-crossing of the
    %       autocorrelation function, or the first minimum of the automutual
    %       information function, respectively)
    % 
    %---OUTPUTS: the raw trev expression, its magnitude, the numerator and its
    % magnitude, and the denominator.
    % 
    %---HISTORY:
    % Ben Fulcher, 15/11/2009
    ----------------------------------------
    """

    outnames = ('abs',
                'absnum',
                'denom',
                'num',
                'raw')

    def __init__(self, tau='ac'):
        super(CO_trev, self).__init__(add_descriptors=False)
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_CO_trev(engine,
                             x,
                             tau=self.tau)


def HCTSA_CP_ML_StepDetect(eng, x, method='kv', params=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Gives information about discrete steps in the signal, using the function
    % l1pwc from Max A. Little's step detection toolkit.
    % 
    % cf.,
    % "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    % Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
    % 
    % "Steps and bumps: precision extraction of discrete states of molecular machines"
    % M. A. Little, B. C. Steel, F. Bai, Y. Sowa, T. Bilyard, D. M. Mueller,
    % R. M. Berry, N. S. Jones. Biophysical Journal, 101(2):477-485 (2011)
    % 
    % Software available at: http://www.maxlittle.net/software/index.php
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % method, the step-detection method:
    %           (i) 'kv': Kalafut-Visscher
    %                 cf. The algorithm described in:
    %                 Kalafut, Visscher, "An objective, model-independent method for
    %                 detection of non-uniform steps in noisy signals", Comp. Phys.
    %                 Comm., 179(2008), 716-723.
    %                 
    %           (ii) 'ck': Chung-Kennedy
    %                 S.H. Chung, R.A. Kennedy (1991), "Forward-backward non-linear
    %                 filtering technique for extracting small biological signals
    %                 from noise", J. Neurosci. Methods. 40(1):71-86.
    %                 
    %           (iii) 'l1pwc': L1 method
    %                 This code is based on code originally written by Kim et al.:
    %                 "l_1 Trend Filtering", S.-J. Kim et al., SIAM Review 51, 339
    %                 (2009).
    % 
    % params, the parameters for the given method used:
    %           (i) 'kv': (no parameters required)
    %           (ii) 'ck': params = [K,M,p]
    %           (iii) 'l1pwc': params = lambda
    % 
    %---OUTPUTS:
    % Statistics on the output of the step-detection method, including the intervals
    % between change points, the proportion of constant segments, the reduction in
    % variance from removing the piece-wise constants, and stationarity in the
    % occurrence of change points.
    % 
    %---HISTORY:
    % Ben Fulcher, 12/4/2010
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['E',
                                                      'diffn12',
                                                      'lambdamax',
                                                      'maxstepint',
                                                      'meanerrstepint',
                                                      'meanstepint',
                                                      'meanstepintgt3',
                                                      'medianstepint',
                                                      'minstepint',
                                                      'nsegments',
                                                      'pshort_3',
                                                      'ratn12',
                                                      'rmsoff',
                                                      'rmsoffpstep',
                                                      's']}
    if method is None:
        out = eng.run_function(1, 'CP_ML_StepDetect', x, )
    elif params is None:
        out = eng.run_function(1, 'CP_ML_StepDetect', x, method)
    else:
        out = eng.run_function(1, 'CP_ML_StepDetect', x, method, params)
    return outfunc(out)


class CP_ML_StepDetect(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Gives information about discrete steps in the signal, using the function
    % l1pwc from Max A. Little's step detection toolkit.
    % 
    % cf.,
    % "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    % Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
    % 
    % "Steps and bumps: precision extraction of discrete states of molecular machines"
    % M. A. Little, B. C. Steel, F. Bai, Y. Sowa, T. Bilyard, D. M. Mueller,
    % R. M. Berry, N. S. Jones. Biophysical Journal, 101(2):477-485 (2011)
    % 
    % Software available at: http://www.maxlittle.net/software/index.php
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % method, the step-detection method:
    %           (i) 'kv': Kalafut-Visscher
    %                 cf. The algorithm described in:
    %                 Kalafut, Visscher, "An objective, model-independent method for
    %                 detection of non-uniform steps in noisy signals", Comp. Phys.
    %                 Comm., 179(2008), 716-723.
    %                 
    %           (ii) 'ck': Chung-Kennedy
    %                 S.H. Chung, R.A. Kennedy (1991), "Forward-backward non-linear
    %                 filtering technique for extracting small biological signals
    %                 from noise", J. Neurosci. Methods. 40(1):71-86.
    %                 
    %           (iii) 'l1pwc': L1 method
    %                 This code is based on code originally written by Kim et al.:
    %                 "l_1 Trend Filtering", S.-J. Kim et al., SIAM Review 51, 339
    %                 (2009).
    % 
    % params, the parameters for the given method used:
    %           (i) 'kv': (no parameters required)
    %           (ii) 'ck': params = [K,M,p]
    %           (iii) 'l1pwc': params = lambda
    % 
    %---OUTPUTS:
    % Statistics on the output of the step-detection method, including the intervals
    % between change points, the proportion of constant segments, the reduction in
    % variance from removing the piece-wise constants, and stationarity in the
    % occurrence of change points.
    % 
    %---HISTORY:
    % Ben Fulcher, 12/4/2010
    % 
    ----------------------------------------
    """

    outnames = ('E',
                'diffn12',
                'lambdamax',
                'maxstepint',
                'meanerrstepint',
                'meanstepint',
                'meanstepintgt3',
                'medianstepint',
                'minstepint',
                'nsegments',
                'pshort_3',
                'ratn12',
                'rmsoff',
                'rmsoffpstep',
                's')

    def __init__(self, method='kv', params=None):
        super(CP_ML_StepDetect, self).__init__(add_descriptors=False)
        self.method = method
        self.params = params

    def eval(self, engine, x):
        return HCTSA_CP_ML_StepDetect(engine,
                                      x,
                                      method=self.method,
                                      params=self.params)


def HCTSA_CP_l1pwc_sweep_lambda(eng, x, lambdar=MatlabSequence('0:0.05:0.95')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Gives information about discrete steps in the signal across a range of
    % regularization parameters lambda, using the function l1pwc from Max Little's
    % step detection toolkit.
    % 
    % cf.,
    % "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    % Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % lambdar, a vector specifying the lambda parameters to use
    % 
    %---OUTPUTS:
    % At each iteration, the CP_ML_StepDetect code was run with a given
    % lambda, and the number of segments, and reduction in root mean square error
    % from removing the piecewise constants was recorded. Outputs summarize how the
    % these quantities vary with lambda.
    % 
    %---HISTORY:
    % Ben Fulcher, 2010-04-13
    % Ben Fulcher, 2014-02-24. Fixed a problem with no matches giving an error
    % instead of a NaN.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['bestlambda',
                                                      'bestrmserrpseg',
                                                      'corrsegerr',
                                                      'nsegsu001',
                                                      'nsegsu005',
                                                      'rmserrsu01',
                                                      'rmserrsu02',
                                                      'rmserrsu05']}
    if lambdar is None:
        out = eng.run_function(1, 'CP_l1pwc_sweep_lambda', x, )
    else:
        out = eng.run_function(1, 'CP_l1pwc_sweep_lambda', x, lambdar)
    return outfunc(out)


class CP_l1pwc_sweep_lambda(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Gives information about discrete steps in the signal across a range of
    % regularization parameters lambda, using the function l1pwc from Max Little's
    % step detection toolkit.
    % 
    % cf.,
    % "Sparse Bayesian Step-Filtering for High-Throughput Analysis of Molecular
    % Machine Dynamics", Max A. Little, and Nick S. Jones, Proc. ICASSP (2010)
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % lambdar, a vector specifying the lambda parameters to use
    % 
    %---OUTPUTS:
    % At each iteration, the CP_ML_StepDetect code was run with a given
    % lambda, and the number of segments, and reduction in root mean square error
    % from removing the piecewise constants was recorded. Outputs summarize how the
    % these quantities vary with lambda.
    % 
    %---HISTORY:
    % Ben Fulcher, 2010-04-13
    % Ben Fulcher, 2014-02-24. Fixed a problem with no matches giving an error
    % instead of a NaN.
    % 
    ----------------------------------------
    """

    outnames = ('bestlambda',
                'bestrmserrpseg',
                'corrsegerr',
                'nsegsu001',
                'nsegsu005',
                'rmserrsu01',
                'rmserrsu02',
                'rmserrsu05')

    def __init__(self, lambdar=MatlabSequence('0:0.05:0.95')):
        super(CP_l1pwc_sweep_lambda, self).__init__(add_descriptors=False)
        self.lambdar = lambdar

    def eval(self, engine, x):
        return HCTSA_CP_l1pwc_sweep_lambda(engine,
                                           x,
                                           lambdar=self.lambdar)


def HCTSA_CP_wavelet_varchg(eng, x, wname='db3', level=3, maxnchpts=10, mindelay=0.01):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds variance change points using functions from Matlab's Wavelet Toolbox,
    % including the primary function wvarchg, which estimates the change points in
    % the time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    % 
    % level, the level of wavelet decomposition
    % 
    % maxnchpts, the maximum number of change points
    % 
    % mindelay, the minimum delay between consecutive change points (can be
    %           specified as a proportion of the time-series length, e.g., 0.02
    %           ensures that change points are separated by at least 2% of the
    %           time-series length)
    % 
    % 
    %---OUTPUT: the optimal number of change points.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if wname is None:
        out = eng.run_function(1, 'CP_wavelet_varchg', x, )
    elif level is None:
        out = eng.run_function(1, 'CP_wavelet_varchg', x, wname)
    elif maxnchpts is None:
        out = eng.run_function(1, 'CP_wavelet_varchg', x, wname, level)
    elif mindelay is None:
        out = eng.run_function(1, 'CP_wavelet_varchg', x, wname, level, maxnchpts)
    else:
        out = eng.run_function(1, 'CP_wavelet_varchg', x, wname, level, maxnchpts, mindelay)
    return outfunc(out)


class CP_wavelet_varchg(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds variance change points using functions from Matlab's Wavelet Toolbox,
    % including the primary function wvarchg, which estimates the change points in
    % the time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    % 
    % level, the level of wavelet decomposition
    % 
    % maxnchpts, the maximum number of change points
    % 
    % mindelay, the minimum delay between consecutive change points (can be
    %           specified as a proportion of the time-series length, e.g., 0.02
    %           ensures that change points are separated by at least 2% of the
    %           time-series length)
    % 
    % 
    %---OUTPUT: the optimal number of change points.
    % 
    ----------------------------------------
    """

    def __init__(self, wname='db3', level=3, maxnchpts=10, mindelay=0.01):
        super(CP_wavelet_varchg, self).__init__(add_descriptors=False)
        self.wname = wname
        self.level = level
        self.maxnchpts = maxnchpts
        self.mindelay = mindelay

    def eval(self, engine, x):
        return HCTSA_CP_wavelet_varchg(engine,
                                       x,
                                       wname=self.wname,
                                       level=self.level,
                                       maxnchpts=self.maxnchpts,
                                       mindelay=self.mindelay)


def HCTSA_DN_Burstiness(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the 'burstiness' statistic from:
    % 
    % Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
    % 81, 48002 (2008)
    % 
    %---INPUT:
    % y, the input time series
    % 
    %---OUTPUT:
    % The burstiness statistic, B.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'DN_Burstiness', x, )
    return outfunc(out)


class DN_Burstiness(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the 'burstiness' statistic from:
    % 
    % Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
    % 81, 48002 (2008)
    % 
    %---INPUT:
    % y, the input time series
    % 
    %---OUTPUT:
    % The burstiness statistic, B.
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(DN_Burstiness, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_DN_Burstiness(engine, x)


def HCTSA_DN_CompareKSFit(eng, x, whatdbn='norm'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns simple statistics on the discrepancy between the
    % kernel-smoothed distribution of the time-series values, and the distribution
    % fitted to it by some model: Gaussian (using normfifit from Matlab's
    % Statistics Toolbox), Extreme Value (evfifit), Uniform (unififit), Beta
    % (betafifit), Rayleigh (raylfifit), Exponential (expfifit), Gamma (gamfit),
    % LogNormal (lognfifit), and Weibull (wblfifit).
    % 
    %---INPUTS:
    % x, the input time series
    % whatdbn, the type of distribution to fit to the data:
    %           'norm' (normal), 'ev' (extreme value), 'uni' (uniform),
    %           'beta' (Beta), 'rayleigh' (Rayleigh), 'exp' (exponential),
    %           'gamma' (Gamma), 'logn' (Log-Normal), 'wbl' (Weibull).
    % 
    %---OUTPUTS: include the absolute area between the two distributions, the peak
    % separation, overlap integral, and relative entropy.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['adiff',
                                                      'olapint',
                                                      'peaksepx',
                                                      'peaksepy',
                                                      'relent']}
    if whatdbn is None:
        out = eng.run_function(1, 'DN_CompareKSFit', x, )
    else:
        out = eng.run_function(1, 'DN_CompareKSFit', x, whatdbn)
    return outfunc(out)


class DN_CompareKSFit(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns simple statistics on the discrepancy between the
    % kernel-smoothed distribution of the time-series values, and the distribution
    % fitted to it by some model: Gaussian (using normfifit from Matlab's
    % Statistics Toolbox), Extreme Value (evfifit), Uniform (unififit), Beta
    % (betafifit), Rayleigh (raylfifit), Exponential (expfifit), Gamma (gamfit),
    % LogNormal (lognfifit), and Weibull (wblfifit).
    % 
    %---INPUTS:
    % x, the input time series
    % whatdbn, the type of distribution to fit to the data:
    %           'norm' (normal), 'ev' (extreme value), 'uni' (uniform),
    %           'beta' (Beta), 'rayleigh' (Rayleigh), 'exp' (exponential),
    %           'gamma' (Gamma), 'logn' (Log-Normal), 'wbl' (Weibull).
    % 
    %---OUTPUTS: include the absolute area between the two distributions, the peak
    % separation, overlap integral, and relative entropy.
    % 
    ----------------------------------------
    """

    outnames = ('adiff',
                'olapint',
                'peaksepx',
                'peaksepy',
                'relent')

    def __init__(self, whatdbn='norm'):
        super(DN_CompareKSFit, self).__init__(add_descriptors=False)
        self.whatdbn = whatdbn

    def eval(self, engine, x):
        return HCTSA_DN_CompareKSFit(engine,
                                     x,
                                     whatdbn=self.whatdbn)


def HCTSA_DN_Compare_zscore(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares the distribution of a time series to a z-scored version of it
    % 
    %---INPUT:
    % x, a (not z-scored) time series
    % 
    %---OUTPUTS: ratios of features between the original and z-scored time series,
    % including the number of peaks, the maximum, and the distributional entropy.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['entropy',
                                                      'max',
                                                      'numpeaks']}
    out = eng.run_function(1, 'DN_Compare_zscore', x, )
    return outfunc(out)


class DN_Compare_zscore(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares the distribution of a time series to a z-scored version of it
    % 
    %---INPUT:
    % x, a (not z-scored) time series
    % 
    %---OUTPUTS: ratios of features between the original and z-scored time series,
    % including the number of peaks, the maximum, and the distributional entropy.
    % 
    ----------------------------------------
    """

    outnames = ('entropy',
                'max',
                'numpeaks')

    def __init__(self, ):
        super(DN_Compare_zscore, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_DN_Compare_zscore(engine, x)


def HCTSA_DN_Cumulants(eng, x, whatcum=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Very simple function that uses the skewness and kurtosis functions in 
    % Matlab's Statistics Toolbox to calculate these higher order moments of input time series, y
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % whatcum, the type of higher order moment:
    %           (i) 'skew1', skewness
    %           (ii) 'skew2', skewness correcting for bias
    %           (iii) 'kurt1', kurtosis
    %           (iv) 'kurt2', kurtosis correcting for bias
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if whatcum is None:
        out = eng.run_function(1, 'DN_Cumulants', x, )
    else:
        out = eng.run_function(1, 'DN_Cumulants', x, whatcum)
    return outfunc(out)


class DN_Cumulants(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Very simple function that uses the skewness and kurtosis functions in 
    % Matlab's Statistics Toolbox to calculate these higher order moments of input time series, y
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % whatcum, the type of higher order moment:
    %           (i) 'skew1', skewness
    %           (ii) 'skew2', skewness correcting for bias
    %           (iii) 'kurt1', kurtosis
    %           (iv) 'kurt2', kurtosis correcting for bias
    % 
    ----------------------------------------
    """

    def __init__(self, whatcum=None):
        super(DN_Cumulants, self).__init__(add_descriptors=False)
        self.whatcum = whatcum

    def eval(self, engine, x):
        return HCTSA_DN_Cumulants(engine,
                                  x,
                                  whatcum=self.whatcum)


def HCTSA_DN_CustomSkewness(eng, x, whichskew='pearson'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates custom skewness measures, the Pearson and Bowley skewnesses.
    % 
    % INPUTS:
    % y, the input time series
    % 
    % whichskew, the skewness measure to calculate, either 'pearson' or 'bowley'
    % 
    % The Bowley skewness uses the quantile function from Matlab's Statistics
    % Toolbox.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if whichskew is None:
        out = eng.run_function(1, 'DN_CustomSkewness', x, )
    else:
        out = eng.run_function(1, 'DN_CustomSkewness', x, whichskew)
    return outfunc(out)


class DN_CustomSkewness(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates custom skewness measures, the Pearson and Bowley skewnesses.
    % 
    % INPUTS:
    % y, the input time series
    % 
    % whichskew, the skewness measure to calculate, either 'pearson' or 'bowley'
    % 
    % The Bowley skewness uses the quantile function from Matlab's Statistics
    % Toolbox.
    % 
    ----------------------------------------
    """

    def __init__(self, whichskew='pearson'):
        super(DN_CustomSkewness, self).__init__(add_descriptors=False)
        self.whichskew = whichskew

    def eval(self, engine, x):
        return HCTSA_DN_CustomSkewness(engine,
                                       x,
                                       whichskew=self.whichskew)


def HCTSA_DN_FitKernelSmooth(eng, x, varargin='numcross'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a kernel-smoothed distribution to the data using the ksdensity function
    % from Matlab's Statistics Toolbox and returns a set of simple statistics.
    % 
    %---INPUTS:
    % x, the input time series
    % <can also produce additional outputs with the following optional settings>
    % [opt] 'numcross': number of times the distribution crosses the given threshold
    %           e.g., usage: DN_FitKernelSmooth(x,'numcross',[0.5,0.7]) for
    %                        thresholds of 0.5 and 0.7
    % [opt] 'area': area under where the distribution crosses the given thresholds.
    %               Usage as for 'numcross' above
    % [opt] 'arclength': arclength between where the distribution passes given
    %       thresholds. Usage as above.
    % 
    %---EXAMPLE USAGE:                  
    % DN_FitKernelSmooth(x,'numcross',[0.05,0.1],'area',[0.1,0.2,0.4],'arclength',[0.5,1,2])
    % returns all the basic outputs, plus those for numcross, area, and arclength
    % for the thresholds given
    % 
    %---OUTPUTS: a set of statistics summarizing the obtained distribution, including
    % the number of peaks, the distributional entropy, the number of times the curve
    % crosses fifixed probability thresholds, the area under the curve for fifixed
    % probability thresholds, the arc length, and the symmetry of probability
    % density above and below the mean.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['arclength_010',
                                                      'arclength_050',
                                                      'arclength_100',
                                                      'arclength_200',
                                                      'area_005',
                                                      'area_010',
                                                      'area_020',
                                                      'area_030',
                                                      'area_040',
                                                      'area_050',
                                                      'asym',
                                                      'entropy',
                                                      'max',
                                                      'npeaks',
                                                      'numcross_005',
                                                      'numcross_010',
                                                      'numcross_020',
                                                      'numcross_030',
                                                      'numcross_040',
                                                      'numcross_050',
                                                      'plsym']}
    if varargin is None:
        out = eng.run_function(1, 'DN_FitKernelSmooth', x, )
    else:
        out = eng.run_function(1, 'DN_FitKernelSmooth', x, varargin)
    return outfunc(out)


class DN_FitKernelSmooth(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a kernel-smoothed distribution to the data using the ksdensity function
    % from Matlab's Statistics Toolbox and returns a set of simple statistics.
    % 
    %---INPUTS:
    % x, the input time series
    % <can also produce additional outputs with the following optional settings>
    % [opt] 'numcross': number of times the distribution crosses the given threshold
    %           e.g., usage: DN_FitKernelSmooth(x,'numcross',[0.5,0.7]) for
    %                        thresholds of 0.5 and 0.7
    % [opt] 'area': area under where the distribution crosses the given thresholds.
    %               Usage as for 'numcross' above
    % [opt] 'arclength': arclength between where the distribution passes given
    %       thresholds. Usage as above.
    % 
    %---EXAMPLE USAGE:                  
    % DN_FitKernelSmooth(x,'numcross',[0.05,0.1],'area',[0.1,0.2,0.4],'arclength',[0.5,1,2])
    % returns all the basic outputs, plus those for numcross, area, and arclength
    % for the thresholds given
    % 
    %---OUTPUTS: a set of statistics summarizing the obtained distribution, including
    % the number of peaks, the distributional entropy, the number of times the curve
    % crosses fifixed probability thresholds, the area under the curve for fifixed
    % probability thresholds, the arc length, and the symmetry of probability
    % density above and below the mean.
    % 
    ----------------------------------------
    """

    outnames = ('arclength_010',
                'arclength_050',
                'arclength_100',
                'arclength_200',
                'area_005',
                'area_010',
                'area_020',
                'area_030',
                'area_040',
                'area_050',
                'asym',
                'entropy',
                'max',
                'npeaks',
                'numcross_005',
                'numcross_010',
                'numcross_020',
                'numcross_030',
                'numcross_040',
                'numcross_050',
                'plsym')

    def __init__(self, varargin='numcross'):
        super(DN_FitKernelSmooth, self).__init__(add_descriptors=False)
        self.varargin = varargin

    def eval(self, engine, x):
        return HCTSA_DN_FitKernelSmooth(engine,
                                        x,
                                        varargin=self.varargin)


def HCTSA_DN_Fit_mle(eng, x, fitwhat='geometric'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits either a Gaussian, Uniform, or Geometric distribution to the data using
    % maximum likelihood estimation via the Matlab function mle
    % from the Statistics Toolbox.
    % 
    %---INPUTS:
    % y, the time series
    % fitwhat, the type of fit to do: 'gaussian', 'uniform', or 'geometric'.
    % 
    %---OUTPUTS: parameters from the fit.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['None',
                                                      'mean',
                                                      'std']}
    if fitwhat is None:
        out = eng.run_function(1, 'DN_Fit_mle', x, )
    else:
        out = eng.run_function(1, 'DN_Fit_mle', x, fitwhat)
    return outfunc(out)


class DN_Fit_mle(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits either a Gaussian, Uniform, or Geometric distribution to the data using
    % maximum likelihood estimation via the Matlab function mle
    % from the Statistics Toolbox.
    % 
    %---INPUTS:
    % y, the time series
    % fitwhat, the type of fit to do: 'gaussian', 'uniform', or 'geometric'.
    % 
    %---OUTPUTS: parameters from the fit.
    % 
    ----------------------------------------
    """

    outnames = ('None',
                'mean',
                'std')

    def __init__(self, fitwhat='geometric'):
        super(DN_Fit_mle, self).__init__(add_descriptors=False)
        self.fitwhat = fitwhat

    def eval(self, engine, x):
        return HCTSA_DN_Fit_mle(engine,
                                x,
                                fitwhat=self.fitwhat)


def HCTSA_DN_HighLowMu(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates a statistic related to the mean of the time series data that
    % is above the (global) time-series mean compared to the mean of the data that
    % is below the global time-series mean.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'DN_HighLowMu', x, )
    return outfunc(out)


class DN_HighLowMu(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates a statistic related to the mean of the time series data that
    % is above the (global) time-series mean compared to the mean of the data that
    % is below the global time-series mean.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(DN_HighLowMu, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_DN_HighLowMu(engine, x)


def HCTSA_DN_HistogramMode(eng, x, nbins=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the mode of the time series using histograms a given numbers
    % of bins.
    % 
    %---INPUTS:
    % 
    % y, the input time series.
    % 
    % nbins, the number of bins to use in the histogram.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if nbins is None:
        out = eng.run_function(1, 'DN_HistogramMode', x, )
    else:
        out = eng.run_function(1, 'DN_HistogramMode', x, nbins)
    return outfunc(out)


class DN_HistogramMode(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the mode of the time series using histograms a given numbers
    % of bins.
    % 
    %---INPUTS:
    % 
    % y, the input time series.
    % 
    % nbins, the number of bins to use in the histogram.
    % 
    ----------------------------------------
    """

    def __init__(self, nbins=10):
        super(DN_HistogramMode, self).__init__(add_descriptors=False)
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_DN_HistogramMode(engine,
                                      x,
                                      nbins=self.nbins)


def HCTSA_DN_Mean(eng, x, meantype='median'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures a given type of 'mean', or measure of location of the time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % meantype, (i) 'norm' or 'arithmetic', arithmetic mean
    %           (ii) 'median', median
    %           (iii) 'geom', geometric mean
    %           (iv) 'harm', harmonic mean
    %           (v) 'rms', root-mean-square
    %           (vi) 'iqm', interquartile mean
    %           (vii) 'midhinge', midhinge
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if meantype is None:
        out = eng.run_function(1, 'DN_Mean', x, )
    else:
        out = eng.run_function(1, 'DN_Mean', x, meantype)
    return outfunc(out)


class DN_Mean(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures a given type of 'mean', or measure of location of the time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % meantype, (i) 'norm' or 'arithmetic', arithmetic mean
    %           (ii) 'median', median
    %           (iii) 'geom', geometric mean
    %           (iv) 'harm', harmonic mean
    %           (v) 'rms', root-mean-square
    %           (vi) 'iqm', interquartile mean
    %           (vii) 'midhinge', midhinge
    % 
    ----------------------------------------
    """

    def __init__(self, meantype='median'):
        super(DN_Mean, self).__init__(add_descriptors=False)
        self.meantype = meantype

    def eval(self, engine, x):
        return HCTSA_DN_Mean(engine,
                             x,
                             meantype=self.meantype)


def HCTSA_DN_MinMax(eng, x, minormax='max'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the maximum and minimum values of the input time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % minormax, either 'min' or 'max' to return either the minimum or maximum of y
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if minormax is None:
        out = eng.run_function(1, 'DN_MinMax', x, )
    else:
        out = eng.run_function(1, 'DN_MinMax', x, minormax)
    return outfunc(out)


class DN_MinMax(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the maximum and minimum values of the input time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % minormax, either 'min' or 'max' to return either the minimum or maximum of y
    % 
    ----------------------------------------
    """

    def __init__(self, minormax='max'):
        super(DN_MinMax, self).__init__(add_descriptors=False)
        self.minormax = minormax

    def eval(self, engine, x):
        return HCTSA_DN_MinMax(engine,
                               x,
                               minormax=self.minormax)


def HCTSA_DN_Moments(eng, x, n=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Output is the moment of the distribution of the input time series.
    % Normalizes by the standard deviation
    % Uses the moment function from Matlab's Statistics Toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % n, the moment to calculate (a scalar)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if n is None:
        out = eng.run_function(1, 'DN_Moments', x, )
    else:
        out = eng.run_function(1, 'DN_Moments', x, n)
    return outfunc(out)


class DN_Moments(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Output is the moment of the distribution of the input time series.
    % Normalizes by the standard deviation
    % Uses the moment function from Matlab's Statistics Toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % n, the moment to calculate (a scalar)
    % 
    ----------------------------------------
    """

    def __init__(self, n=3):
        super(DN_Moments, self).__init__(add_descriptors=False)
        self.n = n

    def eval(self, engine, x):
        return HCTSA_DN_Moments(engine,
                                x,
                                n=self.n)


def HCTSA_DN_OutlierInclude(eng, x, howth='abs', inc=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures a range of different statistics about the time series as more and
    % more outliers are included in the calculation according to a specified rule:
    % 
    % (i) 'abs': outliers are furthest from the mean,
    % (ii) 'p': outliers are the greatest positive deviations from the mean, or
    % (iii) 'n': outliers are the greatest negative deviations from the mean.
    % 
    % The threshold for including time-series data points in the analysis increases
    % from zero to the maximum deviation, in increments of 0.01*sigma (by default),
    % where sigma is the standard deviation of the time series.
    % 
    % At each threshold, the mean, standard error, proportion of time series points
    % included, median, and standard deviation are calculated, and outputs from the
    % algorithm measure how these statistical quantities change as more extreme
    % points are included in the calculation.
    % 
    %---INPUTS:
    % y, the input time series (ideally z-scored)
    % 
    % howth, the method of how to determine outliers: 'abs', 'p', or 'n' (see above
    %           for descriptions)
    % 
    % inc, the increment to move through (fraction of std if input time series is
    %       z-scored)
    % 
    % Most of the outputs measure either exponential, i.e., f(x) = Aexp(Bx)+C, or
    % linear, i.e., f(x) = Ax + B, fits to the sequence of statistics obtained in
    % this way.
    % 
    % [future: could compare differences in outputs obtained with 'p', 'n', and
    %               'abs' -- could give an idea as to asymmetries/nonstationarities??]
    %               
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['mdrm',
                                                      'mdrmd',
                                                      'mdrstd',
                                                      'mfexpa',
                                                      'mfexpadjr2',
                                                      'mfexpb',
                                                      'mfexpc',
                                                      'mfexpr2',
                                                      'mfexprmse',
                                                      'mrm',
                                                      'mrmd',
                                                      'mrstd',
                                                      'nfexpa',
                                                      'nfexpadjr2',
                                                      'nfexpb',
                                                      'nfexpc',
                                                      'nfexpr2',
                                                      'nfexprmse',
                                                      'nfla',
                                                      'nfladjr2',
                                                      'nflb',
                                                      'nflr2',
                                                      'nflrmse',
                                                      'stdrfexpa',
                                                      'stdrfexpadjr2',
                                                      'stdrfexpb',
                                                      'stdrfexpc',
                                                      'stdrfexpr2',
                                                      'stdrfexprmse',
                                                      'stdrfla',
                                                      'stdrfladjr2',
                                                      'stdrflb',
                                                      'stdrflr2',
                                                      'stdrflrmse',
                                                      'xcmerr1',
                                                      'xcmerrn1']}
    if howth is None:
        out = eng.run_function(1, 'DN_OutlierInclude', x, )
    elif inc is None:
        out = eng.run_function(1, 'DN_OutlierInclude', x, howth)
    else:
        out = eng.run_function(1, 'DN_OutlierInclude', x, howth, inc)
    return outfunc(out)


class DN_OutlierInclude(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures a range of different statistics about the time series as more and
    % more outliers are included in the calculation according to a specified rule:
    % 
    % (i) 'abs': outliers are furthest from the mean,
    % (ii) 'p': outliers are the greatest positive deviations from the mean, or
    % (iii) 'n': outliers are the greatest negative deviations from the mean.
    % 
    % The threshold for including time-series data points in the analysis increases
    % from zero to the maximum deviation, in increments of 0.01*sigma (by default),
    % where sigma is the standard deviation of the time series.
    % 
    % At each threshold, the mean, standard error, proportion of time series points
    % included, median, and standard deviation are calculated, and outputs from the
    % algorithm measure how these statistical quantities change as more extreme
    % points are included in the calculation.
    % 
    %---INPUTS:
    % y, the input time series (ideally z-scored)
    % 
    % howth, the method of how to determine outliers: 'abs', 'p', or 'n' (see above
    %           for descriptions)
    % 
    % inc, the increment to move through (fraction of std if input time series is
    %       z-scored)
    % 
    % Most of the outputs measure either exponential, i.e., f(x) = Aexp(Bx)+C, or
    % linear, i.e., f(x) = Ax + B, fits to the sequence of statistics obtained in
    % this way.
    % 
    % [future: could compare differences in outputs obtained with 'p', 'n', and
    %               'abs' -- could give an idea as to asymmetries/nonstationarities??]
    %               
    ----------------------------------------
    """

    outnames = ('mdrm',
                'mdrmd',
                'mdrstd',
                'mfexpa',
                'mfexpadjr2',
                'mfexpb',
                'mfexpc',
                'mfexpr2',
                'mfexprmse',
                'mrm',
                'mrmd',
                'mrstd',
                'nfexpa',
                'nfexpadjr2',
                'nfexpb',
                'nfexpc',
                'nfexpr2',
                'nfexprmse',
                'nfla',
                'nfladjr2',
                'nflb',
                'nflr2',
                'nflrmse',
                'stdrfexpa',
                'stdrfexpadjr2',
                'stdrfexpb',
                'stdrfexpc',
                'stdrfexpr2',
                'stdrfexprmse',
                'stdrfla',
                'stdrfladjr2',
                'stdrflb',
                'stdrflr2',
                'stdrflrmse',
                'xcmerr1',
                'xcmerrn1')

    def __init__(self, howth='abs', inc=None):
        super(DN_OutlierInclude, self).__init__(add_descriptors=False)
        self.howth = howth
        self.inc = inc

    def eval(self, engine, x):
        return HCTSA_DN_OutlierInclude(engine,
                                       x,
                                       howth=self.howth,
                                       inc=self.inc)


def HCTSA_DN_OutlierTest(eng, x, p=2, justme=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Removes the p% of highest and lowest values in the time series (i.e., 2*p%
    % removed from the time series in total) and returns the ratio of either the
    % mean or the standard deviation of the time series, before and after this
    % transformation.
    % 
    %---INPUTS:
    % y, the input time series
    % p, the percentage of values to remove beyond upper and lower percentiles
    % justme [opt], just returns a number:
    %               (i) 'mean' -- returns the mean of the middle portion of the data
    %               (ii) 'std' -- returns the std of the middle portion of the data
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['mean',
                                                      'std']}
    if p is None:
        out = eng.run_function(1, 'DN_OutlierTest', x, )
    elif justme is None:
        out = eng.run_function(1, 'DN_OutlierTest', x, p)
    else:
        out = eng.run_function(1, 'DN_OutlierTest', x, p, justme)
    return outfunc(out)


class DN_OutlierTest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Removes the p% of highest and lowest values in the time series (i.e., 2*p%
    % removed from the time series in total) and returns the ratio of either the
    % mean or the standard deviation of the time series, before and after this
    % transformation.
    % 
    %---INPUTS:
    % y, the input time series
    % p, the percentage of values to remove beyond upper and lower percentiles
    % justme [opt], just returns a number:
    %               (i) 'mean' -- returns the mean of the middle portion of the data
    %               (ii) 'std' -- returns the std of the middle portion of the data
    % 
    ----------------------------------------
    """

    outnames = ('mean',
                'std')

    def __init__(self, p=2, justme=None):
        super(DN_OutlierTest, self).__init__(add_descriptors=False)
        self.p = p
        self.justme = justme

    def eval(self, engine, x):
        return HCTSA_DN_OutlierTest(engine,
                                    x,
                                    p=self.p,
                                    justme=self.justme)


def HCTSA_DN_ProportionValues(eng, x, propwhat='positive'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on the values of the raw time series: the proportion
    % of zeros in the raw time series, the proportion of positive values, and the
    % proportion of values greater than or equal to zero.
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % propwhat, the proportion of a given type of value in the time series:
    %           (i) 'zeros': values that equal zero
    %           (ii) 'positive': values that are strictly positive
    %           (iii) 'geq0': values that are greater than or equal to zero
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if propwhat is None:
        out = eng.run_function(1, 'DN_ProportionValues', x, )
    else:
        out = eng.run_function(1, 'DN_ProportionValues', x, propwhat)
    return outfunc(out)


class DN_ProportionValues(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on the values of the raw time series: the proportion
    % of zeros in the raw time series, the proportion of positive values, and the
    % proportion of values greater than or equal to zero.
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % propwhat, the proportion of a given type of value in the time series:
    %           (i) 'zeros': values that equal zero
    %           (ii) 'positive': values that are strictly positive
    %           (iii) 'geq0': values that are greater than or equal to zero
    % 
    ----------------------------------------
    """

    def __init__(self, propwhat='positive'):
        super(DN_ProportionValues, self).__init__(add_descriptors=False)
        self.propwhat = propwhat

    def eval(self, engine, x):
        return HCTSA_DN_ProportionValues(engine,
                                         x,
                                         propwhat=self.propwhat)


def HCTSA_DN_Quantile(eng, x, p=0.6):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the quantile value at a specified proportion p using the Statistics
    % Toolbox function, quantile.
    % 
    % INPUTS:
    % y, the input time series
    % p, the quantile proportion
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if p is None:
        out = eng.run_function(1, 'DN_Quantile', x, )
    else:
        out = eng.run_function(1, 'DN_Quantile', x, p)
    return outfunc(out)


class DN_Quantile(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the quantile value at a specified proportion p using the Statistics
    % Toolbox function, quantile.
    % 
    % INPUTS:
    % y, the input time series
    % p, the quantile proportion
    % 
    ----------------------------------------
    """

    def __init__(self, p=0.6):
        super(DN_Quantile, self).__init__(add_descriptors=False)
        self.p = p

    def eval(self, engine, x):
        return HCTSA_DN_Quantile(engine,
                                 x,
                                 p=self.p)


def HCTSA_DN_RemovePoints(eng, x, howtorem='absclose', p=0.1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyze how some time-series properties function changes as points are removed
    % from a time series.
    % 
    % A proportion, p, of points are removed from the time series according to some
    % rule, and a set of statistics are computed before and after the change.
    % 
    % INPUTS:
    % y, the input time series
    % howtorem, how to remove points from the time series:
    %               (i) 'absclose': those that are the closest to the mean,
    %               (ii) 'absfar': those that are the furthest from the mean,
    %               (iii) 'min': the lowest values,
    %               (iv) 'max': the highest values,
    %               (v) 'random': at random.
    %               
    % p, the proportion of points to remove
    % 
    % Output statistics include the change in autocorrelation, time scales, mean,
    % spread, and skewness.
    % 
    % Note that this is a similar idea to that implemented in DN_OutlierInclude.
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac2diff',
                                                      'ac2rat',
                                                      'ac3diff',
                                                      'ac3rat',
                                                      'fzcacrat',
                                                      'kurtosisrat',
                                                      'mean',
                                                      'median',
                                                      'skewnessrat',
                                                      'std',
                                                      'sumabsacfdiff']}
    if howtorem is None:
        out = eng.run_function(1, 'DN_RemovePoints', x, )
    elif p is None:
        out = eng.run_function(1, 'DN_RemovePoints', x, howtorem)
    else:
        out = eng.run_function(1, 'DN_RemovePoints', x, howtorem, p)
    return outfunc(out)


class DN_RemovePoints(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyze how some time-series properties function changes as points are removed
    % from a time series.
    % 
    % A proportion, p, of points are removed from the time series according to some
    % rule, and a set of statistics are computed before and after the change.
    % 
    % INPUTS:
    % y, the input time series
    % howtorem, how to remove points from the time series:
    %               (i) 'absclose': those that are the closest to the mean,
    %               (ii) 'absfar': those that are the furthest from the mean,
    %               (iii) 'min': the lowest values,
    %               (iv) 'max': the highest values,
    %               (v) 'random': at random.
    %               
    % p, the proportion of points to remove
    % 
    % Output statistics include the change in autocorrelation, time scales, mean,
    % spread, and skewness.
    % 
    % Note that this is a similar idea to that implemented in DN_OutlierInclude.
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """

    outnames = ('ac2diff',
                'ac2rat',
                'ac3diff',
                'ac3rat',
                'fzcacrat',
                'kurtosisrat',
                'mean',
                'median',
                'skewnessrat',
                'std',
                'sumabsacfdiff')

    def __init__(self, howtorem='absclose', p=0.1):
        super(DN_RemovePoints, self).__init__(add_descriptors=False)
        self.howtorem = howtorem
        self.p = p

    def eval(self, engine, x):
        return HCTSA_DN_RemovePoints(engine,
                                     x,
                                     howtorem=self.howtorem,
                                     p=self.p)


def HCTSA_DN_SimpleFit(eng, x, dmodel='exp1', nbins=0):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits different distributions or simple time-series models to the time series
    % using 'fit' function from Matlab's Curve Fitting Toolbox.
    % 
    % The distribution of time-series values is estimated using either a
    % kernel-smoothed density via the Matlab function ksdensity with the default
    % width parameter, or by a histogram with a specified number of bins, nbins.
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % dmodel, the model to fit:
    %       (I) distribution models:
    %           (i) 'gauss1'
    %           (ii) 'gauss2'
    %           (iii) 'exp1'
    %           (iv) 'power1'
    %       (II) simple time-series models:
    %           (i) 'sin1'
    %           (ii) 'sin2'
    %           (iii) 'sin3'
    %           (iv) 'fourier1'
    %           (v) 'fourier2'
    %           (vi) 'fourier3'
    % 
    % nbins, the number of bins for a histogram-estimate of the distribution of
    %       time-series values. If nbins = 0, uses ksdensity instead of histogram.
    % 
    % 
    %---OUTPUTS: the goodness of fifit, R^2, rootmean square error, the
    % autocorrelation of the residuals, and a runs test on the residuals.
    % 
    % 
    %---HISTORY:
    % Ben Fulcher, 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['adjr2',
                                                      'r2',
                                                      'resAC1',
                                                      'resAC2',
                                                      'resruns',
                                                      'rmse']}
    if dmodel is None:
        out = eng.run_function(1, 'DN_SimpleFit', x, )
    elif nbins is None:
        out = eng.run_function(1, 'DN_SimpleFit', x, dmodel)
    else:
        out = eng.run_function(1, 'DN_SimpleFit', x, dmodel, nbins)
    return outfunc(out)


class DN_SimpleFit(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits different distributions or simple time-series models to the time series
    % using 'fit' function from Matlab's Curve Fitting Toolbox.
    % 
    % The distribution of time-series values is estimated using either a
    % kernel-smoothed density via the Matlab function ksdensity with the default
    % width parameter, or by a histogram with a specified number of bins, nbins.
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % dmodel, the model to fit:
    %       (I) distribution models:
    %           (i) 'gauss1'
    %           (ii) 'gauss2'
    %           (iii) 'exp1'
    %           (iv) 'power1'
    %       (II) simple time-series models:
    %           (i) 'sin1'
    %           (ii) 'sin2'
    %           (iii) 'sin3'
    %           (iv) 'fourier1'
    %           (v) 'fourier2'
    %           (vi) 'fourier3'
    % 
    % nbins, the number of bins for a histogram-estimate of the distribution of
    %       time-series values. If nbins = 0, uses ksdensity instead of histogram.
    % 
    % 
    %---OUTPUTS: the goodness of fifit, R^2, rootmean square error, the
    % autocorrelation of the residuals, and a runs test on the residuals.
    % 
    % 
    %---HISTORY:
    % Ben Fulcher, 2009
    % 
    ----------------------------------------
    """

    outnames = ('adjr2',
                'r2',
                'resAC1',
                'resAC2',
                'resruns',
                'rmse')

    def __init__(self, dmodel='exp1', nbins=0):
        super(DN_SimpleFit, self).__init__(add_descriptors=False)
        self.dmodel = dmodel
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_DN_SimpleFit(engine,
                                  x,
                                  dmodel=self.dmodel,
                                  nbins=self.nbins)


def HCTSA_DN_Spread(eng, x, SpreadMeasure='std'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the spread of the raw time series, as the standard deviation,
    % inter-quartile range, mean absolute deviation, or median absolute deviation.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % SpreadMeasure, the spead measure:
    %               (i) 'std': standard deviation
    %               (ii) 'iqr': interquartile range
    %               (iii) 'mad': mean absolute deviation
    %               (iv) 'mead': median absolute deviation
    % 
    %---HISTORY:
    % Ben Fulcher, 2008
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if SpreadMeasure is None:
        out = eng.run_function(1, 'DN_Spread', x, )
    else:
        out = eng.run_function(1, 'DN_Spread', x, SpreadMeasure)
    return outfunc(out)


class DN_Spread(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns the spread of the raw time series, as the standard deviation,
    % inter-quartile range, mean absolute deviation, or median absolute deviation.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % SpreadMeasure, the spead measure:
    %               (i) 'std': standard deviation
    %               (ii) 'iqr': interquartile range
    %               (iii) 'mad': mean absolute deviation
    %               (iv) 'mead': median absolute deviation
    % 
    %---HISTORY:
    % Ben Fulcher, 2008
    ----------------------------------------
    """

    def __init__(self, SpreadMeasure='std'):
        super(DN_Spread, self).__init__(add_descriptors=False)
        self.SpreadMeasure = SpreadMeasure

    def eval(self, engine, x):
        return HCTSA_DN_Spread(engine,
                               x,
                               SpreadMeasure=self.SpreadMeasure)


def HCTSA_DN_TrimmedMean(eng, x, n=2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Outputs the mean of the trimmed time series using the Matlab function
    % trimmean.
    % 
    %---INPUTS:
    % y, the input time series
    % n, the percent of highest and lowest values in y to exclude from the mean
    %     calculation
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if n is None:
        out = eng.run_function(1, 'DN_TrimmedMean', x, )
    else:
        out = eng.run_function(1, 'DN_TrimmedMean', x, n)
    return outfunc(out)


class DN_TrimmedMean(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Outputs the mean of the trimmed time series using the Matlab function
    % trimmean.
    % 
    %---INPUTS:
    % y, the input time series
    % n, the percent of highest and lowest values in y to exclude from the mean
    %     calculation
    % 
    ----------------------------------------
    """

    def __init__(self, n=2):
        super(DN_TrimmedMean, self).__init__(add_descriptors=False)
        self.n = n

    def eval(self, engine, x):
        return HCTSA_DN_TrimmedMean(engine,
                                    x,
                                    n=self.n)


def HCTSA_DN_Withinp(eng, x, p=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the proportion of the time-series data points that lie within
    % p standard deviations of its mean.
    % 
    %---INPUTS:
    % x, the input time series
    % p, the number (proportion) of standard deviations.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if p is None:
        out = eng.run_function(1, 'DN_Withinp', x, )
    else:
        out = eng.run_function(1, 'DN_Withinp', x, p)
    return outfunc(out)


class DN_Withinp(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the proportion of the time-series data points that lie within
    % p standard deviations of its mean.
    % 
    %---INPUTS:
    % x, the input time series
    % p, the number (proportion) of standard deviations.
    % 
    ----------------------------------------
    """

    def __init__(self, p=3):
        super(DN_Withinp, self).__init__(add_descriptors=False)
        self.p = p

    def eval(self, engine, x):
        return HCTSA_DN_Withinp(engine,
                                x,
                                p=self.p)


def HCTSA_DN_cv(eng, x, k=4):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the coefficient of variation, sigma^k / mu^k, of order k.
    % 
    %---INPUTS:
    % 
    % x, the input time series
    % 
    % k, the order of coefficient of variation (k = 1 is usual)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if k is None:
        out = eng.run_function(1, 'DN_cv', x, )
    else:
        out = eng.run_function(1, 'DN_cv', x, k)
    return outfunc(out)


class DN_cv(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the coefficient of variation, sigma^k / mu^k, of order k.
    % 
    %---INPUTS:
    % 
    % x, the input time series
    % 
    % k, the order of coefficient of variation (k = 1 is usual)
    % 
    ----------------------------------------
    """

    def __init__(self, k=4):
        super(DN_cv, self).__init__(add_descriptors=False)
        self.k = k

    def eval(self, engine, x):
        return HCTSA_DN_cv(engine,
                           x,
                           k=self.k)


def HCTSA_DN_nlogL_norm(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a Gaussian distribution to the data using the normfit function in
    % Matlab's Statistics Toolbox and returns the negative log likelihood of the
    % data coming from a Gaussian distribution using the normlike function.
    % 
    %---INPUT:
    % y, the time series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'DN_nlogL_norm', x, )
    return outfunc(out)


class DN_nlogL_norm(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a Gaussian distribution to the data using the normfit function in
    % Matlab's Statistics Toolbox and returns the negative log likelihood of the
    % data coming from a Gaussian distribution using the normlike function.
    % 
    %---INPUT:
    % y, the time series.
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(DN_nlogL_norm, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_DN_nlogL_norm(engine, x)


def HCTSA_DN_pleft(eng, x, th=0.1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the maximum distance from the mean at which a given fixed proportion,
    % p, of the time-series data points are further.
    % Normalizes by the standard deviation of the time series
    % (could generalize to separate positive and negative deviations in future)
    % Uses the quantile function from Matlab's Statistics Toolbox
    %
    %---INPUTS:
    % y, the input time series
    % th, the proportion of data further than p from the mean
    %           (output p, normalized by standard deviation)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if th is None:
        out = eng.run_function(1, 'DN_pleft', x, )
    else:
        out = eng.run_function(1, 'DN_pleft', x, th)
    return outfunc(out)


class DN_pleft(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the maximum distance from the mean at which a given fixed proportion,
    % p, of the time-series data points are further.
    % Normalizes by the standard deviation of the time series
    % (could generalize to separate positive and negative deviations in future)
    % Uses the quantile function from Matlab's Statistics Toolbox
    %
    %---INPUTS:
    % y, the input time series
    % th, the proportion of data further than p from the mean
    %           (output p, normalized by standard deviation)
    % 
    ----------------------------------------
    """

    def __init__(self, th=0.1):
        super(DN_pleft, self).__init__(add_descriptors=False)
        self.th = th

    def eval(self, engine, x):
        return HCTSA_DN_pleft(engine,
                              x,
                              th=self.th)


def HCTSA_DT_IsSeasonal(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % A simple test of seasonality by fitting a 'sin1' model to the time series
    % using fit function from the Curve Fitting Toolbox. The output is binary: 1 if
    % the goodness of fit, R^2, exceeds 0.3 and the amplitude of the fitted periodic
    % component exceeds 0.5, and 0 otherwise.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    %---OUTPUT: Binary: 1 (= seasonal), 0 (= non-seasonal)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'DT_IsSeasonal', x, )
    return outfunc(out)


class DT_IsSeasonal(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % A simple test of seasonality by fitting a 'sin1' model to the time series
    % using fit function from the Curve Fitting Toolbox. The output is binary: 1 if
    % the goodness of fit, R^2, exceeds 0.3 and the amplitude of the fitted periodic
    % component exceeds 0.5, and 0 otherwise.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    %---OUTPUT: Binary: 1 (= seasonal), 0 (= non-seasonal)
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(DT_IsSeasonal, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_DT_IsSeasonal(engine, x)


def HCTSA_EN_ApEn(eng, x, mnom=1, rth=0.2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the Approximate Entropy of the time series, ApEn(m,r).
    % 
    % cf. S. M. Pincus, "Approximate entropy as a measure of system complexity",
    % P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)
    %
    % For more information, cf. http://physionet.org/physiotools/ApEn/
    % 
    %--INPUTS:
    % y, the input time series
    % mnom, the embedding dimension
    % rth, the threshold for judging closeness/similarity
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if mnom is None:
        out = eng.run_function(1, 'EN_ApEn', x, )
    elif rth is None:
        out = eng.run_function(1, 'EN_ApEn', x, mnom)
    else:
        out = eng.run_function(1, 'EN_ApEn', x, mnom, rth)
    return outfunc(out)


class EN_ApEn(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the Approximate Entropy of the time series, ApEn(m,r).
    % 
    % cf. S. M. Pincus, "Approximate entropy as a measure of system complexity",
    % P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)
    %
    % For more information, cf. http://physionet.org/physiotools/ApEn/
    % 
    %--INPUTS:
    % y, the input time series
    % mnom, the embedding dimension
    % rth, the threshold for judging closeness/similarity
    %
    ----------------------------------------
    """

    def __init__(self, mnom=1, rth=0.2):
        super(EN_ApEn, self).__init__(add_descriptors=False)
        self.mnom = mnom
        self.rth = rth

    def eval(self, engine, x):
        return HCTSA_EN_ApEn(engine,
                             x,
                             mnom=self.mnom,
                             rth=self.rth)


def HCTSA_EN_DistributionEntropy(eng, x, historks='ks', nbins=(), olremp=0.1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates of entropy from the static distribution of the time series. The
    % distribution is estimated either using a histogram with nbins bins, or as a
    % kernel-smoothed distribution, using the ksdensity function from Matlab's
    % Statistics Toolbox with width parameter, w (specified as the iunput nbins).
    % 
    % An optional additional parameter can be used to remove a proportion of the
    % most extreme positive and negative deviations from the mean as an initial
    % pre-processing.
    % 
    %---INPUTS:
    % y, the input time series
    % historks: 'hist' for histogram, or 'ks' for ksdensity
    % nbins: (*) (for 'hist'): an integer, uses a histogram with that many bins (for 'hist')
    %        (*) (for 'ks'): a positive real number, for the width parameter for ksdensity
    %                        (can also be empty for default width parameter, optimum for Gaussian)
    % olremp [opt]: the proportion of outliers at both extremes to remove
    %               (e.g., if olremp = 0.01; keeps only the middle 98% of data; 0 keeps all data.
    %               This parameter ought to be less than 0.5, which keeps none of the data)
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if historks is None:
        out = eng.run_function(1, 'EN_DistributionEntropy', x, )
    elif nbins is None:
        out = eng.run_function(1, 'EN_DistributionEntropy', x, historks)
    elif olremp is None:
        out = eng.run_function(1, 'EN_DistributionEntropy', x, historks, nbins)
    else:
        out = eng.run_function(1, 'EN_DistributionEntropy', x, historks, nbins, olremp)
    return outfunc(out)


class EN_DistributionEntropy(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates of entropy from the static distribution of the time series. The
    % distribution is estimated either using a histogram with nbins bins, or as a
    % kernel-smoothed distribution, using the ksdensity function from Matlab's
    % Statistics Toolbox with width parameter, w (specified as the iunput nbins).
    % 
    % An optional additional parameter can be used to remove a proportion of the
    % most extreme positive and negative deviations from the mean as an initial
    % pre-processing.
    % 
    %---INPUTS:
    % y, the input time series
    % historks: 'hist' for histogram, or 'ks' for ksdensity
    % nbins: (*) (for 'hist'): an integer, uses a histogram with that many bins (for 'hist')
    %        (*) (for 'ks'): a positive real number, for the width parameter for ksdensity
    %                        (can also be empty for default width parameter, optimum for Gaussian)
    % olremp [opt]: the proportion of outliers at both extremes to remove
    %               (e.g., if olremp = 0.01; keeps only the middle 98% of data; 0 keeps all data.
    %               This parameter ought to be less than 0.5, which keeps none of the data)
    %
    ----------------------------------------
    """

    def __init__(self, historks='ks', nbins=(), olremp=0.1):
        super(EN_DistributionEntropy, self).__init__(add_descriptors=False)
        self.historks = historks
        self.nbins = nbins
        self.olremp = olremp

    def eval(self, engine, x):
        return HCTSA_EN_DistributionEntropy(engine,
                                            x,
                                            historks=self.historks,
                                            nbins=self.nbins,
                                            olremp=self.olremp)


def HCTSA_EN_MS_shannon(eng, x, nbin=2, depth=2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the approximate Shannon entropy of a time series using an
    % nbin-bin encoding and depth-symbol sequences.
    % Uniform population binning is used, and the implementation uses Michael Small's code
    % MS_shannon.m (renamed from the original, simply shannon.m)
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Michael Small's code is available at available at http://small.eie.polyu.edu.hk/matlab/
    % 
    % In this wrapper function, you can evaluate the code at a given n and d, and
    % also across a range of depth and nbin to return statistics on how the obtained
    % entropies change.
    % 
    % INPUTS:
    % y, the input time series
    % nbin, the number of bins to discretize the time series into (i.e., alphabet size)
    % depth, the length of strings to analyze
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['None',
                                                      'maxent',
                                                      'meanent',
                                                      'medent',
                                                      'minent',
                                                      'stdent']}
    if nbin is None:
        out = eng.run_function(1, 'EN_MS_shannon', x, )
    elif depth is None:
        out = eng.run_function(1, 'EN_MS_shannon', x, nbin)
    else:
        out = eng.run_function(1, 'EN_MS_shannon', x, nbin, depth)
    return outfunc(out)


class EN_MS_shannon(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the approximate Shannon entropy of a time series using an
    % nbin-bin encoding and depth-symbol sequences.
    % Uniform population binning is used, and the implementation uses Michael Small's code
    % MS_shannon.m (renamed from the original, simply shannon.m)
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Michael Small's code is available at available at http://small.eie.polyu.edu.hk/matlab/
    % 
    % In this wrapper function, you can evaluate the code at a given n and d, and
    % also across a range of depth and nbin to return statistics on how the obtained
    % entropies change.
    % 
    % INPUTS:
    % y, the input time series
    % nbin, the number of bins to discretize the time series into (i.e., alphabet size)
    % depth, the length of strings to analyze
    % 
    ----------------------------------------
    """

    outnames = ('None',
                'maxent',
                'meanent',
                'medent',
                'minent',
                'stdent')

    def __init__(self, nbin=2, depth=2):
        super(EN_MS_shannon, self).__init__(add_descriptors=False)
        self.nbin = nbin
        self.depth = depth

    def eval(self, engine, x):
        return HCTSA_EN_MS_shannon(engine,
                                   x,
                                   nbin=self.nbin,
                                   depth=self.depth)


def HCTSA_EN_PermEn(eng, x, ord_=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the permutation entropy of order, ord, of a time series.
    % 
    % "Permutation Entropy: A Natural Complexity Measure for Time Series"
    % C. Bandt and B. Pompe, Phys. Rev. Lett. 88(17) 174102 (2002)
    % 
    % Code is adapted from logisticPE.m code obtained from
    % http://people.ece.cornell.edu/land/PROJECTS/Complexity/
    % http://people.ece.cornell.edu/land/PROJECTS/Complexity/logisticPE.m
    % 
    % INPUTS:
    % y, a time series
    % ord, the order of permutation entropy
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if ord_ is None:
        out = eng.run_function(1, 'EN_PermEn', x, )
    else:
        out = eng.run_function(1, 'EN_PermEn', x, ord_)
    return outfunc(out)


class EN_PermEn(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the permutation entropy of order, ord, of a time series.
    % 
    % "Permutation Entropy: A Natural Complexity Measure for Time Series"
    % C. Bandt and B. Pompe, Phys. Rev. Lett. 88(17) 174102 (2002)
    % 
    % Code is adapted from logisticPE.m code obtained from
    % http://people.ece.cornell.edu/land/PROJECTS/Complexity/
    % http://people.ece.cornell.edu/land/PROJECTS/Complexity/logisticPE.m
    % 
    % INPUTS:
    % y, a time series
    % ord, the order of permutation entropy
    % 
    ----------------------------------------
    """

    def __init__(self, ord_=3):
        super(EN_PermEn, self).__init__(add_descriptors=False)
        self.ord_ = ord_

    def eval(self, engine, x):
        return HCTSA_EN_PermEn(engine,
                               x,
                               ord_=self.ord_)


def HCTSA_EN_RM_entropy(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the entropy of the time series using a function by Rudy Moddemeijer
    % 
    % Original code, now RM_entropy, was obtained from:
    % http://www.cs.rug.nl/~rudy/matlab/
    % 
    % The above website has code and documentation for the function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'EN_RM_entropy', x, )
    return outfunc(out)


class EN_RM_entropy(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the entropy of the time series using a function by Rudy Moddemeijer
    % 
    % Original code, now RM_entropy, was obtained from:
    % http://www.cs.rug.nl/~rudy/matlab/
    % 
    % The above website has code and documentation for the function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(EN_RM_entropy, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_EN_RM_entropy(engine, x)


def HCTSA_EN_Randomize(eng, x, howtorand='statdist'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Progressively randomizes the input time series according to some randomization
    % scheme, and returns measures of how the properties of the time series change
    % with this process.
    % 
    % The procedure is repeated 2N times, where N is the length of the time series.
    % 
    %---INPUTS:
    % y, the input (z-scored) time series
    % howtorand, specifies the randomization scheme for each iteration:
    %      (i) 'statdist' -- substitutes a random element of the time series with
    %                           one from the original time-series distribution
    %      (ii) 'dyndist' -- overwrites a random element of the time
    %                       series with another random element
    %      (iii) 'permute' -- permutes pairs of elements of the time
    %                       series randomly
    % 
    %---OUTPUTS: summarize how the properties change as one of these
    % randomization procedures is iterated, including the cross correlation with the
    % original time series, the autocorrelation of the randomized time series, its
    % entropy, and stationarity.
    % 
    % These statistics are calculated every N/10 iterations, and thus 20 times
    % throughout the process in total.
    % 
    % Most statistics measure how these properties decay with randomization, by
    % fitting a function f(x) = Aexp(Bx).
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1diff',
                                                      'ac1fexpa',
                                                      'ac1fexpadjr2',
                                                      'ac1fexpb',
                                                      'ac1fexpr2',
                                                      'ac1fexprmse',
                                                      'ac1hp',
                                                      'ac2diff',
                                                      'ac2fexpa',
                                                      'ac2fexpadjr2',
                                                      'ac2fexpb',
                                                      'ac2fexpr2',
                                                      'ac2fexprmse',
                                                      'ac2hp',
                                                      'ac3diff',
                                                      'ac3fexpa',
                                                      'ac3fexpadjr2',
                                                      'ac3fexpb',
                                                      'ac3fexpr2',
                                                      'ac3fexprmse',
                                                      'ac3hp',
                                                      'ac4diff',
                                                      'ac4fexpa',
                                                      'ac4fexpadjr2',
                                                      'ac4fexpb',
                                                      'ac4fexpr2',
                                                      'ac4fexprmse',
                                                      'ac4hp',
                                                      'd1diff',
                                                      'd1fexpa',
                                                      'd1fexpadjr2',
                                                      'd1fexpb',
                                                      'd1fexpc',
                                                      'd1fexpr2',
                                                      'd1fexprmse',
                                                      'd1hp',
                                                      'sampen2_02diff',
                                                      'sampen2_02fexpa',
                                                      'sampen2_02fexpadjr2',
                                                      'sampen2_02fexpb',
                                                      'sampen2_02fexpc',
                                                      'sampen2_02fexpr2',
                                                      'sampen2_02fexprmse',
                                                      'sampen2_02hp',
                                                      'shendiff',
                                                      'shenfexpa',
                                                      'shenfexpadjr2',
                                                      'shenfexpb',
                                                      'shenfexpc',
                                                      'shenfexpr2',
                                                      'shenfexprmse',
                                                      'shenhp',
                                                      'statav5diff',
                                                      'statav5fexpa',
                                                      'statav5fexpadjr2',
                                                      'statav5fexpb',
                                                      'statav5fexpc',
                                                      'statav5fexpr2',
                                                      'statav5fexprmse',
                                                      'statav5hp',
                                                      'swss5_1diff',
                                                      'swss5_1fexpa',
                                                      'swss5_1fexpadjr2',
                                                      'swss5_1fexpb',
                                                      'swss5_1fexpc',
                                                      'swss5_1fexpr2',
                                                      'swss5_1fexprmse',
                                                      'swss5_1hp',
                                                      'xc1diff',
                                                      'xc1fexpa',
                                                      'xc1fexpadjr2',
                                                      'xc1fexpb',
                                                      'xc1fexpr2',
                                                      'xc1fexprmse',
                                                      'xc1hp',
                                                      'xcn1diff',
                                                      'xcn1fexpa',
                                                      'xcn1fexpadjr2',
                                                      'xcn1fexpb',
                                                      'xcn1fexpr2',
                                                      'xcn1fexprmse',
                                                      'xcn1hp']}
    if howtorand is None:
        out = eng.run_function(1, 'EN_Randomize', x, )
    else:
        out = eng.run_function(1, 'EN_Randomize', x, howtorand)
    return outfunc(out)


class EN_Randomize(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Progressively randomizes the input time series according to some randomization
    % scheme, and returns measures of how the properties of the time series change
    % with this process.
    % 
    % The procedure is repeated 2N times, where N is the length of the time series.
    % 
    %---INPUTS:
    % y, the input (z-scored) time series
    % howtorand, specifies the randomization scheme for each iteration:
    %      (i) 'statdist' -- substitutes a random element of the time series with
    %                           one from the original time-series distribution
    %      (ii) 'dyndist' -- overwrites a random element of the time
    %                       series with another random element
    %      (iii) 'permute' -- permutes pairs of elements of the time
    %                       series randomly
    % 
    %---OUTPUTS: summarize how the properties change as one of these
    % randomization procedures is iterated, including the cross correlation with the
    % original time series, the autocorrelation of the randomized time series, its
    % entropy, and stationarity.
    % 
    % These statistics are calculated every N/10 iterations, and thus 20 times
    % throughout the process in total.
    % 
    % Most statistics measure how these properties decay with randomization, by
    % fitting a function f(x) = Aexp(Bx).
    % 
    ----------------------------------------
    """

    outnames = ('ac1diff',
                'ac1fexpa',
                'ac1fexpadjr2',
                'ac1fexpb',
                'ac1fexpr2',
                'ac1fexprmse',
                'ac1hp',
                'ac2diff',
                'ac2fexpa',
                'ac2fexpadjr2',
                'ac2fexpb',
                'ac2fexpr2',
                'ac2fexprmse',
                'ac2hp',
                'ac3diff',
                'ac3fexpa',
                'ac3fexpadjr2',
                'ac3fexpb',
                'ac3fexpr2',
                'ac3fexprmse',
                'ac3hp',
                'ac4diff',
                'ac4fexpa',
                'ac4fexpadjr2',
                'ac4fexpb',
                'ac4fexpr2',
                'ac4fexprmse',
                'ac4hp',
                'd1diff',
                'd1fexpa',
                'd1fexpadjr2',
                'd1fexpb',
                'd1fexpc',
                'd1fexpr2',
                'd1fexprmse',
                'd1hp',
                'sampen2_02diff',
                'sampen2_02fexpa',
                'sampen2_02fexpadjr2',
                'sampen2_02fexpb',
                'sampen2_02fexpc',
                'sampen2_02fexpr2',
                'sampen2_02fexprmse',
                'sampen2_02hp',
                'shendiff',
                'shenfexpa',
                'shenfexpadjr2',
                'shenfexpb',
                'shenfexpc',
                'shenfexpr2',
                'shenfexprmse',
                'shenhp',
                'statav5diff',
                'statav5fexpa',
                'statav5fexpadjr2',
                'statav5fexpb',
                'statav5fexpc',
                'statav5fexpr2',
                'statav5fexprmse',
                'statav5hp',
                'swss5_1diff',
                'swss5_1fexpa',
                'swss5_1fexpadjr2',
                'swss5_1fexpb',
                'swss5_1fexpc',
                'swss5_1fexpr2',
                'swss5_1fexprmse',
                'swss5_1hp',
                'xc1diff',
                'xc1fexpa',
                'xc1fexpadjr2',
                'xc1fexpb',
                'xc1fexpr2',
                'xc1fexprmse',
                'xc1hp',
                'xcn1diff',
                'xcn1fexpa',
                'xcn1fexpadjr2',
                'xcn1fexpb',
                'xcn1fexpr2',
                'xcn1fexprmse',
                'xcn1hp')

    def __init__(self, howtorand='statdist'):
        super(EN_Randomize, self).__init__(add_descriptors=False)
        self.howtorand = howtorand

    def eval(self, engine, x):
        return HCTSA_EN_Randomize(engine,
                                  x,
                                  howtorand=self.howtorand)


def HCTSA_EN_SampEn(eng, x, M=4, r=0.1, preprocess=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the Sample Entropy of the time series, SampEn(m,r), by referencing
    % code from PhysioNet.
    % 
    % The publicly-available PhysioNet code, sampenc (renamed here to RN_sampenc) is
    % available from:
    % http://www.physionet.org/physiotools/sampen/matlab/1.1/sampenc.m
    %
    % cf. "Physiological time-series analysis using approximate entropy and sample
    % entropy", J. S. Richman and J. R. Moorman, Am. J. Physiol. Heart Circ.
    % Physiol., 278(6) H2039 (2000)
    % 
    % This function can also calculate the SampEn of successive increments of time
    % series, i.e., we using an incremental differencing pre-processing, as
    % used in the so-called Control Entropy quantity:
    % 
    % "Control Entropy: A complexity measure for nonstationary signals"
    % E. M. Bollt and J. Skufca, Math. Biosci. Eng., 6(1) 1 (2009)
    % 
    %---INPUTS:
    % y, the input time series
    % M, the embedding dimension
    % r, the threshold
    % preprocess [opt], (i) 'diff1', incremental difference preprocessing.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['meanchp',
                                                      'meanchsampen',
                                                      'p1',
                                                      'p2',
                                                      'p3',
                                                      'p4',
                                                      'sampen1',
                                                      'sampen2',
                                                      'sampen3',
                                                      'sampen4']}
    if M is None:
        out = eng.run_function(1, 'EN_SampEn', x, )
    elif r is None:
        out = eng.run_function(1, 'EN_SampEn', x, M)
    elif preprocess is None:
        out = eng.run_function(1, 'EN_SampEn', x, M, r)
    else:
        out = eng.run_function(1, 'EN_SampEn', x, M, r, preprocess)
    return outfunc(out)


class EN_SampEn(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the Sample Entropy of the time series, SampEn(m,r), by referencing
    % code from PhysioNet.
    % 
    % The publicly-available PhysioNet code, sampenc (renamed here to RN_sampenc) is
    % available from:
    % http://www.physionet.org/physiotools/sampen/matlab/1.1/sampenc.m
    %
    % cf. "Physiological time-series analysis using approximate entropy and sample
    % entropy", J. S. Richman and J. R. Moorman, Am. J. Physiol. Heart Circ.
    % Physiol., 278(6) H2039 (2000)
    % 
    % This function can also calculate the SampEn of successive increments of time
    % series, i.e., we using an incremental differencing pre-processing, as
    % used in the so-called Control Entropy quantity:
    % 
    % "Control Entropy: A complexity measure for nonstationary signals"
    % E. M. Bollt and J. Skufca, Math. Biosci. Eng., 6(1) 1 (2009)
    % 
    %---INPUTS:
    % y, the input time series
    % M, the embedding dimension
    % r, the threshold
    % preprocess [opt], (i) 'diff1', incremental difference preprocessing.
    % 
    ----------------------------------------
    """

    outnames = ('meanchp',
                'meanchsampen',
                'p1',
                'p2',
                'p3',
                'p4',
                'sampen1',
                'sampen2',
                'sampen3',
                'sampen4')

    def __init__(self, M=4, r=0.1, preprocess=None):
        super(EN_SampEn, self).__init__(add_descriptors=False)
        self.M = M
        self.r = r
        self.preprocess = preprocess

    def eval(self, engine, x):
        return HCTSA_EN_SampEn(engine,
                               x,
                               M=self.M,
                               r=self.r,
                               preprocess=self.preprocess)


def HCTSA_EN_Shannonpdf(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the (log_2) Shannon entropy from the probability distribution of the time
    % series.
    % 
    %---INPUT:
    % y, the input time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'EN_Shannonpdf', x, )
    return outfunc(out)


class EN_Shannonpdf(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the (log_2) Shannon entropy from the probability distribution of the time
    % series.
    % 
    %---INPUT:
    % y, the input time series
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(EN_Shannonpdf, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_EN_Shannonpdf(engine, x)


def HCTSA_EN_TSentropy(eng, x, q=0.6):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the Tsallis entropy of a signal using a parameter q that
    % measures the non-extensivity of the system; q = 1 recovers the Shannon
    % entropy.
    % 
    %---INPUTS:
    % x, the time series
    % q, the non-extensivity parameter
    % 
    % Uses code written by D. Tolstonogov and obtained from
    % http://download.tsresearchgroup.com/all/tsmatlablink/TSentropy.m.
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if q is None:
        out = eng.run_function(1, 'EN_TSentropy', x, )
    else:
        out = eng.run_function(1, 'EN_TSentropy', x, q)
    return outfunc(out)


class EN_TSentropy(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the Tsallis entropy of a signal using a parameter q that
    % measures the non-extensivity of the system; q = 1 recovers the Shannon
    % entropy.
    % 
    %---INPUTS:
    % x, the time series
    % q, the non-extensivity parameter
    % 
    % Uses code written by D. Tolstonogov and obtained from
    % http://download.tsresearchgroup.com/all/tsmatlablink/TSentropy.m.
    %
    ----------------------------------------
    """

    def __init__(self, q=0.6):
        super(EN_TSentropy, self).__init__(add_descriptors=False)
        self.q = q

    def eval(self, engine, x):
        return HCTSA_EN_TSentropy(engine,
                                  x,
                                  q=self.q)


def HCTSA_EN_wentropy(eng, x, whaten='threshold', p=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the wentropy function from Matlab's Wavelet toolbox to
    % estimate the entropy of the input time series.
    % 
    %--INPUTS:
    % y, the input time series
    % whaten, the entropy type:
    %               'shannon',
    %               'logenergy',
    %               'threshold' (with a given threshold),
    %               'sure' (with a given parameter).
    %               (see the wentropy documentaiton for information)
    % p, the additional parameter needed for threshold and sure entropies
    % 
    % Author's cautionary note: it seems likely that this implementation of wentropy
    % is nonsense.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if whaten is None:
        out = eng.run_function(1, 'EN_wentropy', x, )
    elif p is None:
        out = eng.run_function(1, 'EN_wentropy', x, whaten)
    else:
        out = eng.run_function(1, 'EN_wentropy', x, whaten, p)
    return outfunc(out)


class EN_wentropy(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the wentropy function from Matlab's Wavelet toolbox to
    % estimate the entropy of the input time series.
    % 
    %--INPUTS:
    % y, the input time series
    % whaten, the entropy type:
    %               'shannon',
    %               'logenergy',
    %               'threshold' (with a given threshold),
    %               'sure' (with a given parameter).
    %               (see the wentropy documentaiton for information)
    % p, the additional parameter needed for threshold and sure entropies
    % 
    % Author's cautionary note: it seems likely that this implementation of wentropy
    % is nonsense.
    % 
    ----------------------------------------
    """

    def __init__(self, whaten='threshold', p=1):
        super(EN_wentropy, self).__init__(add_descriptors=False)
        self.whaten = whaten
        self.p = p

    def eval(self, engine, x):
        return HCTSA_EN_wentropy(engine,
                                 x,
                                 whaten=self.whaten,
                                 p=self.p)


def HCTSA_EX_MovingThreshold(eng, x, a=1, b=0.02):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % A measure based on a moving threshold model for extreme events. Inspired by an
    % idea contained in the following paper:
    % "Reactions to extreme events: Moving threshold model"
    % Altmann et al., Physica A 364, 435--444 (2006)
    % 
    % This algorithm is based on this idea: it uses the occurrence of extreme events
    % to modify a hypothetical 'barrier' that classes new points as 'extreme' or not.
    % The barrier begins at sigma, and if the absolute value of the next data point
    % is greater than the barrier, the barrier is increased by a proportion 'a',
    % otherwise the position of the barrier is decreased by a proportion 'b'.
    % 
    %---INPUTS:
    % y, the input (z-scored) time series
    % a, the barrier jump parameter (in extreme event)
    % b, the barrier decay proportion (in absence of extreme event)
    % 
    %---OUTPUTS: the mean, spread, maximum, and minimum of the time series for the
    % barrier, the mean of the difference between the barrier and the time series
    % values, and statistics on the occurrence of 'kicks' (times at which the
    % threshold is modified), and by how much the threshold changes on average.
    % 
    % In future could make a variant operation that optimizes a and b to minimize the
    % quantity meanqover/pkick (hugged the shape as close as possible with the
    % minimum number of kicks), and returns a and b...?
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['iqrq',
                                                      'maxq',
                                                      'meankickf',
                                                      'meanq',
                                                      'meanqover',
                                                      'mediankickf',
                                                      'medianq',
                                                      'minq',
                                                      'pkick',
                                                      'stdkickf',
                                                      'stdq']}
    if a is None:
        out = eng.run_function(1, 'EX_MovingThreshold', x, )
    elif b is None:
        out = eng.run_function(1, 'EX_MovingThreshold', x, a)
    else:
        out = eng.run_function(1, 'EX_MovingThreshold', x, a, b)
    return outfunc(out)


class EX_MovingThreshold(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % A measure based on a moving threshold model for extreme events. Inspired by an
    % idea contained in the following paper:
    % "Reactions to extreme events: Moving threshold model"
    % Altmann et al., Physica A 364, 435--444 (2006)
    % 
    % This algorithm is based on this idea: it uses the occurrence of extreme events
    % to modify a hypothetical 'barrier' that classes new points as 'extreme' or not.
    % The barrier begins at sigma, and if the absolute value of the next data point
    % is greater than the barrier, the barrier is increased by a proportion 'a',
    % otherwise the position of the barrier is decreased by a proportion 'b'.
    % 
    %---INPUTS:
    % y, the input (z-scored) time series
    % a, the barrier jump parameter (in extreme event)
    % b, the barrier decay proportion (in absence of extreme event)
    % 
    %---OUTPUTS: the mean, spread, maximum, and minimum of the time series for the
    % barrier, the mean of the difference between the barrier and the time series
    % values, and statistics on the occurrence of 'kicks' (times at which the
    % threshold is modified), and by how much the threshold changes on average.
    % 
    % In future could make a variant operation that optimizes a and b to minimize the
    % quantity meanqover/pkick (hugged the shape as close as possible with the
    % minimum number of kicks), and returns a and b...?
    % 
    ----------------------------------------
    """

    outnames = ('iqrq',
                'maxq',
                'meankickf',
                'meanq',
                'meanqover',
                'mediankickf',
                'medianq',
                'minq',
                'pkick',
                'stdkickf',
                'stdq')

    def __init__(self, a=1, b=0.02):
        super(EX_MovingThreshold, self).__init__(add_descriptors=False)
        self.a = a
        self.b = b

    def eval(self, engine, x):
        return HCTSA_EX_MovingThreshold(engine,
                                        x,
                                        a=self.a,
                                        b=self.b)


def HCTSA_FC_LocalSimple(eng, x, fmeth='mean', ltrain=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Does local forecasting using very simple predictors using the past l values
    % of the time series to predict its next value.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % fmeth, the forecasting method:
    %          (i) 'mean': local mean prediction using the past ltrain time-series
    %                       values,
    %          (ii) 'median': local median prediction using the past ltrain
    %                         time-series values
    %          (iii) 'lfit': local linear prediction using the past ltrain
    %                         time-series values.
    % 
    % ltrain, the number of time-series values to use to forecast the next value
    % 
    %---OUTPUTS: the mean error, stationarity of residuals, Gaussianity of
    % residuals, and their autocorrelation structure.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac2',
                                                      'gofnadjr2',
                                                      'meanabserr',
                                                      'meanerr',
                                                      'rmserr',
                                                      'stderr',
                                                      'swm',
                                                      'sws',
                                                      'taures',
                                                      'tauresrat']}
    if fmeth is None:
        out = eng.run_function(1, 'FC_LocalSimple', x, )
    elif ltrain is None:
        out = eng.run_function(1, 'FC_LocalSimple', x, fmeth)
    else:
        out = eng.run_function(1, 'FC_LocalSimple', x, fmeth, ltrain)
    return outfunc(out)


class FC_LocalSimple(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Does local forecasting using very simple predictors using the past l values
    % of the time series to predict its next value.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % fmeth, the forecasting method:
    %          (i) 'mean': local mean prediction using the past ltrain time-series
    %                       values,
    %          (ii) 'median': local median prediction using the past ltrain
    %                         time-series values
    %          (iii) 'lfit': local linear prediction using the past ltrain
    %                         time-series values.
    % 
    % ltrain, the number of time-series values to use to forecast the next value
    % 
    %---OUTPUTS: the mean error, stationarity of residuals, Gaussianity of
    % residuals, and their autocorrelation structure.
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac2',
                'gofnadjr2',
                'meanabserr',
                'meanerr',
                'rmserr',
                'stderr',
                'swm',
                'sws',
                'taures',
                'tauresrat')

    def __init__(self, fmeth='mean', ltrain=3):
        super(FC_LocalSimple, self).__init__(add_descriptors=False)
        self.fmeth = fmeth
        self.ltrain = ltrain

    def eval(self, engine, x):
        return HCTSA_FC_LocalSimple(engine,
                                    x,
                                    fmeth=self.fmeth,
                                    ltrain=self.ltrain)


def HCTSA_FC_LoopLocalSimple(eng, x, fmeth='mean'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes the outputs of FC_LocalSimple for a range of local window lengths, l.
    % Loops over the length of the data to use for FC_LocalSimple prediction
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % fmeth, the prediction method:
    %            (i) 'mean', local mean prediction
    %            (ii) 'median', local median prediction
    % 
    %---OUTPUTS: statistics including whether the mean square error increases or
    % decreases, testing for peaks, variability, autocorrelation, stationarity, and
    % a fit of exponential decay, f(x) = A*exp(Bx) + C, to the variation.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1_chn',
                                                      'ac1_meansgndiff',
                                                      'ac1_stdn',
                                                      'ac2_chn',
                                                      'ac2_meansgndiff',
                                                      'ac2_stdn',
                                                      'rmserr_chn',
                                                      'rmserr_meansgndiff',
                                                      'rmserr_peakpos',
                                                      'rmserr_peaksize',
                                                      'swm_chn',
                                                      'swm_meansgndiff',
                                                      'swm_stdn',
                                                      'sws_chn',
                                                      'sws_fexp_a',
                                                      'sws_fexp_adjr2',
                                                      'sws_fexp_b',
                                                      'sws_fexp_c',
                                                      'sws_fexp_rmse',
                                                      'sws_meansgndiff',
                                                      'sws_stdn',
                                                      'xcorrstdrmserr']}
    if fmeth is None:
        out = eng.run_function(1, 'FC_LoopLocalSimple', x, )
    else:
        out = eng.run_function(1, 'FC_LoopLocalSimple', x, fmeth)
    return outfunc(out)


class FC_LoopLocalSimple(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes the outputs of FC_LocalSimple for a range of local window lengths, l.
    % Loops over the length of the data to use for FC_LocalSimple prediction
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % fmeth, the prediction method:
    %            (i) 'mean', local mean prediction
    %            (ii) 'median', local median prediction
    % 
    %---OUTPUTS: statistics including whether the mean square error increases or
    % decreases, testing for peaks, variability, autocorrelation, stationarity, and
    % a fit of exponential decay, f(x) = A*exp(Bx) + C, to the variation.
    % 
    ----------------------------------------
    """

    outnames = ('ac1_chn',
                'ac1_meansgndiff',
                'ac1_stdn',
                'ac2_chn',
                'ac2_meansgndiff',
                'ac2_stdn',
                'rmserr_chn',
                'rmserr_meansgndiff',
                'rmserr_peakpos',
                'rmserr_peaksize',
                'swm_chn',
                'swm_meansgndiff',
                'swm_stdn',
                'sws_chn',
                'sws_fexp_a',
                'sws_fexp_adjr2',
                'sws_fexp_b',
                'sws_fexp_c',
                'sws_fexp_rmse',
                'sws_meansgndiff',
                'sws_stdn',
                'xcorrstdrmserr')

    def __init__(self, fmeth='mean'):
        super(FC_LoopLocalSimple, self).__init__(add_descriptors=False)
        self.fmeth = fmeth

    def eval(self, engine, x):
        return HCTSA_FC_LoopLocalSimple(engine,
                                        x,
                                        fmeth=self.fmeth)


def HCTSA_FC_Surprise(eng, x, whatinf='dist', memory=50, ng=3, cgmeth='quantile', nits=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % How surprised you might be by the next recorded data points given the data
    % recorded in recent memory.
    % 
    % Coarse-grains the time series, turning it into a sequence of symbols of a
    % given alphabet size, ng, and quantifies measures of surprise of a
    % process with local memory of the past memory values of the symbolic string.
    % 
    % We then consider a memory length, memory, of the time series, and
    % use the data in the proceeding memory samples to inform our expectations of
    % the following sample.
    % 
    % The 'information gained', log(1/p), at each sample using expectations
    % calculated from the previous memory samples, is estimated.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % whatinf, the type of information to store in memory:
    %           (i) 'dist': the values of the time series in the previous memory
    %                       samples,
    %           (ii) 'T1': the one-point transition probabilities in the previous
    %                       memory samples, and
    %           (iii) 'T2': the two-point transition probabilities in the previous
    %                       memory samples.
    %                       
    % memory, the memory length (either number of samples, or a proportion of the
    %           time-series length, if between 0 and 1)
    %           
    % ng, the number of groups to coarse-grain the time series into
    % 
    % cgmeth, the coarse-graining, or symbolization method:
    %          (i) 'quantile': an equiprobable alphabet by the value of each
    %                          time-series datapoint,
    %          (ii) 'updown': an equiprobable alphabet by the value of incremental
    %                         changes in the time-series values, and
    %          (iii) 'embed2quadrants': by the quadrant each data point resides in
    %                          in a two-dimensional embedding space.
    % 
    % nits, the number of iterations to repeat the procedure for.
    % 
    %---OUTPUTS: summaries of this series of information gains, including the
    %            minimum, maximum, mean, median, lower and upper quartiles, and
    %            standard deviation.
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['lq',
                                                      'max',
                                                      'mean',
                                                      'median',
                                                      'min',
                                                      'std',
                                                      'tstat',
                                                      'uq']}
    if whatinf is None:
        out = eng.run_function(1, 'FC_Surprise', x, )
    elif memory is None:
        out = eng.run_function(1, 'FC_Surprise', x, whatinf)
    elif ng is None:
        out = eng.run_function(1, 'FC_Surprise', x, whatinf, memory)
    elif cgmeth is None:
        out = eng.run_function(1, 'FC_Surprise', x, whatinf, memory, ng)
    elif nits is None:
        out = eng.run_function(1, 'FC_Surprise', x, whatinf, memory, ng, cgmeth)
    else:
        out = eng.run_function(1, 'FC_Surprise', x, whatinf, memory, ng, cgmeth, nits)
    return outfunc(out)


class FC_Surprise(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % How surprised you might be by the next recorded data points given the data
    % recorded in recent memory.
    % 
    % Coarse-grains the time series, turning it into a sequence of symbols of a
    % given alphabet size, ng, and quantifies measures of surprise of a
    % process with local memory of the past memory values of the symbolic string.
    % 
    % We then consider a memory length, memory, of the time series, and
    % use the data in the proceeding memory samples to inform our expectations of
    % the following sample.
    % 
    % The 'information gained', log(1/p), at each sample using expectations
    % calculated from the previous memory samples, is estimated.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % whatinf, the type of information to store in memory:
    %           (i) 'dist': the values of the time series in the previous memory
    %                       samples,
    %           (ii) 'T1': the one-point transition probabilities in the previous
    %                       memory samples, and
    %           (iii) 'T2': the two-point transition probabilities in the previous
    %                       memory samples.
    %                       
    % memory, the memory length (either number of samples, or a proportion of the
    %           time-series length, if between 0 and 1)
    %           
    % ng, the number of groups to coarse-grain the time series into
    % 
    % cgmeth, the coarse-graining, or symbolization method:
    %          (i) 'quantile': an equiprobable alphabet by the value of each
    %                          time-series datapoint,
    %          (ii) 'updown': an equiprobable alphabet by the value of incremental
    %                         changes in the time-series values, and
    %          (iii) 'embed2quadrants': by the quadrant each data point resides in
    %                          in a two-dimensional embedding space.
    % 
    % nits, the number of iterations to repeat the procedure for.
    % 
    %---OUTPUTS: summaries of this series of information gains, including the
    %            minimum, maximum, mean, median, lower and upper quartiles, and
    %            standard deviation.
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """

    outnames = ('lq',
                'max',
                'mean',
                'median',
                'min',
                'std',
                'tstat',
                'uq')

    def __init__(self, whatinf='dist', memory=50, ng=3, cgmeth='quantile', nits=None):
        super(FC_Surprise, self).__init__(add_descriptors=False)
        self.whatinf = whatinf
        self.memory = memory
        self.ng = ng
        self.cgmeth = cgmeth
        self.nits = nits

    def eval(self, engine, x):
        return HCTSA_FC_Surprise(engine,
                                 x,
                                 whatinf=self.whatinf,
                                 memory=self.memory,
                                 ng=self.ng,
                                 cgmeth=self.cgmeth,
                                 nits=self.nits)


def HCTSA_HT_DistributionTest(eng, x, thetest='ks', thedistn='beta', nbins=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a distribution to the data and then performs an appropriate hypothesis
    % test to quantify the difference between the two distributions. We fit
    % Gaussian, Extreme Value, Uniform, Beta, Rayleigh, Exponential, Gamma,
    % Log-Normal, and Weibull distributions, using code described for
    % DN_M_kscomp.
    % 
    %---INPUTS:
    % x, the input time series
    % thetest, the hypothesis test to perform:
    %           (i) 'chi2gof': chi^2 goodness of fit test
    %           (ii) 'ks': Kolmogorov-Smirnov test
    %           (iii) 'lillie': Lilliefors test
    % 
    % thedistn, the distribution to fit:
    %           (i) 'norm' (Normal)
    %           (ii) 'ev' (Extreme value)
    %           (iii) 'uni' (Uniform)
    %           (iv) 'beta' (Beta)
    %           (v) 'rayleigh' (Rayleigh)
    %           (vi) 'exp' (Exponential)
    %           (vii) 'gamma' (Gamma)
    %           (viii) 'logn' (Log-normal)
    %           (ix) 'wbl' (Weibull)
    % 
    % nbins, the number of bins to use for the chi2 goodness of fit test
    % 
    % All of these functions for hypothesis testing are implemented in Matlab's
    % Statistics Toolbox.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if thetest is None:
        out = eng.run_function(1, 'HT_DistributionTest', x, )
    elif thedistn is None:
        out = eng.run_function(1, 'HT_DistributionTest', x, thetest)
    elif nbins is None:
        out = eng.run_function(1, 'HT_DistributionTest', x, thetest, thedistn)
    else:
        out = eng.run_function(1, 'HT_DistributionTest', x, thetest, thedistn, nbins)
    return outfunc(out)


class HT_DistributionTest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a distribution to the data and then performs an appropriate hypothesis
    % test to quantify the difference between the two distributions. We fit
    % Gaussian, Extreme Value, Uniform, Beta, Rayleigh, Exponential, Gamma,
    % Log-Normal, and Weibull distributions, using code described for
    % DN_M_kscomp.
    % 
    %---INPUTS:
    % x, the input time series
    % thetest, the hypothesis test to perform:
    %           (i) 'chi2gof': chi^2 goodness of fit test
    %           (ii) 'ks': Kolmogorov-Smirnov test
    %           (iii) 'lillie': Lilliefors test
    % 
    % thedistn, the distribution to fit:
    %           (i) 'norm' (Normal)
    %           (ii) 'ev' (Extreme value)
    %           (iii) 'uni' (Uniform)
    %           (iv) 'beta' (Beta)
    %           (v) 'rayleigh' (Rayleigh)
    %           (vi) 'exp' (Exponential)
    %           (vii) 'gamma' (Gamma)
    %           (viii) 'logn' (Log-normal)
    %           (ix) 'wbl' (Weibull)
    % 
    % nbins, the number of bins to use for the chi2 goodness of fit test
    % 
    % All of these functions for hypothesis testing are implemented in Matlab's
    % Statistics Toolbox.
    % 
    ----------------------------------------
    """

    def __init__(self, thetest='ks', thedistn='beta', nbins=None):
        super(HT_DistributionTest, self).__init__(add_descriptors=False)
        self.thetest = thetest
        self.thedistn = thedistn
        self.nbins = nbins

    def eval(self, engine, x):
        return HCTSA_HT_DistributionTest(engine,
                                         x,
                                         thetest=self.thetest,
                                         thedistn=self.thedistn,
                                         nbins=self.nbins)


def HCTSA_HT_HypothesisTest(eng, x, thetest='runstest'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Outputs the p-value from a statistical hypothesis test applied to the
    % time series.
    % 
    % Tests are implemented as functions in Matlab's Statistics Toolbox.
    % (except Ljung-Box Q-test, which uses the Econometrics Toolbox)
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % thetest, the hypothesis test to perform:
    %           (i) sign test ('signtest'),
    %           (ii) runs test ('runstest'),
    %           (iii) variance test ('vartest'),
    %           (iv) Z-test ('ztest'),
    %           (v) Wilcoxon signed rank test for a zero median ('signrank'),
    %           (vi) Jarque-Bera test of composite normality ('jbtest').
    %           (vii) Ljung-Box Q-test for residual autocorrelation ('lbq')
    %           
    %---OUTPUT:
    % the p-value from the statistical test
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if thetest is None:
        out = eng.run_function(1, 'HT_HypothesisTest', x, )
    else:
        out = eng.run_function(1, 'HT_HypothesisTest', x, thetest)
    return outfunc(out)


class HT_HypothesisTest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Outputs the p-value from a statistical hypothesis test applied to the
    % time series.
    % 
    % Tests are implemented as functions in Matlab's Statistics Toolbox.
    % (except Ljung-Box Q-test, which uses the Econometrics Toolbox)
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % thetest, the hypothesis test to perform:
    %           (i) sign test ('signtest'),
    %           (ii) runs test ('runstest'),
    %           (iii) variance test ('vartest'),
    %           (iv) Z-test ('ztest'),
    %           (v) Wilcoxon signed rank test for a zero median ('signrank'),
    %           (vi) Jarque-Bera test of composite normality ('jbtest').
    %           (vii) Ljung-Box Q-test for residual autocorrelation ('lbq')
    %           
    %---OUTPUT:
    % the p-value from the statistical test
    % 
    ----------------------------------------
    """

    def __init__(self, thetest='runstest'):
        super(HT_HypothesisTest, self).__init__(add_descriptors=False)
        self.thetest = thetest

    def eval(self, engine, x):
        return HCTSA_HT_HypothesisTest(engine,
                                       x,
                                       thetest=self.thetest)


def HCTSA_MD_hrv_classic(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Packages up a bunch of classic heart rate variability (HRV) statistics and
    % applies them to the input time series.
    % 
    % Assumes an NN/RR time series in units of seconds.
    % 
    %---INPUTS:
    % y, the input time series.
    % 
    % Includes:
    %  (i) pNNx
    %  cf. "The pNNx files: re-examining a widely used heart rate variability
    %           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)
    % 
    %  (ii) Power spectral density ratios in different frequency ranges
    %   cf. "Heart rate variability: Standards of measurement, physiological
    %       interpretation, and clinical use",
    %       M. Malik et al., Eur. Heart J. 17(3) 354 (1996)
    % 
    %  (iii) Triangular histogram index, and
    %  
    %  (iv) Poincare plot measures
    %  cf. "Do existing measures of Poincare plot geometry reflect nonlinear
    %       features of heart rate variability?"
    %       M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)
    %  
    % Code is heavily derived from that provided by Max A. Little:
    % http://www.maxlittle.net/
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['SD1',
                                                      'SD2',
                                                      'hf',
                                                      'lf',
                                                      'lfhf',
                                                      'pnn10',
                                                      'pnn20',
                                                      'pnn30',
                                                      'pnn40',
                                                      'pnn5',
                                                      'tri',
                                                      'vlf']}
    out = eng.run_function(1, 'MD_hrv_classic', x, )
    return outfunc(out)


class MD_hrv_classic(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Packages up a bunch of classic heart rate variability (HRV) statistics and
    % applies them to the input time series.
    % 
    % Assumes an NN/RR time series in units of seconds.
    % 
    %---INPUTS:
    % y, the input time series.
    % 
    % Includes:
    %  (i) pNNx
    %  cf. "The pNNx files: re-examining a widely used heart rate variability
    %           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)
    % 
    %  (ii) Power spectral density ratios in different frequency ranges
    %   cf. "Heart rate variability: Standards of measurement, physiological
    %       interpretation, and clinical use",
    %       M. Malik et al., Eur. Heart J. 17(3) 354 (1996)
    % 
    %  (iii) Triangular histogram index, and
    %  
    %  (iv) Poincare plot measures
    %  cf. "Do existing measures of Poincare plot geometry reflect nonlinear
    %       features of heart rate variability?"
    %       M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)
    %  
    % Code is heavily derived from that provided by Max A. Little:
    % http://www.maxlittle.net/
    % 
    ----------------------------------------
    """

    outnames = ('SD1',
                'SD2',
                'hf',
                'lf',
                'lfhf',
                'pnn10',
                'pnn20',
                'pnn30',
                'pnn40',
                'pnn5',
                'tri',
                'vlf')

    def __init__(self, ):
        super(MD_hrv_classic, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_MD_hrv_classic(engine, x)


def HCTSA_MD_pNN(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Applies pNNx measures to time series assumed to represent sequences of
    % consecutive RR intervals measured in milliseconds.
    % 
    % cf. "The pNNx files: re-examining a widely used heart rate variability
    %           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % This code is derived from MD_hrv_classic.m becuase it doesn't make medical
    % sense to do PNN on a z-scored time series.
    % 
    % But now PSD doesn't make too much sense, so we just evaluate the pNN measures.
    % 
    % Code is heavily derived from that provided by Max A. Little:
    % http://www.maxlittle.net/
    %
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['pnn10',
                                                      'pnn100',
                                                      'pnn20',
                                                      'pnn30',
                                                      'pnn40',
                                                      'pnn5',
                                                      'pnn50',
                                                      'pnn60',
                                                      'pnn70',
                                                      'pnn80',
                                                      'pnn90']}
    out = eng.run_function(1, 'MD_pNN', x, )
    return outfunc(out)


class MD_pNN(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Applies pNNx measures to time series assumed to represent sequences of
    % consecutive RR intervals measured in milliseconds.
    % 
    % cf. "The pNNx files: re-examining a widely used heart rate variability
    %           measure", J.E. Mietus et al., Heart 88(4) 378 (2002)
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % This code is derived from MD_hrv_classic.m becuase it doesn't make medical
    % sense to do PNN on a z-scored time series.
    % 
    % But now PSD doesn't make too much sense, so we just evaluate the pNN measures.
    % 
    % Code is heavily derived from that provided by Max A. Little:
    % http://www.maxlittle.net/
    %
    ----------------------------------------
    """

    outnames = ('pnn10',
                'pnn100',
                'pnn20',
                'pnn30',
                'pnn40',
                'pnn5',
                'pnn50',
                'pnn60',
                'pnn70',
                'pnn80',
                'pnn90')

    def __init__(self, ):
        super(MD_pNN, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_MD_pNN(engine, x)


def HCTSA_MD_polvar(eng, x, d=0.1, D=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the POLVARd measure for a time series.
    % The first mention may be in Wessel et al., PRE (2000), called Plvar
    % cf. "Short-term forecasting of life-threatening cardiac arrhythmias based on
    % symbolic dynamics and finite-time growth rates",
    %       N. Wessel et al., Phys. Rev. E 61(1) 733 (2000)
    % 
    % The output from this function is the probability of obtaining a sequence of
    % consecutive ones or zeros.
    % 
    % Although the original measure used raw thresholds, d, on RR interval sequences
    % (measured in milliseconds), this code can be applied to general z-scored time
    % series. So now d is not the time difference in milliseconds, but in units of
    % std.
    % 
    % The measure was originally applied to sequences of RR intervals and this code
    % is heavily derived from that provided by Max A. Little, January 2009.
    % cf. http://www.maxlittle.net/
    % 
    %---INPUTS:
    % x, the input time series
    % d, the symbolic coding (amplitude) difference,
    % D, the word length (classically words of length 6).
    % 
    %---OUPUT:
    % p - probability of obtaining a sequence of consecutive ones/zeros
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if d is None:
        out = eng.run_function(1, 'MD_polvar', x, )
    elif D is None:
        out = eng.run_function(1, 'MD_polvar', x, d)
    else:
        out = eng.run_function(1, 'MD_polvar', x, d, D)
    return outfunc(out)


class MD_polvar(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the POLVARd measure for a time series.
    % The first mention may be in Wessel et al., PRE (2000), called Plvar
    % cf. "Short-term forecasting of life-threatening cardiac arrhythmias based on
    % symbolic dynamics and finite-time growth rates",
    %       N. Wessel et al., Phys. Rev. E 61(1) 733 (2000)
    % 
    % The output from this function is the probability of obtaining a sequence of
    % consecutive ones or zeros.
    % 
    % Although the original measure used raw thresholds, d, on RR interval sequences
    % (measured in milliseconds), this code can be applied to general z-scored time
    % series. So now d is not the time difference in milliseconds, but in units of
    % std.
    % 
    % The measure was originally applied to sequences of RR intervals and this code
    % is heavily derived from that provided by Max A. Little, January 2009.
    % cf. http://www.maxlittle.net/
    % 
    %---INPUTS:
    % x, the input time series
    % d, the symbolic coding (amplitude) difference,
    % D, the word length (classically words of length 6).
    % 
    %---OUPUT:
    % p - probability of obtaining a sequence of consecutive ones/zeros
    % 
    ----------------------------------------
    """

    def __init__(self, d=0.1, D=3):
        super(MD_polvar, self).__init__(add_descriptors=False)
        self.d = d
        self.D = D

    def eval(self, engine, x):
        return HCTSA_MD_polvar(engine,
                               x,
                               d=self.d,
                               D=self.D)


def HCTSA_MD_rawHRVmeas(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Evaluates the triangular histogram index and Poincare plot measures to a time
    % series assumed to measure sequences of consecutive RR intervals measured in
    % milliseconds. Doesn't make much sense for other time series
    % 
    % cf. "Do existing measures of Poincare plot geometry reflect nonlinear
    %      features of heart rate variability?"
    %      M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)
    % 
    % Note that pNNx is not done here, but in MD_pNN.m
    % 
    % This code is heavily derived from Max Little's hrv_classic.m code
    % Max Little: http://www.maxlittle.net/
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['SD1',
                                                      'SD2',
                                                      'tri10',
                                                      'tri20',
                                                      'trisqrt']}
    out = eng.run_function(1, 'MD_rawHRVmeas', x, )
    return outfunc(out)


class MD_rawHRVmeas(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Evaluates the triangular histogram index and Poincare plot measures to a time
    % series assumed to measure sequences of consecutive RR intervals measured in
    % milliseconds. Doesn't make much sense for other time series
    % 
    % cf. "Do existing measures of Poincare plot geometry reflect nonlinear
    %      features of heart rate variability?"
    %      M. Brennan, et al., IEEE T. Bio.-Med. Eng. 48(11) 1342 (2001)
    % 
    % Note that pNNx is not done here, but in MD_pNN.m
    % 
    % This code is heavily derived from Max Little's hrv_classic.m code
    % Max Little: http://www.maxlittle.net/
    % 
    ----------------------------------------
    """

    outnames = ('SD1',
                'SD2',
                'tri10',
                'tri20',
                'trisqrt')

    def __init__(self, ):
        super(MD_rawHRVmeas, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_MD_rawHRVmeas(engine, x)


def HCTSA_MF_ARMA_orders(eng, x, pr=MatlabSequence('1:6'), qr=MatlabSequence('1:4')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Given a set of AR orders, p, and a set of MA orders, q, this operation fits
    % ARMA(p,q) models to the time series and evaluates the goodness of fit from all
    % combinations (p,q).
    % 
    % Uses functions iddata, armax, and aic from Matlab's System Identification toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % pr, a vector specifying the range of AR model orders to analyze
    % qr, a vector specifying the range of MA model orders to analyze
    % 
    %---OUTPUTS: statistics on the appropriateness of different types of models,
    % including the goodness of fit from the best model, and the optimal orders of
    % fitted ARMA(p,q) models.
    % 
    % ** Future improvements **
    % (1) May want to quantify where a particular order starts to stand out, i.e.,
    % may be quite sensitive to wanting p > 2, but may be quite indiscriminate
    % when it comes to the MA order.
    % (2) May want to do some prediction and get more statistics on quality of
    % model rather than just in-sample FPE/AIC...
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['aic_min',
                                                      'mean_all_aics',
                                                      'meanstd_aicsp',
                                                      'meanstd_aicsq',
                                                      'p_aic_opt',
                                                      'q_aic_opt',
                                                      'std_all_aics']}
    if pr is None:
        out = eng.run_function(1, 'MF_ARMA_orders', x, )
    elif qr is None:
        out = eng.run_function(1, 'MF_ARMA_orders', x, pr)
    else:
        out = eng.run_function(1, 'MF_ARMA_orders', x, pr, qr)
    return outfunc(out)


class MF_ARMA_orders(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Given a set of AR orders, p, and a set of MA orders, q, this operation fits
    % ARMA(p,q) models to the time series and evaluates the goodness of fit from all
    % combinations (p,q).
    % 
    % Uses functions iddata, armax, and aic from Matlab's System Identification toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % pr, a vector specifying the range of AR model orders to analyze
    % qr, a vector specifying the range of MA model orders to analyze
    % 
    %---OUTPUTS: statistics on the appropriateness of different types of models,
    % including the goodness of fit from the best model, and the optimal orders of
    % fitted ARMA(p,q) models.
    % 
    % ** Future improvements **
    % (1) May want to quantify where a particular order starts to stand out, i.e.,
    % may be quite sensitive to wanting p > 2, but may be quite indiscriminate
    % when it comes to the MA order.
    % (2) May want to do some prediction and get more statistics on quality of
    % model rather than just in-sample FPE/AIC...
    % 
    ----------------------------------------
    """

    outnames = ('aic_min',
                'mean_all_aics',
                'meanstd_aicsp',
                'meanstd_aicsq',
                'p_aic_opt',
                'q_aic_opt',
                'std_all_aics')

    def __init__(self, pr=MatlabSequence('1:6'), qr=MatlabSequence('1:4')):
        super(MF_ARMA_orders, self).__init__(add_descriptors=False)
        self.pr = pr
        self.qr = qr

    def eval(self, engine, x):
        return HCTSA_MF_ARMA_orders(engine,
                                    x,
                                    pr=self.pr,
                                    qr=self.qr)


def HCTSA_MF_AR_arcov(eng, x, p=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits an AR model of a given order, p, using arcov code from Matlab's Signal
    % Processing Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % p, the AR model order
    % 
    %---OUTPUTS: include the parameters of the fitted model, the variance estimate of a
    % white noise input to the AR model, the root-mean-square (RMS) error of a
    % reconstructed time series, and the autocorrelation of residuals.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['AC1',
                                                      'AC2',
                                                      'a2',
                                                      'a3',
                                                      'a4',
                                                      'a5',
                                                      'a6',
                                                      'e',
                                                      'mu',
                                                      'rms',
                                                      'std']}
    if p is None:
        out = eng.run_function(1, 'MF_AR_arcov', x, )
    else:
        out = eng.run_function(1, 'MF_AR_arcov', x, p)
    return outfunc(out)


class MF_AR_arcov(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits an AR model of a given order, p, using arcov code from Matlab's Signal
    % Processing Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % p, the AR model order
    % 
    %---OUTPUTS: include the parameters of the fitted model, the variance estimate of a
    % white noise input to the AR model, the root-mean-square (RMS) error of a
    % reconstructed time series, and the autocorrelation of residuals.
    % 
    ----------------------------------------
    """

    outnames = ('AC1',
                'AC2',
                'a2',
                'a3',
                'a4',
                'a5',
                'a6',
                'e',
                'mu',
                'rms',
                'std')

    def __init__(self, p=3):
        super(MF_AR_arcov, self).__init__(add_descriptors=False)
        self.p = p

    def eval(self, engine, x):
        return HCTSA_MF_AR_arcov(engine,
                                 x,
                                 p=self.p)


def HCTSA_MF_CompareAR(eng, x, orders=MatlabSequence('1:10'), howtotest=0.5):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares fits of AR models of various orders to the input time series.
    % 
    % Uses functions from Matlab's System Identification Toolbox: iddata, arxstruc,
    % and selstruc
    % 
    %---INPUTS:
    % y, vector of equally-spaced time series data
    % orders, a vector of possible model orders
    % howtotest, specify a fraction, or provide a string 'all' to train and test on
    %            all the data
    % 
    %---OUTPUTS: statistics on the loss at each model order, which are obtained by
    % applying the model trained on the training data to the testing data.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['aic_n',
                                                      'best_n',
                                                      'bestaic',
                                                      'bestmdl',
                                                      'firstonmin',
                                                      'maxdiff',
                                                      'maxonmed',
                                                      'maxv',
                                                      'mdl_n',
                                                      'meandiff',
                                                      'meanv',
                                                      'meddiff',
                                                      'medianv',
                                                      'minstdfromi',
                                                      'minv',
                                                      'stddiff',
                                                      'where01max',
                                                      'whereen4']}
    if orders is None:
        out = eng.run_function(1, 'MF_CompareAR', x, )
    elif howtotest is None:
        out = eng.run_function(1, 'MF_CompareAR', x, orders)
    else:
        out = eng.run_function(1, 'MF_CompareAR', x, orders, howtotest)
    return outfunc(out)


class MF_CompareAR(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares fits of AR models of various orders to the input time series.
    % 
    % Uses functions from Matlab's System Identification Toolbox: iddata, arxstruc,
    % and selstruc
    % 
    %---INPUTS:
    % y, vector of equally-spaced time series data
    % orders, a vector of possible model orders
    % howtotest, specify a fraction, or provide a string 'all' to train and test on
    %            all the data
    % 
    %---OUTPUTS: statistics on the loss at each model order, which are obtained by
    % applying the model trained on the training data to the testing data.
    % 
    ----------------------------------------
    """

    outnames = ('aic_n',
                'best_n',
                'bestaic',
                'bestmdl',
                'firstonmin',
                'maxdiff',
                'maxonmed',
                'maxv',
                'mdl_n',
                'meandiff',
                'meanv',
                'meddiff',
                'medianv',
                'minstdfromi',
                'minv',
                'stddiff',
                'where01max',
                'whereen4')

    def __init__(self, orders=MatlabSequence('1:10'), howtotest=0.5):
        super(MF_CompareAR, self).__init__(add_descriptors=False)
        self.orders = orders
        self.howtotest = howtotest

    def eval(self, engine, x):
        return HCTSA_MF_CompareAR(engine,
                                  x,
                                  orders=self.orders,
                                  howtotest=self.howtotest)


def HCTSA_MF_CompareTestSets(eng, x, model='ar', ord_='best', howtosubset='uniform', samplep=(25, 0.1), steps=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Looks at robustness of test set goodness of fit over different samples in
    % the time series from fitting a given time-series model.
    % 
    % Similar to MF_FitSubsegments, except fits the model on the full time
    % series and compares how well it predicts time series in different local
    % time-series segments.
    % 
    % Says something of stationarity in spread of values, and something of the
    % suitability of model in level of values.
    % 
    % Uses function iddata and predict from Matlab's System Identification Toolbox,
    % as well as either ar, n4sid, or armax from Matlab's System Identification
    % Toolbox to fit the models, depending on the specified model to fit to the data.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % model, the type of time-series model to fit:
    %           (i) 'ar', fits an AR model
    %           (ii) 'ss', first a state-space model
    %           (iii) 'arma', first an ARMA model
    %
    % ord, the order of the specified model to fit
    %
    % howtosubset, how to select random subsets of the time series to fit:
    %           (i) 'rand', select at random
    %           (ii) 'uniform', uniformly distributed segments throughout the time
    %                   series
    %                   
    % samplep, a two-vector specifying the sampling parameters
    %           e.g., [20, 0.1] repeats 20 times for segments 10% the length of the
    %                           time series
    % 
    % steps, the number of steps ahead to do the predictions.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1s_iqr',
                                                      'ac1s_mean',
                                                      'ac1s_median',
                                                      'ac1s_std',
                                                      'mabserr_iqr',
                                                      'mabserr_mean',
                                                      'mabserr_median',
                                                      'mabserr_std',
                                                      'meandiffs_iqr',
                                                      'meandiffs_mean',
                                                      'meandiffs_median',
                                                      'meandiffs_std',
                                                      'rmserr_iqr',
                                                      'rmserr_mean',
                                                      'rmserr_median',
                                                      'rmserr_std',
                                                      'stdrats_iqr',
                                                      'stdrats_mean',
                                                      'stdrats_median',
                                                      'stdrats_std']}
    if model is None:
        out = eng.run_function(1, 'MF_CompareTestSets', x, )
    elif ord_ is None:
        out = eng.run_function(1, 'MF_CompareTestSets', x, model)
    elif howtosubset is None:
        out = eng.run_function(1, 'MF_CompareTestSets', x, model, ord_)
    elif samplep is None:
        out = eng.run_function(1, 'MF_CompareTestSets', x, model, ord_, howtosubset)
    elif steps is None:
        out = eng.run_function(1, 'MF_CompareTestSets', x, model, ord_, howtosubset, samplep)
    else:
        out = eng.run_function(1, 'MF_CompareTestSets', x, model, ord_, howtosubset, samplep, steps)
    return outfunc(out)


class MF_CompareTestSets(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Looks at robustness of test set goodness of fit over different samples in
    % the time series from fitting a given time-series model.
    % 
    % Similar to MF_FitSubsegments, except fits the model on the full time
    % series and compares how well it predicts time series in different local
    % time-series segments.
    % 
    % Says something of stationarity in spread of values, and something of the
    % suitability of model in level of values.
    % 
    % Uses function iddata and predict from Matlab's System Identification Toolbox,
    % as well as either ar, n4sid, or armax from Matlab's System Identification
    % Toolbox to fit the models, depending on the specified model to fit to the data.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % model, the type of time-series model to fit:
    %           (i) 'ar', fits an AR model
    %           (ii) 'ss', first a state-space model
    %           (iii) 'arma', first an ARMA model
    %
    % ord, the order of the specified model to fit
    %
    % howtosubset, how to select random subsets of the time series to fit:
    %           (i) 'rand', select at random
    %           (ii) 'uniform', uniformly distributed segments throughout the time
    %                   series
    %                   
    % samplep, a two-vector specifying the sampling parameters
    %           e.g., [20, 0.1] repeats 20 times for segments 10% the length of the
    %                           time series
    % 
    % steps, the number of steps ahead to do the predictions.
    % 
    ----------------------------------------
    """

    outnames = ('ac1s_iqr',
                'ac1s_mean',
                'ac1s_median',
                'ac1s_std',
                'mabserr_iqr',
                'mabserr_mean',
                'mabserr_median',
                'mabserr_std',
                'meandiffs_iqr',
                'meandiffs_mean',
                'meandiffs_median',
                'meandiffs_std',
                'rmserr_iqr',
                'rmserr_mean',
                'rmserr_median',
                'rmserr_std',
                'stdrats_iqr',
                'stdrats_mean',
                'stdrats_median',
                'stdrats_std')

    def __init__(self, model='ar', ord_='best', howtosubset='uniform', samplep=(25, 0.1), steps=1):
        super(MF_CompareTestSets, self).__init__(add_descriptors=False)
        self.model = model
        self.ord_ = ord_
        self.howtosubset = howtosubset
        self.samplep = samplep
        self.steps = steps

    def eval(self, engine, x):
        return HCTSA_MF_CompareTestSets(engine,
                                        x,
                                        model=self.model,
                                        ord_=self.ord_,
                                        howtosubset=self.howtosubset,
                                        samplep=self.samplep,
                                        steps=self.steps)


def HCTSA_MF_ExpSmoothing(eng, x, ntrain=0.5, alpha='best'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits an exponential smoothing model to the time series using a training set to
    % fit the optimal smoothing parameter, alpha, and then applies the result to the
    % try to predict the rest of the time series.
    % 
    % cf. "The Analysis of Time Series", C. Chatfield, CRC Press LLC (2004)
    % 
    % Code is adapted from original code provided by Siddharth Arora:
    % Siddharth.Arora@sbs.ox.ac.uk
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % ntrain, the number of samples to use for training (can be a proportion of the
    %           time-series length)
    %           
    % alpha, the exponential smoothing parameter
    % 
    %---OUTPUTS: include the fitted alpha, and statistics on the residuals from the
    % prediction phase.
    % 
    % Future alteration could take a number of training sets and average to some
    % optimal alpha, for example, rather than just fitting it in an initial portion
    % of the time series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac1n',
                                                      'ac2',
                                                      'ac2n',
                                                      'ac3',
                                                      'ac3n',
                                                      'acmnd0',
                                                      'acsnd0',
                                                      'alphamin',
                                                      'alphamin_1',
                                                      'cup_1',
                                                      'dwts',
                                                      'ftbth',
                                                      'maxonmean',
                                                      'meanabs',
                                                      'meane',
                                                      'minfpe',
                                                      'minsbc',
                                                      'mms',
                                                      'normksstat',
                                                      'normp',
                                                      'p1_1',
                                                      'p1_5',
                                                      'p2_5',
                                                      'p3_5',
                                                      'p4_5',
                                                      'p5_5',
                                                      'popt',
                                                      'propbth',
                                                      'rmse',
                                                      'sbc1',
                                                      'stde']}
    if ntrain is None:
        out = eng.run_function(1, 'MF_ExpSmoothing', x, )
    elif alpha is None:
        out = eng.run_function(1, 'MF_ExpSmoothing', x, ntrain)
    else:
        out = eng.run_function(1, 'MF_ExpSmoothing', x, ntrain, alpha)
    return outfunc(out)


class MF_ExpSmoothing(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits an exponential smoothing model to the time series using a training set to
    % fit the optimal smoothing parameter, alpha, and then applies the result to the
    % try to predict the rest of the time series.
    % 
    % cf. "The Analysis of Time Series", C. Chatfield, CRC Press LLC (2004)
    % 
    % Code is adapted from original code provided by Siddharth Arora:
    % Siddharth.Arora@sbs.ox.ac.uk
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % ntrain, the number of samples to use for training (can be a proportion of the
    %           time-series length)
    %           
    % alpha, the exponential smoothing parameter
    % 
    %---OUTPUTS: include the fitted alpha, and statistics on the residuals from the
    % prediction phase.
    % 
    % Future alteration could take a number of training sets and average to some
    % optimal alpha, for example, rather than just fitting it in an initial portion
    % of the time series.
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac1n',
                'ac2',
                'ac2n',
                'ac3',
                'ac3n',
                'acmnd0',
                'acsnd0',
                'alphamin',
                'alphamin_1',
                'cup_1',
                'dwts',
                'ftbth',
                'maxonmean',
                'meanabs',
                'meane',
                'minfpe',
                'minsbc',
                'mms',
                'normksstat',
                'normp',
                'p1_1',
                'p1_5',
                'p2_5',
                'p3_5',
                'p4_5',
                'p5_5',
                'popt',
                'propbth',
                'rmse',
                'sbc1',
                'stde')

    def __init__(self, ntrain=0.5, alpha='best'):
        super(MF_ExpSmoothing, self).__init__(add_descriptors=False)
        self.ntrain = ntrain
        self.alpha = alpha

    def eval(self, engine, x):
        return HCTSA_MF_ExpSmoothing(engine,
                                     x,
                                     ntrain=self.ntrain,
                                     alpha=self.alpha)


def HCTSA_MF_FitSubsegments(eng, x, model='ar', order=2, howtosubset='uniform', samplep=(25, 0.1)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Looks at the robustness of fitted parameters from a model applied to different
    % segments of the time series.
    % 
    % The spread of parameters obtained (including in-sample goodness of fit
    % statistics) provide some indication of stationarity.
    % 
    % Values of goodness of fit provide some indication of model suitability.
    % 
    % N.B., this code inherits strongly from this MF_CompareTestSets
    % 
    %---INPUTS:
    % y, the input time series.
    % 
    % model, the model to fit in each segments of the time series:
    %           'arsbc': fits an AR model using the ARfit package. Outputs
    %                       statistics are on how the optimal order, p_{opt}, and
    %                       the goodness of fit varies in different parts of the
    %                       time series.
    %           'ar': fits an AR model of a specified order using the code
    %                   ar from Matlab's System Identification Toolbox. Outputs are
    %                   on how Akaike's Final Prediction Error (FPE), and the fitted
    %                   AR parameters vary across the different segments of time
    %                   series.
    %           'ss': fits a state space model of a given order using the code
    %                   n4sid from Matlab's System Identification Toolbox. Outputs
    %                   are on how the FPE varies.
    %           'arma': fits an ARMA model using armax code from Matlab's System
    %                   Identification Toolbox. Outputs are statistics on the FPE,
    %                   and fitted AR and MA parameters.
    %                   
    % howtosubset, how to choose segments from the time series, either 'uniform'
    %               (uniformly) or 'rand' (at random).
    %               
    % samplep, a two-vector specifying how many segments to take and of what length.
    %           Of the form [nsamples, length], where length can be a proportion of
    %           the time-series length. e.g., [20,0.1] takes 20 segments of 10% the
    %           time-series length.
    % 
    %---OUTPUTS: depend on the model, as described above.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['a_1_max',
                                                      'a_1_mean',
                                                      'a_1_min',
                                                      'a_1_std',
                                                      'a_2_max',
                                                      'a_2_mean',
                                                      'a_2_min',
                                                      'a_2_std',
                                                      'fpe_max',
                                                      'fpe_mean',
                                                      'fpe_min',
                                                      'fpe_range',
                                                      'fpe_std',
                                                      'orders_max',
                                                      'orders_mean',
                                                      'orders_min',
                                                      'orders_mode',
                                                      'orders_range',
                                                      'orders_std',
                                                      'p_1_max',
                                                      'p_1_mean',
                                                      'p_1_min',
                                                      'p_1_std',
                                                      'p_2_max',
                                                      'p_2_mean',
                                                      'p_2_min',
                                                      'p_2_std',
                                                      'q_1_max',
                                                      'q_1_mean',
                                                      'q_1_min',
                                                      'q_1_std',
                                                      'q_2_max',
                                                      'q_2_mean',
                                                      'q_2_min',
                                                      'q_2_std',
                                                      'sbcs_max',
                                                      'sbcs_mean',
                                                      'sbcs_min',
                                                      'sbcs_range',
                                                      'sbcs_std']}
    if model is None:
        out = eng.run_function(1, 'MF_FitSubsegments', x, )
    elif order is None:
        out = eng.run_function(1, 'MF_FitSubsegments', x, model)
    elif howtosubset is None:
        out = eng.run_function(1, 'MF_FitSubsegments', x, model, order)
    elif samplep is None:
        out = eng.run_function(1, 'MF_FitSubsegments', x, model, order, howtosubset)
    else:
        out = eng.run_function(1, 'MF_FitSubsegments', x, model, order, howtosubset, samplep)
    return outfunc(out)


class MF_FitSubsegments(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Looks at the robustness of fitted parameters from a model applied to different
    % segments of the time series.
    % 
    % The spread of parameters obtained (including in-sample goodness of fit
    % statistics) provide some indication of stationarity.
    % 
    % Values of goodness of fit provide some indication of model suitability.
    % 
    % N.B., this code inherits strongly from this MF_CompareTestSets
    % 
    %---INPUTS:
    % y, the input time series.
    % 
    % model, the model to fit in each segments of the time series:
    %           'arsbc': fits an AR model using the ARfit package. Outputs
    %                       statistics are on how the optimal order, p_{opt}, and
    %                       the goodness of fit varies in different parts of the
    %                       time series.
    %           'ar': fits an AR model of a specified order using the code
    %                   ar from Matlab's System Identification Toolbox. Outputs are
    %                   on how Akaike's Final Prediction Error (FPE), and the fitted
    %                   AR parameters vary across the different segments of time
    %                   series.
    %           'ss': fits a state space model of a given order using the code
    %                   n4sid from Matlab's System Identification Toolbox. Outputs
    %                   are on how the FPE varies.
    %           'arma': fits an ARMA model using armax code from Matlab's System
    %                   Identification Toolbox. Outputs are statistics on the FPE,
    %                   and fitted AR and MA parameters.
    %                   
    % howtosubset, how to choose segments from the time series, either 'uniform'
    %               (uniformly) or 'rand' (at random).
    %               
    % samplep, a two-vector specifying how many segments to take and of what length.
    %           Of the form [nsamples, length], where length can be a proportion of
    %           the time-series length. e.g., [20,0.1] takes 20 segments of 10% the
    %           time-series length.
    % 
    %---OUTPUTS: depend on the model, as described above.
    % 
    ----------------------------------------
    """

    outnames = ('a_1_max',
                'a_1_mean',
                'a_1_min',
                'a_1_std',
                'a_2_max',
                'a_2_mean',
                'a_2_min',
                'a_2_std',
                'fpe_max',
                'fpe_mean',
                'fpe_min',
                'fpe_range',
                'fpe_std',
                'orders_max',
                'orders_mean',
                'orders_min',
                'orders_mode',
                'orders_range',
                'orders_std',
                'p_1_max',
                'p_1_mean',
                'p_1_min',
                'p_1_std',
                'p_2_max',
                'p_2_mean',
                'p_2_min',
                'p_2_std',
                'q_1_max',
                'q_1_mean',
                'q_1_min',
                'q_1_std',
                'q_2_max',
                'q_2_mean',
                'q_2_min',
                'q_2_std',
                'sbcs_max',
                'sbcs_mean',
                'sbcs_min',
                'sbcs_range',
                'sbcs_std')

    def __init__(self, model='ar', order=2, howtosubset='uniform', samplep=(25, 0.1)):
        super(MF_FitSubsegments, self).__init__(add_descriptors=False)
        self.model = model
        self.order = order
        self.howtosubset = howtosubset
        self.samplep = samplep

    def eval(self, engine, x):
        return HCTSA_MF_FitSubsegments(engine,
                                       x,
                                       model=self.model,
                                       order=self.order,
                                       howtosubset=self.howtosubset,
                                       samplep=self.samplep)


def HCTSA_MF_GARCHcompare(eng, x, preproc='ar', pr=MatlabSequence('1:3'), qr=MatlabSequence('1:3')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This code fits a set of GARCH(p,q) models to the time series and
    % returns statistics on the goodness of fits across a range of p and
    % q parameters.
    % 
    % Uses the following functions from Matlab's Econometrics Toolbox: archtest,
    % lbqtest, autocorr, parcorr, garchset, garchfit, garchcount, aicbic
    % 
    %---INPUTS:
    % y, the input time series
    % preproc, a preprocessing to apply:
    %           (i) 'none': no preprocessing is performed
    %           (ii) 'ar': performs a preprocessing that maximizes AR(2) whiteness,
    %           
    % pr, a vector of model orders, p, to compare
    % 
    % qr, a vector of model orders, q, to compare
    % 
    % Compares all combinations of p and q and output statistics are on the models
    % with the best fit.
    % 
    % This operation focuses on the GARCH/variance component, and therefore
    % attempts to pre-whiten and assumes a constant mean process.
    % 
    %---OUTPUTS: include log-likelihoods, Bayesian Information  Criteria (BIC),
    % Akaike's Information Criteria (AIC), outputs from Engle's ARCH test and the
    % Ljung-Box Q-test, and estimates of optimal model orders.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['Ks_vary_p',
                                                      'Ks_vary_q',
                                                      'bestpAIC',
                                                      'bestpLLF',
                                                      'bestqAIC',
                                                      'bestqLLF',
                                                      'maxAIC',
                                                      'maxBIC',
                                                      'maxK',
                                                      'maxLLF',
                                                      'max_maxarchps',
                                                      'max_maxlbqps',
                                                      'max_meanarchps',
                                                      'max_meanlbqps',
                                                      'meanAIC',
                                                      'meanBIC',
                                                      'meanK',
                                                      'meanLLF',
                                                      'mean_maxarchps',
                                                      'mean_maxlbqps',
                                                      'mean_meanarchps',
                                                      'mean_meanlbqps',
                                                      'minAIC',
                                                      'minBIC',
                                                      'minK',
                                                      'minLLF',
                                                      'min_maxarchps',
                                                      'min_maxlbqps',
                                                      'min_meanarchps',
                                                      'min_meanlbqps']}
    if preproc is None:
        out = eng.run_function(1, 'MF_GARCHcompare', x, )
    elif pr is None:
        out = eng.run_function(1, 'MF_GARCHcompare', x, preproc)
    elif qr is None:
        out = eng.run_function(1, 'MF_GARCHcompare', x, preproc, pr)
    else:
        out = eng.run_function(1, 'MF_GARCHcompare', x, preproc, pr, qr)
    return outfunc(out)


class MF_GARCHcompare(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This code fits a set of GARCH(p,q) models to the time series and
    % returns statistics on the goodness of fits across a range of p and
    % q parameters.
    % 
    % Uses the following functions from Matlab's Econometrics Toolbox: archtest,
    % lbqtest, autocorr, parcorr, garchset, garchfit, garchcount, aicbic
    % 
    %---INPUTS:
    % y, the input time series
    % preproc, a preprocessing to apply:
    %           (i) 'none': no preprocessing is performed
    %           (ii) 'ar': performs a preprocessing that maximizes AR(2) whiteness,
    %           
    % pr, a vector of model orders, p, to compare
    % 
    % qr, a vector of model orders, q, to compare
    % 
    % Compares all combinations of p and q and output statistics are on the models
    % with the best fit.
    % 
    % This operation focuses on the GARCH/variance component, and therefore
    % attempts to pre-whiten and assumes a constant mean process.
    % 
    %---OUTPUTS: include log-likelihoods, Bayesian Information  Criteria (BIC),
    % Akaike's Information Criteria (AIC), outputs from Engle's ARCH test and the
    % Ljung-Box Q-test, and estimates of optimal model orders.
    % 
    ----------------------------------------
    """

    outnames = ('Ks_vary_p',
                'Ks_vary_q',
                'bestpAIC',
                'bestpLLF',
                'bestqAIC',
                'bestqLLF',
                'maxAIC',
                'maxBIC',
                'maxK',
                'maxLLF',
                'max_maxarchps',
                'max_maxlbqps',
                'max_meanarchps',
                'max_meanlbqps',
                'meanAIC',
                'meanBIC',
                'meanK',
                'meanLLF',
                'mean_maxarchps',
                'mean_maxlbqps',
                'mean_meanarchps',
                'mean_meanlbqps',
                'minAIC',
                'minBIC',
                'minK',
                'minLLF',
                'min_maxarchps',
                'min_maxlbqps',
                'min_meanarchps',
                'min_meanlbqps')

    def __init__(self, preproc='ar', pr=MatlabSequence('1:3'), qr=MatlabSequence('1:3')):
        super(MF_GARCHcompare, self).__init__(add_descriptors=False)
        self.preproc = preproc
        self.pr = pr
        self.qr = qr

    def eval(self, engine, x):
        return HCTSA_MF_GARCHcompare(engine,
                                     x,
                                     preproc=self.preproc,
                                     pr=self.pr,
                                     qr=self.qr)


def HCTSA_MF_GARCHfit(eng, x, preproc='ar', params="'R',2,'M',1,'P',2,'Q',1"):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Simulates a procedure for fitting Generalized Autoregressive Conditional
    % Heteroskedasticity (GARCH) models to a time series, namely:
    % 
    % (1) Preprocessing the data to remove strong trends,
    % (2) Pre-estimation to calculate initial correlation properties of the time
    %       series and motivate a GARCH model,
    % (3) Fitting a GARCH model, returning goodness of fit statistics and parameters
    %           of the fitted model, and
    % (4) Post-estimation, involves calculating statistics on residuals and
    %           standardized residuals.
    % 
    % The idea is that all of these stages can be pre-specified or skipped using
    % arguments to the function.
    % 
    % Uses functions from Matlab's Econometrics Toolbox: archtest, lbqtest,
    % autocorr, parcorr, garchset, garchfit, garchcount, aicbic
    % 
    % All methods implemented are from Matlab's Econometrics Toolbox, including
    % Engle's ARCH test (archtest), the Ljung-Box Q-test (lbqtest), estimating the
    % partial autocorrelation function (parcorr), as well as specifying (garchset)
    % and fitting (garchfit) the GARCH model to the time series.
    % 
    % As part of this code, a very basic automatic pre-processing routine,
    % PP_ModelFit, is implemented, that applies a range of pre-processings and
    % returns the preprocessing of the time series that shows the worst fit to an
    % AR(2) model.
    % 
    % In the case that no simple transformations of the time series are
    % significantly more stationary/less trivially correlated than the original time
    % series (by more than 5%), the original time series is simply used without
    % transformation.
    % 
    % Where r and m are the autoregressive and moving average orders of the model,
    % respectively, and p and q control the conditional variance parameters.
    % 
    %---INPUTS:
    % y, the input time series
    % preproc, the preprocessing to apply, can be 'ar' or 'none'
    % params, the parameters of the GARCH model to fit, can be:
    %           (i) 'default', fits the default model
    %           (ii) 'auto', automated routine to select parameters for this time series
    %           (iii) e.g., params = '''R'',2,''M'',1,''P'',2,''Q'',1', sets r = 2,
    %                                   m = 1, p = 2, q = 1
    % 
    % 
    % ***In future this code should be revised by an expert in GARCH model fitting.***
    % 
    %---HISTORY:
    % Ben Fulcher, 25/2/2010
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ARCH_1',
                                                      'ARCHerr_1',
                                                      'AR_1',
                                                      'AR_2',
                                                      'ARerr_1',
                                                      'ARerr_2',
                                                      'GARCH_1',
                                                      'GARCH_2',
                                                      'GARCHerr_1',
                                                      'GARCHerr_2',
                                                      'K',
                                                      'LLF',
                                                      'MA_1',
                                                      'MAerr_1',
                                                      'ac1_stde2',
                                                      'aic',
                                                      'bic',
                                                      'diff_ac1',
                                                      'engle_max_diff_p',
                                                      'engle_mean_diff_p',
                                                      'engle_pval_stde_1',
                                                      'engle_pval_stde_10',
                                                      'engle_pval_stde_5',
                                                      'lbq_max_diff_p',
                                                      'lbq_mean_diff_p',
                                                      'lbq_pval_stde_1',
                                                      'lbq_pval_stde_10',
                                                      'lbq_pval_stde_5',
                                                      'maxenglepval_stde',
                                                      'maxlbqpval_stde2',
                                                      'maxsigma',
                                                      'meansigma',
                                                      'minenglepval_stde',
                                                      'minlbqpval_stde2',
                                                      'minsigma',
                                                      'nparams',
                                                      'rangesigma',
                                                      'stde_ac1',
                                                      'stde_ac1n',
                                                      'stde_ac2',
                                                      'stde_ac2n',
                                                      'stde_ac3',
                                                      'stde_ac3n',
                                                      'stde_acmnd0',
                                                      'stde_acsnd0',
                                                      'stde_dwts',
                                                      'stde_ftbth',
                                                      'stde_maxonmean',
                                                      'stde_meanabs',
                                                      'stde_meane',
                                                      'stde_minfpe',
                                                      'stde_minsbc',
                                                      'stde_mms',
                                                      'stde_normksstat',
                                                      'stde_normp',
                                                      'stde_p1_5',
                                                      'stde_p2_5',
                                                      'stde_p3_5',
                                                      'stde_p4_5',
                                                      'stde_p5_5',
                                                      'stde_popt',
                                                      'stde_propbth',
                                                      'stde_rmse',
                                                      'stde_sbc1',
                                                      'stde_stde',
                                                      'stdsigma',
                                                      'summaryexitflag']}
    if preproc is None:
        out = eng.run_function(1, 'MF_GARCHfit', x, )
    elif params is None:
        out = eng.run_function(1, 'MF_GARCHfit', x, preproc)
    else:
        out = eng.run_function(1, 'MF_GARCHfit', x, preproc, params)
    return outfunc(out)


class MF_GARCHfit(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Simulates a procedure for fitting Generalized Autoregressive Conditional
    % Heteroskedasticity (GARCH) models to a time series, namely:
    % 
    % (1) Preprocessing the data to remove strong trends,
    % (2) Pre-estimation to calculate initial correlation properties of the time
    %       series and motivate a GARCH model,
    % (3) Fitting a GARCH model, returning goodness of fit statistics and parameters
    %           of the fitted model, and
    % (4) Post-estimation, involves calculating statistics on residuals and
    %           standardized residuals.
    % 
    % The idea is that all of these stages can be pre-specified or skipped using
    % arguments to the function.
    % 
    % Uses functions from Matlab's Econometrics Toolbox: archtest, lbqtest,
    % autocorr, parcorr, garchset, garchfit, garchcount, aicbic
    % 
    % All methods implemented are from Matlab's Econometrics Toolbox, including
    % Engle's ARCH test (archtest), the Ljung-Box Q-test (lbqtest), estimating the
    % partial autocorrelation function (parcorr), as well as specifying (garchset)
    % and fitting (garchfit) the GARCH model to the time series.
    % 
    % As part of this code, a very basic automatic pre-processing routine,
    % PP_ModelFit, is implemented, that applies a range of pre-processings and
    % returns the preprocessing of the time series that shows the worst fit to an
    % AR(2) model.
    % 
    % In the case that no simple transformations of the time series are
    % significantly more stationary/less trivially correlated than the original time
    % series (by more than 5%), the original time series is simply used without
    % transformation.
    % 
    % Where r and m are the autoregressive and moving average orders of the model,
    % respectively, and p and q control the conditional variance parameters.
    % 
    %---INPUTS:
    % y, the input time series
    % preproc, the preprocessing to apply, can be 'ar' or 'none'
    % params, the parameters of the GARCH model to fit, can be:
    %           (i) 'default', fits the default model
    %           (ii) 'auto', automated routine to select parameters for this time series
    %           (iii) e.g., params = '''R'',2,''M'',1,''P'',2,''Q'',1', sets r = 2,
    %                                   m = 1, p = 2, q = 1
    % 
    % 
    % ***In future this code should be revised by an expert in GARCH model fitting.***
    % 
    %---HISTORY:
    % Ben Fulcher, 25/2/2010
    % 
    ----------------------------------------
    """

    outnames = ('ARCH_1',
                'ARCHerr_1',
                'AR_1',
                'AR_2',
                'ARerr_1',
                'ARerr_2',
                'GARCH_1',
                'GARCH_2',
                'GARCHerr_1',
                'GARCHerr_2',
                'K',
                'LLF',
                'MA_1',
                'MAerr_1',
                'ac1_stde2',
                'aic',
                'bic',
                'diff_ac1',
                'engle_max_diff_p',
                'engle_mean_diff_p',
                'engle_pval_stde_1',
                'engle_pval_stde_10',
                'engle_pval_stde_5',
                'lbq_max_diff_p',
                'lbq_mean_diff_p',
                'lbq_pval_stde_1',
                'lbq_pval_stde_10',
                'lbq_pval_stde_5',
                'maxenglepval_stde',
                'maxlbqpval_stde2',
                'maxsigma',
                'meansigma',
                'minenglepval_stde',
                'minlbqpval_stde2',
                'minsigma',
                'nparams',
                'rangesigma',
                'stde_ac1',
                'stde_ac1n',
                'stde_ac2',
                'stde_ac2n',
                'stde_ac3',
                'stde_ac3n',
                'stde_acmnd0',
                'stde_acsnd0',
                'stde_dwts',
                'stde_ftbth',
                'stde_maxonmean',
                'stde_meanabs',
                'stde_meane',
                'stde_minfpe',
                'stde_minsbc',
                'stde_mms',
                'stde_normksstat',
                'stde_normp',
                'stde_p1_5',
                'stde_p2_5',
                'stde_p3_5',
                'stde_p4_5',
                'stde_p5_5',
                'stde_popt',
                'stde_propbth',
                'stde_rmse',
                'stde_sbc1',
                'stde_stde',
                'stdsigma',
                'summaryexitflag')

    def __init__(self, preproc='ar', params="'R',2,'M',1,'P',2,'Q',1"):
        super(MF_GARCHfit, self).__init__(add_descriptors=False)
        self.preproc = preproc
        self.params = params

    def eval(self, engine, x):
        return HCTSA_MF_GARCHfit(engine,
                                 x,
                                 preproc=self.preproc,
                                 params=self.params)


def HCTSA_MF_GP_FitAcross(eng, x, covfunc=('covSum', ('covSEiso', 'covNoise')), npoints=20):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Trains a Gaussian Process model on equally-spaced points throughout the time
    % series and uses the model to predict its intermediate values.
    % 
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    % 
    %---INPUTS:
    % y, the input time series
    % covfunc, the covariance function (structured in the standard way for the gpml toolbox)
    % npoints, the number of points through the time series to fit the GP model to
    % 
    %---OUTPUTS: summarize the error and fitted hyperparameters.
    % 
    % In future could do a better job of the sampling of points -- perhaps to take
    % into account the autocorrelation of the time series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['h_lonN',
                                                      'logh1',
                                                      'logh2',
                                                      'logh3',
                                                      'meanS',
                                                      'meanstderr',
                                                      'mlikelihood',
                                                      'rmserr',
                                                      'stdS',
                                                      'stdmu']}
    if covfunc is None:
        out = eng.run_function(1, 'MF_GP_FitAcross', x, )
    elif npoints is None:
        out = eng.run_function(1, 'MF_GP_FitAcross', x, covfunc)
    else:
        out = eng.run_function(1, 'MF_GP_FitAcross', x, covfunc, npoints)
    return outfunc(out)


class MF_GP_FitAcross(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Trains a Gaussian Process model on equally-spaced points throughout the time
    % series and uses the model to predict its intermediate values.
    % 
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    % 
    %---INPUTS:
    % y, the input time series
    % covfunc, the covariance function (structured in the standard way for the gpml toolbox)
    % npoints, the number of points through the time series to fit the GP model to
    % 
    %---OUTPUTS: summarize the error and fitted hyperparameters.
    % 
    % In future could do a better job of the sampling of points -- perhaps to take
    % into account the autocorrelation of the time series.
    % 
    ----------------------------------------
    """

    outnames = ('h_lonN',
                'logh1',
                'logh2',
                'logh3',
                'meanS',
                'meanstderr',
                'mlikelihood',
                'rmserr',
                'stdS',
                'stdmu')

    def __init__(self, covfunc=('covSum', ('covSEiso', 'covNoise')), npoints=20):
        super(MF_GP_FitAcross, self).__init__(add_descriptors=False)
        self.covfunc = covfunc
        self.npoints = npoints

    def eval(self, engine, x):
        return HCTSA_MF_GP_FitAcross(engine,
                                     x,
                                     covfunc=self.covfunc,
                                     npoints=self.npoints)


def HCTSA_MF_GP_LearnHyperp(eng, x, nfevals=None, t=None, y=None, init_loghyper=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Function used by main Gaussian Process model fitting operations that learns
    % Gaussian Process hyperparameters for the time series.
    % 
    % References code 'minimize' from the GAUSSIAN PROCESS REGRESSION AND
    % CLASSIFICATION Toolbox version 3.2, which is avilable at:
    % http://gaussianprocess.org/gpml/code
    % 
    %---INPUTS:
    % 
    % covfunc,       the covariance function, formated as gpml likes it
    % nfevals,       the number of function evaluations
    % t,             time
    % y,             data
    % init_loghyper, inital values for hyperparameters
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if nfevals is None:
        out = eng.run_function(1, 'MF_GP_LearnHyperp', x, )
    elif t is None:
        out = eng.run_function(1, 'MF_GP_LearnHyperp', x, nfevals)
    elif y is None:
        out = eng.run_function(1, 'MF_GP_LearnHyperp', x, nfevals, t)
    elif init_loghyper is None:
        out = eng.run_function(1, 'MF_GP_LearnHyperp', x, nfevals, t, y)
    else:
        out = eng.run_function(1, 'MF_GP_LearnHyperp', x, nfevals, t, y, init_loghyper)
    return outfunc(out)


class MF_GP_LearnHyperp(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Function used by main Gaussian Process model fitting operations that learns
    % Gaussian Process hyperparameters for the time series.
    % 
    % References code 'minimize' from the GAUSSIAN PROCESS REGRESSION AND
    % CLASSIFICATION Toolbox version 3.2, which is avilable at:
    % http://gaussianprocess.org/gpml/code
    % 
    %---INPUTS:
    % 
    % covfunc,       the covariance function, formated as gpml likes it
    % nfevals,       the number of function evaluations
    % t,             time
    % y,             data
    % init_loghyper, inital values for hyperparameters
    % 
    ----------------------------------------
    """

    def __init__(self, nfevals=None, t=None, y=None, init_loghyper=None):
        super(MF_GP_LearnHyperp, self).__init__(add_descriptors=False)
        self.nfevals = nfevals
        self.t = t
        self.y = y
        self.init_loghyper = init_loghyper

    def eval(self, engine, x):
        return HCTSA_MF_GP_LearnHyperp(engine,
                                       x,
                                       nfevals=self.nfevals,
                                       t=self.t,
                                       y=self.y,
                                       init_loghyper=self.init_loghyper)


def HCTSA_MF_GP_LocalPrediction(eng, x, covfunc=('covSum', ('covSEiso', 'covNoise')),
                                ntrain=10, ntest=3, npreds=20, pmode='frombefore'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a given Gaussian Process model to a section of the time series and uses
    % it to predict to the subsequent datapoint.
    % 
    % % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % covfunc, covariance function in the standard form for the gpml package.
    %           E.g., covfunc = {'covSum', {'covSEiso','covNoise'}} combines squared 
    %           exponential and noise terms
    %           
    % ntrain, the number of training samples (for each iteration)
    % 
    % ntest, the number of testing samples (for each interation)
    % 
    % npreds, the number of predictions to make
    % 
    % pmode, the prediction mode:
    %       (i) 'beforeafter': predicts the preceding time series values by training
    %                           on the following values,
    %       (ii) 'frombefore': predicts the following values of the time series by
    %                    training on preceding values, and
    %       (iii) 'randomgap': predicts random values within a segment of time
    %                    series by training on the other values in that segment.
    % 
    % 
    %---OUTPUTS: summaries of the quality of predictions made, the mean and
    % spread of obtained hyperparameter values, and marginal likelihoods.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['maxabserr',
                                                      'maxabserr_run',
                                                      'maxerrbar',
                                                      'maxmlik',
                                                      'maxstderr',
                                                      'maxstderr_run',
                                                      'meanabserr',
                                                      'meanabserr_run',
                                                      'meanerrbar',
                                                      'meanlogh1',
                                                      'meanlogh2',
                                                      'meanlogh3',
                                                      'meanstderr',
                                                      'meanstderr_run',
                                                      'minabserr',
                                                      'minabserr_run',
                                                      'minerrbar',
                                                      'minmlik',
                                                      'minstderr',
                                                      'minstderr_run',
                                                      'stdlogh1',
                                                      'stdlogh2',
                                                      'stdlogh3',
                                                      'stdmlik']}
    if covfunc is None:
        out = eng.run_function(1, 'MF_GP_LocalPrediction', x, )
    elif ntrain is None:
        out = eng.run_function(1, 'MF_GP_LocalPrediction', x, covfunc)
    elif ntest is None:
        out = eng.run_function(1, 'MF_GP_LocalPrediction', x, covfunc, ntrain)
    elif npreds is None:
        out = eng.run_function(1, 'MF_GP_LocalPrediction', x, covfunc, ntrain, ntest)
    elif pmode is None:
        out = eng.run_function(1, 'MF_GP_LocalPrediction', x, covfunc, ntrain, ntest, npreds)
    else:
        out = eng.run_function(1, 'MF_GP_LocalPrediction', x, covfunc, ntrain, ntest, npreds, pmode)
    return outfunc(out)


class MF_GP_LocalPrediction(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a given Gaussian Process model to a section of the time series and uses
    % it to predict to the subsequent datapoint.
    % 
    % % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % covfunc, covariance function in the standard form for the gpml package.
    %           E.g., covfunc = {'covSum', {'covSEiso','covNoise'}} combines squared 
    %           exponential and noise terms
    %           
    % ntrain, the number of training samples (for each iteration)
    % 
    % ntest, the number of testing samples (for each interation)
    % 
    % npreds, the number of predictions to make
    % 
    % pmode, the prediction mode:
    %       (i) 'beforeafter': predicts the preceding time series values by training
    %                           on the following values,
    %       (ii) 'frombefore': predicts the following values of the time series by
    %                    training on preceding values, and
    %       (iii) 'randomgap': predicts random values within a segment of time
    %                    series by training on the other values in that segment.
    % 
    % 
    %---OUTPUTS: summaries of the quality of predictions made, the mean and
    % spread of obtained hyperparameter values, and marginal likelihoods.
    % 
    ----------------------------------------
    """

    outnames = ('maxabserr',
                'maxabserr_run',
                'maxerrbar',
                'maxmlik',
                'maxstderr',
                'maxstderr_run',
                'meanabserr',
                'meanabserr_run',
                'meanerrbar',
                'meanlogh1',
                'meanlogh2',
                'meanlogh3',
                'meanstderr',
                'meanstderr_run',
                'minabserr',
                'minabserr_run',
                'minerrbar',
                'minmlik',
                'minstderr',
                'minstderr_run',
                'stdlogh1',
                'stdlogh2',
                'stdlogh3',
                'stdmlik')

    def __init__(self, covfunc=('covSum', ('covSEiso', 'covNoise')), ntrain=10, ntest=3, npreds=20, pmode='frombefore'):
        super(MF_GP_LocalPrediction, self).__init__(add_descriptors=False)
        self.covfunc = covfunc
        self.ntrain = ntrain
        self.ntest = ntest
        self.npreds = npreds
        self.pmode = pmode

    def eval(self, engine, x):
        return HCTSA_MF_GP_LocalPrediction(engine,
                                           x,
                                           covfunc=self.covfunc,
                                           ntrain=self.ntrain,
                                           ntest=self.ntest,
                                           npreds=self.npreds,
                                           pmode=self.pmode)


def HCTSA_MF_GP_hyperparameters(eng, x, covfunc=('covSum', ('covSEiso', 'covNoise')),
                                squishorsquash=1, maxN=200, methds='first'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a Gaussian Process model to a portion of the time series and returns the
    % fitted hyperparameters, as well as statistics describing the goodness of fit
    % of the model.
    % 
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    % 
    % The code can accomodate a range of covariance functions, e.g.:
    % (i) a sum of squared exponential and noise terms, and
    % (ii) a sum of squared exponential, periodic, and noise terms.
    % 
    % The model is fitted to <> samples from the time series, which are
    % chosen by:
    % (i) resampling the time series down to this many data points,
    % (ii) taking the first 200 samples from the time series, or
    % (iii) taking random samples from the time series.
    % 
    % INPUTS:
    % y, the input time series
    % covfunc, the covariance function, in the standard form fo the gmpl package
    % squishorsquash, whether to squash onto the unit interval, or spread across 1:N
    % maxN, the maximum length of time series to consider -- greater than this
    %               length, time series are resampled down to maxN
    % methds, specifies the method of how to resample time series longer than maxN
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['h1',
                                                      'h2',
                                                      'h3',
                                                      'h4',
                                                      'h5',
                                                      'logh1',
                                                      'logh2',
                                                      'logh3',
                                                      'logh4',
                                                      'logh5',
                                                      'mabserr_std',
                                                      'maxS',
                                                      'meanS',
                                                      'minS',
                                                      'mlikelihood',
                                                      'rmserr',
                                                      'std_S_data',
                                                      'std_mu_data']}
    if covfunc is None:
        out = eng.run_function(1, 'MF_GP_hyperparameters', x, )
    elif squishorsquash is None:
        out = eng.run_function(1, 'MF_GP_hyperparameters', x, covfunc)
    elif maxN is None:
        out = eng.run_function(1, 'MF_GP_hyperparameters', x, covfunc, squishorsquash)
    elif methds is None:
        out = eng.run_function(1, 'MF_GP_hyperparameters', x, covfunc, squishorsquash, maxN)
    else:
        out = eng.run_function(1, 'MF_GP_hyperparameters', x, covfunc, squishorsquash, maxN, methds)
    return outfunc(out)


class MF_GP_hyperparameters(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a Gaussian Process model to a portion of the time series and returns the
    % fitted hyperparameters, as well as statistics describing the goodness of fit
    % of the model.
    % 
    % Uses GP fitting code from the gpml toolbox, which is available here:
    % http://gaussianprocess.org/gpml/code.
    % 
    % The code can accomodate a range of covariance functions, e.g.:
    % (i) a sum of squared exponential and noise terms, and
    % (ii) a sum of squared exponential, periodic, and noise terms.
    % 
    % The model is fitted to <> samples from the time series, which are
    % chosen by:
    % (i) resampling the time series down to this many data points,
    % (ii) taking the first 200 samples from the time series, or
    % (iii) taking random samples from the time series.
    % 
    % INPUTS:
    % y, the input time series
    % covfunc, the covariance function, in the standard form fo the gmpl package
    % squishorsquash, whether to squash onto the unit interval, or spread across 1:N
    % maxN, the maximum length of time series to consider -- greater than this
    %               length, time series are resampled down to maxN
    % methds, specifies the method of how to resample time series longer than maxN
    % 
    ----------------------------------------
    """

    outnames = ('h1',
                'h2',
                'h3',
                'h4',
                'h5',
                'logh1',
                'logh2',
                'logh3',
                'logh4',
                'logh5',
                'mabserr_std',
                'maxS',
                'meanS',
                'minS',
                'mlikelihood',
                'rmserr',
                'std_S_data',
                'std_mu_data')

    def __init__(self, covfunc=('covSum', ('covSEiso', 'covNoise')), squishorsquash=1, maxN=200, methds='first'):
        super(MF_GP_hyperparameters, self).__init__(add_descriptors=False)
        self.covfunc = covfunc
        self.squishorsquash = squishorsquash
        self.maxN = maxN
        self.methds = methds

    def eval(self, engine, x):
        return HCTSA_MF_GP_hyperparameters(engine,
                                           x,
                                           covfunc=self.covfunc,
                                           squishorsquash=self.squishorsquash,
                                           maxN=self.maxN,
                                           methds=self.methds)


def HCTSA_MF_ResidualAnalysis(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Given an input residual time series residuals, e, this function exports a
    % structure with fields corresponding to structural tests on the residuals.
    % These are motivated by a general expectation of model residuals to be
    % uncorrelated.
    %
    %---INPUT:
    % e, should be raw residuals as prediction minus data (e = yp - y) as a column
    %       vector. Will take absolute values / even powers of e as necessary.
    % 
    %---HISTORY:
    % Ben Fulcher, 10/2/2010
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'MF_ResidualAnalysis', x, )
    return outfunc(out)


class MF_ResidualAnalysis(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Given an input residual time series residuals, e, this function exports a
    % structure with fields corresponding to structural tests on the residuals.
    % These are motivated by a general expectation of model residuals to be
    % uncorrelated.
    %
    %---INPUT:
    % e, should be raw residuals as prediction minus data (e = yp - y) as a column
    %       vector. Will take absolute values / even powers of e as necessary.
    % 
    %---HISTORY:
    % Ben Fulcher, 10/2/2010
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(MF_ResidualAnalysis, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_MF_ResidualAnalysis(engine, x)


def HCTSA_MF_StateSpaceCompOrder(eng, x, maxorder=8):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits state space models using n4sid (from Matlab's System Identification
    % Toolbox) of orders 1, 2, ..., maxorder and returns statistics on how the
    % goodness of fit changes across this range.
    % 
    % c.f., MF_CompareAR -- does a similar thing for AR models
    % Uses the functions iddata, n4sid, and aic from Matlab's System Identification
    % Toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % maxorder, the maximum model order to consider.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['aic2',
                                                      'aicopt',
                                                      'fpe2',
                                                      'lossfn2',
                                                      'lossfnopt',
                                                      'maxdiffaic',
                                                      'meandiffaic',
                                                      'minaic',
                                                      'mindiffaic',
                                                      'minlossfn',
                                                      'ndownaic']}
    if maxorder is None:
        out = eng.run_function(1, 'MF_StateSpaceCompOrder', x, )
    else:
        out = eng.run_function(1, 'MF_StateSpaceCompOrder', x, maxorder)
    return outfunc(out)


class MF_StateSpaceCompOrder(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits state space models using n4sid (from Matlab's System Identification
    % Toolbox) of orders 1, 2, ..., maxorder and returns statistics on how the
    % goodness of fit changes across this range.
    % 
    % c.f., MF_CompareAR -- does a similar thing for AR models
    % Uses the functions iddata, n4sid, and aic from Matlab's System Identification
    % Toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % maxorder, the maximum model order to consider.
    % 
    ----------------------------------------
    """

    outnames = ('aic2',
                'aicopt',
                'fpe2',
                'lossfn2',
                'lossfnopt',
                'maxdiffaic',
                'meandiffaic',
                'minaic',
                'mindiffaic',
                'minlossfn',
                'ndownaic')

    def __init__(self, maxorder=8):
        super(MF_StateSpaceCompOrder, self).__init__(add_descriptors=False)
        self.maxorder = maxorder

    def eval(self, engine, x):
        return HCTSA_MF_StateSpaceCompOrder(engine,
                                            x,
                                            maxorder=self.maxorder)


def HCTSA_MF_StateSpace_n4sid(eng, x, ord_=3, ptrain=0.5, steps=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a state space model of given order to the time series using the n4sid
    % function in Matlab's System Identification Toolbox.
    % 
    % First fits the model to the whole time series, then trains it on the first
    % portion and tries to predict the rest.
    % 
    % In the second portion of this code, the state space model is fitted to the
    % first p*N samples of the time series, where p is a given proportion and N is
    % the length of the time series.
    % 
    % This model is then used to predict the latter portion of the time
    % series (i.e., the subsequent (1-p)*N samples).
    % 
    % Uses the functions iddata, n4sid, aic, and predict from Matlab's System Identification Toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % ord, the order of state-space model to implement (can also be the string 'best')
    % ptrain, the proportion of the time series to use for training
    % steps, the number of steps ahead to predict
    % 
    %---OUTPUTS: parameters from the model fitted to the entire time series, and
    % goodness of fit and residual analysis from n4sid prediction.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['A_1',
                                                      'A_2',
                                                      'A_3',
                                                      'A_4',
                                                      'A_5',
                                                      'A_6',
                                                      'A_7',
                                                      'A_8',
                                                      'A_9',
                                                      'ac1',
                                                      'ac1diff',
                                                      'ac1n',
                                                      'ac2',
                                                      'ac2n',
                                                      'ac3',
                                                      'ac3n',
                                                      'acmnd0',
                                                      'acsnd0',
                                                      'c_1',
                                                      'c_2',
                                                      'c_3',
                                                      'dwts',
                                                      'ftbth',
                                                      'k_1',
                                                      'k_2',
                                                      'k_3',
                                                      'm_Ts',
                                                      'm_aic',
                                                      'm_fpe',
                                                      'm_lossfn',
                                                      'm_noisevar',
                                                      'maxonmean',
                                                      'meanabs',
                                                      'meane',
                                                      'minfpe',
                                                      'minsbc',
                                                      'mms',
                                                      'normksstat',
                                                      'normp',
                                                      'np',
                                                      'p1_5',
                                                      'p2_5',
                                                      'p3_5',
                                                      'p4_5',
                                                      'p5_5',
                                                      'popt',
                                                      'propbth',
                                                      'rmse',
                                                      'sbc1',
                                                      'stde',
                                                      'x0mod']}
    if ord_ is None:
        out = eng.run_function(1, 'MF_StateSpace_n4sid', x, )
    elif ptrain is None:
        out = eng.run_function(1, 'MF_StateSpace_n4sid', x, ord_)
    elif steps is None:
        out = eng.run_function(1, 'MF_StateSpace_n4sid', x, ord_, ptrain)
    else:
        out = eng.run_function(1, 'MF_StateSpace_n4sid', x, ord_, ptrain, steps)
    return outfunc(out)


class MF_StateSpace_n4sid(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a state space model of given order to the time series using the n4sid
    % function in Matlab's System Identification Toolbox.
    % 
    % First fits the model to the whole time series, then trains it on the first
    % portion and tries to predict the rest.
    % 
    % In the second portion of this code, the state space model is fitted to the
    % first p*N samples of the time series, where p is a given proportion and N is
    % the length of the time series.
    % 
    % This model is then used to predict the latter portion of the time
    % series (i.e., the subsequent (1-p)*N samples).
    % 
    % Uses the functions iddata, n4sid, aic, and predict from Matlab's System Identification Toolbox
    % 
    %---INPUTS:
    % y, the input time series
    % ord, the order of state-space model to implement (can also be the string 'best')
    % ptrain, the proportion of the time series to use for training
    % steps, the number of steps ahead to predict
    % 
    %---OUTPUTS: parameters from the model fitted to the entire time series, and
    % goodness of fit and residual analysis from n4sid prediction.
    % 
    ----------------------------------------
    """

    outnames = ('A_1',
                'A_2',
                'A_3',
                'A_4',
                'A_5',
                'A_6',
                'A_7',
                'A_8',
                'A_9',
                'ac1',
                'ac1diff',
                'ac1n',
                'ac2',
                'ac2n',
                'ac3',
                'ac3n',
                'acmnd0',
                'acsnd0',
                'c_1',
                'c_2',
                'c_3',
                'dwts',
                'ftbth',
                'k_1',
                'k_2',
                'k_3',
                'm_Ts',
                'm_aic',
                'm_fpe',
                'm_lossfn',
                'm_noisevar',
                'maxonmean',
                'meanabs',
                'meane',
                'minfpe',
                'minsbc',
                'mms',
                'normksstat',
                'normp',
                'np',
                'p1_5',
                'p2_5',
                'p3_5',
                'p4_5',
                'p5_5',
                'popt',
                'propbth',
                'rmse',
                'sbc1',
                'stde',
                'x0mod')

    def __init__(self, ord_=3, ptrain=0.5, steps=1):
        super(MF_StateSpace_n4sid, self).__init__(add_descriptors=False)
        self.ord_ = ord_
        self.ptrain = ptrain
        self.steps = steps

    def eval(self, engine, x):
        return HCTSA_MF_StateSpace_n4sid(engine,
                                         x,
                                         ord_=self.ord_,
                                         ptrain=self.ptrain,
                                         steps=self.steps)


def HCTSA_MF_arfit(eng, x, pmin=1, pmax=8, selector='sbc'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fit an AR model to the time series then returns statistics about it.
    % 
    % Uses various functions implemented in the ARfit package, which is
    % freely-available at http://www.gps.caltech.edu/~tapio/arfit/
    % 
    % cf. "Estimation of parameters and eigenmodes of multivariate autoregressive
    %       models", A. Neumaier and T. Schneider, ACM Trans. Math. Softw. 27, 27 (2001)
    % 
    % cf. "Algorithm 808: ARFIT---a Matlab package for the estimation of parameters
    %      and eigenmodes of multivariate autoregressive models",
    %      T. Schneider and A. Neumaier, ACM Trans. Math. Softw. 27, 58 (2001)
    % 
    % Autoregressive (AR) models are fitted with orders p = pmin, pmin + 1, ..., pmax.
    % 
    % The optimal model order is selected using Schwartz's Bayesian Criterion (SBC).
    % 
    %---INPUTS:
    % y, the input time series
    % pmin, the minimum AR model order to fit
    % pmax, the maximum AR model order to fit
    % selector, crierion to select optimal time-series model order (e.g., 'sbc', cf.
    %           ARFIT package documentation)
    % 
    %---OUTPUTS: include the model coefficients obtained, the SBCs at each model order,
    % various tests on residuals, and statistics from an eigendecomposition of the
    % time series using the estimated AR model.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['A1',
                                                      'A2',
                                                      'A3',
                                                      'A4',
                                                      'A5',
                                                      'A6',
                                                      'C',
                                                      'aerr_max',
                                                      'aerr_mean',
                                                      'aerr_min',
                                                      'aroundmin_fpe',
                                                      'aroundmin_sbc',
                                                      'fpe_1',
                                                      'fpe_2',
                                                      'fpe_3',
                                                      'fpe_4',
                                                      'fpe_5',
                                                      'fpe_6',
                                                      'fpe_7',
                                                      'fpe_8',
                                                      'maxA',
                                                      'maxImS',
                                                      'maxReS',
                                                      'maxabsS',
                                                      'maxexctn',
                                                      'maxper',
                                                      'maxtau',
                                                      'meanA',
                                                      'meanexctn',
                                                      'meanper',
                                                      'meanpererr',
                                                      'meantau',
                                                      'meantauerr',
                                                      'minA',
                                                      'minexctn',
                                                      'minfpe',
                                                      'minper',
                                                      'minsbc',
                                                      'mintau',
                                                      'pcorr_res',
                                                      'popt_fpe',
                                                      'popt_sbc',
                                                      'res_ac1',
                                                      'res_ac1_norm',
                                                      'res_siglev',
                                                      'rmsA',
                                                      'sbc_1',
                                                      'sbc_2',
                                                      'sbc_3',
                                                      'sbc_4',
                                                      'sbc_5',
                                                      'sbc_6',
                                                      'sbc_7',
                                                      'sbc_8',
                                                      'stdA',
                                                      'stdabsS',
                                                      'stdexctn',
                                                      'stdper',
                                                      'stdtau',
                                                      'sumA',
                                                      'sumsqA']}
    if pmin is None:
        out = eng.run_function(1, 'MF_arfit', x, )
    elif pmax is None:
        out = eng.run_function(1, 'MF_arfit', x, pmin)
    elif selector is None:
        out = eng.run_function(1, 'MF_arfit', x, pmin, pmax)
    else:
        out = eng.run_function(1, 'MF_arfit', x, pmin, pmax, selector)
    return outfunc(out)


class MF_arfit(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fit an AR model to the time series then returns statistics about it.
    % 
    % Uses various functions implemented in the ARfit package, which is
    % freely-available at http://www.gps.caltech.edu/~tapio/arfit/
    % 
    % cf. "Estimation of parameters and eigenmodes of multivariate autoregressive
    %       models", A. Neumaier and T. Schneider, ACM Trans. Math. Softw. 27, 27 (2001)
    % 
    % cf. "Algorithm 808: ARFIT---a Matlab package for the estimation of parameters
    %      and eigenmodes of multivariate autoregressive models",
    %      T. Schneider and A. Neumaier, ACM Trans. Math. Softw. 27, 58 (2001)
    % 
    % Autoregressive (AR) models are fitted with orders p = pmin, pmin + 1, ..., pmax.
    % 
    % The optimal model order is selected using Schwartz's Bayesian Criterion (SBC).
    % 
    %---INPUTS:
    % y, the input time series
    % pmin, the minimum AR model order to fit
    % pmax, the maximum AR model order to fit
    % selector, crierion to select optimal time-series model order (e.g., 'sbc', cf.
    %           ARFIT package documentation)
    % 
    %---OUTPUTS: include the model coefficients obtained, the SBCs at each model order,
    % various tests on residuals, and statistics from an eigendecomposition of the
    % time series using the estimated AR model.
    % 
    ----------------------------------------
    """

    outnames = ('A1',
                'A2',
                'A3',
                'A4',
                'A5',
                'A6',
                'C',
                'aerr_max',
                'aerr_mean',
                'aerr_min',
                'aroundmin_fpe',
                'aroundmin_sbc',
                'fpe_1',
                'fpe_2',
                'fpe_3',
                'fpe_4',
                'fpe_5',
                'fpe_6',
                'fpe_7',
                'fpe_8',
                'maxA',
                'maxImS',
                'maxReS',
                'maxabsS',
                'maxexctn',
                'maxper',
                'maxtau',
                'meanA',
                'meanexctn',
                'meanper',
                'meanpererr',
                'meantau',
                'meantauerr',
                'minA',
                'minexctn',
                'minfpe',
                'minper',
                'minsbc',
                'mintau',
                'pcorr_res',
                'popt_fpe',
                'popt_sbc',
                'res_ac1',
                'res_ac1_norm',
                'res_siglev',
                'rmsA',
                'sbc_1',
                'sbc_2',
                'sbc_3',
                'sbc_4',
                'sbc_5',
                'sbc_6',
                'sbc_7',
                'sbc_8',
                'stdA',
                'stdabsS',
                'stdexctn',
                'stdper',
                'stdtau',
                'sumA',
                'sumsqA')

    def __init__(self, pmin=1, pmax=8, selector='sbc'):
        super(MF_arfit, self).__init__(add_descriptors=False)
        self.pmin = pmin
        self.pmax = pmax
        self.selector = selector

    def eval(self, engine, x):
        return HCTSA_MF_arfit(engine,
                              x,
                              pmin=self.pmin,
                              pmax=self.pmax,
                              selector=self.selector)


def HCTSA_MF_armax(eng, x, orders=(2, 2), ptrain=0.5, nsteps=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits an ARMA(p,q) model to the time series and returns various statistics on
    % the result.
    % 
    % Uses the functions iddata, armax, aic, and predict from Matlab's System
    % Identification Toolbox
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % orders, a two-vector for p and q, the AR and MA components of the model,
    %           respectively,
    % 
    % ptrain, the proportion of data to train the model on (the remainder is used
    %           for testing),
    % 
    % nsteps, number of steps to predict into the future for testing the model.
    % 
    % 
    %---OUTPUTS: include the fitted AR and MA coefficients, the goodness of fit in
    % the training data, and statistics on the residuals from using the fitted model
    % to predict the testing data.
    % 
    %---HISTORY:
    % Ben Fulcher, 1/2/2010
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['AR_1',
                                                      'AR_2',
                                                      'AR_3',
                                                      'MA_1',
                                                      'MA_2',
                                                      'ac1',
                                                      'ac1n',
                                                      'ac2',
                                                      'ac2n',
                                                      'ac3',
                                                      'ac3n',
                                                      'acmnd0',
                                                      'acsnd0',
                                                      'aic',
                                                      'dwts',
                                                      'fpe',
                                                      'ftbth',
                                                      'lastimprovement',
                                                      'lossfn',
                                                      'maxda',
                                                      'maxdc',
                                                      'maxonmean',
                                                      'meanabs',
                                                      'meane',
                                                      'minfpe',
                                                      'minsbc',
                                                      'mms',
                                                      'noisevar',
                                                      'normksstat',
                                                      'normp',
                                                      'p1_5',
                                                      'p2_5',
                                                      'p3_5',
                                                      'p4_5',
                                                      'p5_5',
                                                      'popt',
                                                      'propbth',
                                                      'rmse',
                                                      'sbc1',
                                                      'stde']}
    if orders is None:
        out = eng.run_function(1, 'MF_armax', x, )
    elif ptrain is None:
        out = eng.run_function(1, 'MF_armax', x, orders)
    elif nsteps is None:
        out = eng.run_function(1, 'MF_armax', x, orders, ptrain)
    else:
        out = eng.run_function(1, 'MF_armax', x, orders, ptrain, nsteps)
    return outfunc(out)


class MF_armax(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits an ARMA(p,q) model to the time series and returns various statistics on
    % the result.
    % 
    % Uses the functions iddata, armax, aic, and predict from Matlab's System
    % Identification Toolbox
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % orders, a two-vector for p and q, the AR and MA components of the model,
    %           respectively,
    % 
    % ptrain, the proportion of data to train the model on (the remainder is used
    %           for testing),
    % 
    % nsteps, number of steps to predict into the future for testing the model.
    % 
    % 
    %---OUTPUTS: include the fitted AR and MA coefficients, the goodness of fit in
    % the training data, and statistics on the residuals from using the fitted model
    % to predict the testing data.
    % 
    %---HISTORY:
    % Ben Fulcher, 1/2/2010
    ----------------------------------------
    """

    outnames = ('AR_1',
                'AR_2',
                'AR_3',
                'MA_1',
                'MA_2',
                'ac1',
                'ac1n',
                'ac2',
                'ac2n',
                'ac3',
                'ac3n',
                'acmnd0',
                'acsnd0',
                'aic',
                'dwts',
                'fpe',
                'ftbth',
                'lastimprovement',
                'lossfn',
                'maxda',
                'maxdc',
                'maxonmean',
                'meanabs',
                'meane',
                'minfpe',
                'minsbc',
                'mms',
                'noisevar',
                'normksstat',
                'normp',
                'p1_5',
                'p2_5',
                'p3_5',
                'p4_5',
                'p5_5',
                'popt',
                'propbth',
                'rmse',
                'sbc1',
                'stde')

    def __init__(self, orders=(2, 2), ptrain=0.5, nsteps=1):
        super(MF_armax, self).__init__(add_descriptors=False)
        self.orders = orders
        self.ptrain = ptrain
        self.nsteps = nsteps

    def eval(self, engine, x):
        return HCTSA_MF_armax(engine,
                              x,
                              orders=self.orders,
                              ptrain=self.ptrain,
                              nsteps=self.nsteps)


def HCTSA_MF_hmm_CompareNStates(eng, x, trainp=0.6, nstater=MatlabSequence('2:4')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits HMMs with different numbers of states, and compares the resulting
    % test-set likelihoods.
    % 
    % The code relies on Zoubin Gharamani's implementation of HMMs for real-valued
    % Gassian-distributed observations, including the hmm and hmm_cl routines (
    % renamed ZG_hmm and ZG_hmm_cl here).
    % Implementation of HMMs for real-valued Gaussian observations:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
    % or, specifically:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % trainp, the initial proportion of the time series to train the model on
    % 
    % nstater, the vector of state numbers to compare. E.g., (2:4) compares a number
    %               of states 2, 3, and 4.
    % 
    %---OUTPUTS: statistics on how the log likelihood of the test data changes with
    % the number of states n_{states}$. We implement the code for p_{train} = 0.6$
    % as n_{states}$ varies across the range n_{states} = 2, 3, 4$.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['LLtestdiff1',
                                                      'LLtestdiff2',
                                                      'chLLtest',
                                                      'chLLtrain',
                                                      'maxLLtest',
                                                      'maxLLtrain',
                                                      'meanLLtest',
                                                      'meanLLtrain',
                                                      'meandiffLLtt']}
    if trainp is None:
        out = eng.run_function(1, 'MF_hmm_CompareNStates', x, )
    elif nstater is None:
        out = eng.run_function(1, 'MF_hmm_CompareNStates', x, trainp)
    else:
        out = eng.run_function(1, 'MF_hmm_CompareNStates', x, trainp, nstater)
    return outfunc(out)


class MF_hmm_CompareNStates(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits HMMs with different numbers of states, and compares the resulting
    % test-set likelihoods.
    % 
    % The code relies on Zoubin Gharamani's implementation of HMMs for real-valued
    % Gassian-distributed observations, including the hmm and hmm_cl routines (
    % renamed ZG_hmm and ZG_hmm_cl here).
    % Implementation of HMMs for real-valued Gaussian observations:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
    % or, specifically:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % trainp, the initial proportion of the time series to train the model on
    % 
    % nstater, the vector of state numbers to compare. E.g., (2:4) compares a number
    %               of states 2, 3, and 4.
    % 
    %---OUTPUTS: statistics on how the log likelihood of the test data changes with
    % the number of states n_{states}$. We implement the code for p_{train} = 0.6$
    % as n_{states}$ varies across the range n_{states} = 2, 3, 4$.
    % 
    ----------------------------------------
    """

    outnames = ('LLtestdiff1',
                'LLtestdiff2',
                'chLLtest',
                'chLLtrain',
                'maxLLtest',
                'maxLLtrain',
                'meanLLtest',
                'meanLLtrain',
                'meandiffLLtt')

    def __init__(self, trainp=0.6, nstater=MatlabSequence('2:4')):
        super(MF_hmm_CompareNStates, self).__init__(add_descriptors=False)
        self.trainp = trainp
        self.nstater = nstater

    def eval(self, engine, x):
        return HCTSA_MF_hmm_CompareNStates(engine,
                                           x,
                                           trainp=self.trainp,
                                           nstater=self.nstater)


def HCTSA_MF_hmm_fit(eng, x, trainp=0.7, nstates=3):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses Zoubin Gharamani's implementation of HMMs for real-valued Gaussian
    % observations:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
    % or, specifically:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
    % 
    % Uses ZG_hmm (renamed from hmm) and ZG_hmm_cl (renamed from hmm_cl)
    % 
    %---INPUTS:
    % y, the input time series
    % trainp, the proportion of data to train on, 0 < trainp < 1
    % nstates, the number of states in the HMM
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['Cov',
                                                      'LLdifference',
                                                      'LLtestpersample',
                                                      'LLtrainpersample',
                                                      'Mu_1',
                                                      'Mu_2',
                                                      'Mu_3',
                                                      'Pmeandiag',
                                                      'maxP',
                                                      'meanMu',
                                                      'meanP',
                                                      'nit',
                                                      'rangeMu',
                                                      'stdP',
                                                      'stdmeanP']}
    if trainp is None:
        out = eng.run_function(1, 'MF_hmm_fit', x, )
    elif nstates is None:
        out = eng.run_function(1, 'MF_hmm_fit', x, trainp)
    else:
        out = eng.run_function(1, 'MF_hmm_fit', x, trainp, nstates)
    return outfunc(out)


class MF_hmm_fit(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses Zoubin Gharamani's implementation of HMMs for real-valued Gaussian
    % observations:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software.html
    % or, specifically:
    % http://www.gatsby.ucl.ac.uk/~zoubin/software/hmm.tar.gz
    % 
    % Uses ZG_hmm (renamed from hmm) and ZG_hmm_cl (renamed from hmm_cl)
    % 
    %---INPUTS:
    % y, the input time series
    % trainp, the proportion of data to train on, 0 < trainp < 1
    % nstates, the number of states in the HMM
    % 
    ----------------------------------------
    """

    outnames = ('Cov',
                'LLdifference',
                'LLtestpersample',
                'LLtrainpersample',
                'Mu_1',
                'Mu_2',
                'Mu_3',
                'Pmeandiag',
                'maxP',
                'meanMu',
                'meanP',
                'nit',
                'rangeMu',
                'stdP',
                'stdmeanP')

    def __init__(self, trainp=0.7, nstates=3):
        super(MF_hmm_fit, self).__init__(add_descriptors=False)
        self.trainp = trainp
        self.nstates = nstates

    def eval(self, engine, x):
        return HCTSA_MF_hmm_fit(engine,
                                x,
                                trainp=self.trainp,
                                nstates=self.nstates)


def HCTSA_MF_steps_ahead(eng, x, model='ar', order='best', maxsteps=6):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Given a model, characterizes the variation in goodness of model predictions
    % across a range of prediction lengths, l, which is made to vary from
    % 1-step ahead to maxsteps steps-ahead predictions.
    % 
    % Models are fit using code from Matlab's System Identification Toolbox:
    % (i) AR models using the ar function,
    % (ii) ARMA models using armax code, and
    % (iii) state-space models using n4sid code.
    % 
    % The model is fitted on the full time series and then used to predict the same
    % data.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % model, the time-series model to fit: 'ar', 'arma', or 'ss'
    % 
    % order, the order of the model to fit
    % 
    % maxsteps, the maximum number of steps ahead to predict
    % 
    %---OUTPUTS: include the errors, for prediction lengths l = 1, 2, ..., maxsteps,
    % returned for each model relative to the best performance from basic null
    % predictors, including sliding 1- and 2-sample mean predictors and simply
    % predicting each point as the mean of the full time series. Additional outputs
    % quantify how the errors change as the prediction length increases from l = 1,
    % ..., maxsteps (relative to a simple predictor).
    %
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1_1',
                                                      'ac1_2',
                                                      'ac1_3',
                                                      'ac1_4',
                                                      'ac1_5',
                                                      'ac1_6',
                                                      'mabserr_1',
                                                      'mabserr_2',
                                                      'mabserr_3',
                                                      'mabserr_4',
                                                      'mabserr_5',
                                                      'mabserr_6',
                                                      'maxdiffrms',
                                                      'meandiffrms',
                                                      'meandiffrmsabs',
                                                      'ndown',
                                                      'rmserr_1',
                                                      'rmserr_2',
                                                      'rmserr_3',
                                                      'rmserr_4',
                                                      'rmserr_5',
                                                      'rmserr_6',
                                                      'stddiffrms']}
    if model is None:
        out = eng.run_function(1, 'MF_steps_ahead', x, )
    elif order is None:
        out = eng.run_function(1, 'MF_steps_ahead', x, model)
    elif maxsteps is None:
        out = eng.run_function(1, 'MF_steps_ahead', x, model, order)
    else:
        out = eng.run_function(1, 'MF_steps_ahead', x, model, order, maxsteps)
    return outfunc(out)


class MF_steps_ahead(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Given a model, characterizes the variation in goodness of model predictions
    % across a range of prediction lengths, l, which is made to vary from
    % 1-step ahead to maxsteps steps-ahead predictions.
    % 
    % Models are fit using code from Matlab's System Identification Toolbox:
    % (i) AR models using the ar function,
    % (ii) ARMA models using armax code, and
    % (iii) state-space models using n4sid code.
    % 
    % The model is fitted on the full time series and then used to predict the same
    % data.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % model, the time-series model to fit: 'ar', 'arma', or 'ss'
    % 
    % order, the order of the model to fit
    % 
    % maxsteps, the maximum number of steps ahead to predict
    % 
    %---OUTPUTS: include the errors, for prediction lengths l = 1, 2, ..., maxsteps,
    % returned for each model relative to the best performance from basic null
    % predictors, including sliding 1- and 2-sample mean predictors and simply
    % predicting each point as the mean of the full time series. Additional outputs
    % quantify how the errors change as the prediction length increases from l = 1,
    % ..., maxsteps (relative to a simple predictor).
    %
    ----------------------------------------
    """

    outnames = ('ac1_1',
                'ac1_2',
                'ac1_3',
                'ac1_4',
                'ac1_5',
                'ac1_6',
                'mabserr_1',
                'mabserr_2',
                'mabserr_3',
                'mabserr_4',
                'mabserr_5',
                'mabserr_6',
                'maxdiffrms',
                'meandiffrms',
                'meandiffrmsabs',
                'ndown',
                'rmserr_1',
                'rmserr_2',
                'rmserr_3',
                'rmserr_4',
                'rmserr_5',
                'rmserr_6',
                'stddiffrms')

    def __init__(self, model='ar', order='best', maxsteps=6):
        super(MF_steps_ahead, self).__init__(add_descriptors=False)
        self.model = model
        self.order = order
        self.maxsteps = maxsteps

    def eval(self, engine, x):
        return HCTSA_MF_steps_ahead(engine,
                                    x,
                                    model=self.model,
                                    order=self.order,
                                    maxsteps=self.maxsteps)


def HCTSA_NL_BoxCorrDim(eng, x, nbins=50, embedparams=('ac', 5)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % References TSTOOL code, corrdim, to estimate the correlation dimension of a
    % time-delay embedded time series using a box-counting approach.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time series data
    % 
    % nbins, maximum number of partitions per axis
    % 
    % embedparams [opt], embedding parameters as {tau,m} in 2-entry cell for a
    %                   time-delay, tau, and embedding dimension, m. As inputs to BF_embed.
    % 
    %---OUTPUTS: Simple summaries of the outputs from corrdim.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['iqrstretch',
                                                      'meanchr10',
                                                      'meanchr11',
                                                      'meanchr12',
                                                      'meanchr13',
                                                      'meanchr14',
                                                      'meanchr15',
                                                      'meanchr16',
                                                      'meanchr17',
                                                      'meanchr18',
                                                      'meanchr2',
                                                      'meanchr3',
                                                      'meanchr4',
                                                      'meanchr5',
                                                      'meanchr6',
                                                      'meanchr7',
                                                      'meanchr8',
                                                      'meanchr9',
                                                      'meand2',
                                                      'meand3',
                                                      'meand4',
                                                      'meand5',
                                                      'meanr10',
                                                      'meanr11',
                                                      'meanr12',
                                                      'meanr13',
                                                      'meanr14',
                                                      'meanr15',
                                                      'meanr16',
                                                      'meanr17',
                                                      'meanr18',
                                                      'meanr2',
                                                      'meanr3',
                                                      'meanr4',
                                                      'meanr5',
                                                      'meanr6',
                                                      'meanr7',
                                                      'meanr8',
                                                      'meanr9',
                                                      'mediand2',
                                                      'mediand3',
                                                      'mediand4',
                                                      'mediand5',
                                                      'medianr10',
                                                      'medianr11',
                                                      'medianr12',
                                                      'medianr13',
                                                      'medianr14',
                                                      'medianr15',
                                                      'medianr16',
                                                      'medianr17',
                                                      'medianr18',
                                                      'medianr2',
                                                      'medianr3',
                                                      'medianr4',
                                                      'medianr5',
                                                      'medianr6',
                                                      'medianr7',
                                                      'medianr8',
                                                      'medianr9',
                                                      'medianstretch',
                                                      'mind2',
                                                      'mind3',
                                                      'mind4',
                                                      'mind5',
                                                      'minr10',
                                                      'minr11',
                                                      'minr12',
                                                      'minr13',
                                                      'minr14',
                                                      'minr15',
                                                      'minr16',
                                                      'minr17',
                                                      'minr18',
                                                      'minr2',
                                                      'minr3',
                                                      'minr4',
                                                      'minr5',
                                                      'minr6',
                                                      'minr7',
                                                      'minr8',
                                                      'minr9',
                                                      'minstretch',
                                                      'stdmean',
                                                      'stdmedian']}
    if nbins is None:
        out = eng.run_function(1, 'NL_BoxCorrDim', x, )
    elif embedparams is None:
        out = eng.run_function(1, 'NL_BoxCorrDim', x, nbins)
    else:
        out = eng.run_function(1, 'NL_BoxCorrDim', x, nbins, embedparams)
    return outfunc(out)


class NL_BoxCorrDim(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % References TSTOOL code, corrdim, to estimate the correlation dimension of a
    % time-delay embedded time series using a box-counting approach.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time series data
    % 
    % nbins, maximum number of partitions per axis
    % 
    % embedparams [opt], embedding parameters as {tau,m} in 2-entry cell for a
    %                   time-delay, tau, and embedding dimension, m. As inputs to BF_embed.
    % 
    %---OUTPUTS: Simple summaries of the outputs from corrdim.
    % 
    ----------------------------------------
    """

    outnames = ('iqrstretch',
                'meanchr10',
                'meanchr11',
                'meanchr12',
                'meanchr13',
                'meanchr14',
                'meanchr15',
                'meanchr16',
                'meanchr17',
                'meanchr18',
                'meanchr2',
                'meanchr3',
                'meanchr4',
                'meanchr5',
                'meanchr6',
                'meanchr7',
                'meanchr8',
                'meanchr9',
                'meand2',
                'meand3',
                'meand4',
                'meand5',
                'meanr10',
                'meanr11',
                'meanr12',
                'meanr13',
                'meanr14',
                'meanr15',
                'meanr16',
                'meanr17',
                'meanr18',
                'meanr2',
                'meanr3',
                'meanr4',
                'meanr5',
                'meanr6',
                'meanr7',
                'meanr8',
                'meanr9',
                'mediand2',
                'mediand3',
                'mediand4',
                'mediand5',
                'medianr10',
                'medianr11',
                'medianr12',
                'medianr13',
                'medianr14',
                'medianr15',
                'medianr16',
                'medianr17',
                'medianr18',
                'medianr2',
                'medianr3',
                'medianr4',
                'medianr5',
                'medianr6',
                'medianr7',
                'medianr8',
                'medianr9',
                'medianstretch',
                'mind2',
                'mind3',
                'mind4',
                'mind5',
                'minr10',
                'minr11',
                'minr12',
                'minr13',
                'minr14',
                'minr15',
                'minr16',
                'minr17',
                'minr18',
                'minr2',
                'minr3',
                'minr4',
                'minr5',
                'minr6',
                'minr7',
                'minr8',
                'minr9',
                'minstretch',
                'stdmean',
                'stdmedian')

    def __init__(self, nbins=50, embedparams=('ac', 5)):
        super(NL_BoxCorrDim, self).__init__(add_descriptors=False)
        self.nbins = nbins
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_BoxCorrDim(engine,
                                   x,
                                   nbins=self.nbins,
                                   embedparams=self.embedparams)


def HCTSA_NL_CaosMethod(eng, x, maxdim=10, tau='mi', NNR=2, Nref=100, justanum=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % References TSTOOL code cao, to determine the minimum embedding dimension for a
    % time series using Cao's method:
    % 
    % "Practical method for determining the minimum embedding dimension of a scalar
    % time series", L. Cao, Physica D 110(1-2) 43 (1997)
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % It computes the quantities E and E* for a range of embedding dimensions
    % m = 1, ..., m_{max}.
    % 
    %---INPUTS:
    % 
    % y, time series as a column vector
    % 
    % maxdim, maximum embedding dimension to consider
    % 
    % tau, time delay (can also be 'ac' or 'mi' for first zero-crossing of the
    %          autocorrelation function or the first minimum of the automutual information
    %          function)
    %          
    % NNR, number of nearest neighbours to use
    % 
    % Nref, number of reference points (can also be a fraction; of data length)
    % 
    % justanum [opt]: if not empty can just return a number, the embedding
    %                   dimension, based on the specified criterion:
    %                 (i) 'thresh', caoo1 passes above a threshold, th (given as
    %                               justanum = {'thresh',th})
    %                 (ii) 'mthresh', when gradient passes below a threshold (levels
    %                                 off), given as {'mthresh',mthresh}, where
    %                                 mthresh in the threshold
    %                 (iii) 'mmthresh', analyzes incremental differences to find
    %                                   level-off point
    %   
    % 
    %---OUTPUTS: statistics on the result, including when the output quantity first
    % passes a given threshold, and the m at which it levels off.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['caoo1_1',
                                                      'caoo1_10',
                                                      'caoo1_2',
                                                      'caoo1_3',
                                                      'caoo1_4',
                                                      'caoo1_5',
                                                      'caoo1_6',
                                                      'caoo1_7',
                                                      'caoo1_8',
                                                      'caoo1_9',
                                                      'caoo2_1',
                                                      'caoo2_10',
                                                      'caoo2_2',
                                                      'caoo2_3',
                                                      'caoo2_4',
                                                      'caoo2_5',
                                                      'caoo2_6',
                                                      'caoo2_7',
                                                      'caoo2_8',
                                                      'caoo2_9',
                                                      'fm005_1',
                                                      'fm005_2',
                                                      'fm01_1',
                                                      'fm01_2',
                                                      'fm02_1',
                                                      'fm02_2',
                                                      'fmm10_1',
                                                      'fmm10_2',
                                                      'fmm20_1',
                                                      'fmm20_2',
                                                      'fmm40_1',
                                                      'fmm40_2',
                                                      'fmmmax_1',
                                                      'fmmmax_2',
                                                      'fp05_1',
                                                      'fp05_2',
                                                      'fp08_1',
                                                      'fp08_2',
                                                      'fp09_1',
                                                      'fp09_2',
                                                      'max1',
                                                      'max2',
                                                      'median1',
                                                      'median2',
                                                      'min1',
                                                      'min2',
                                                      'std1',
                                                      'std2']}
    if maxdim is None:
        out = eng.run_function(1, 'NL_CaosMethod', x, )
    elif tau is None:
        out = eng.run_function(1, 'NL_CaosMethod', x, maxdim)
    elif NNR is None:
        out = eng.run_function(1, 'NL_CaosMethod', x, maxdim, tau)
    elif Nref is None:
        out = eng.run_function(1, 'NL_CaosMethod', x, maxdim, tau, NNR)
    elif justanum is None:
        out = eng.run_function(1, 'NL_CaosMethod', x, maxdim, tau, NNR, Nref)
    else:
        out = eng.run_function(1, 'NL_CaosMethod', x, maxdim, tau, NNR, Nref, justanum)
    return outfunc(out)


class NL_CaosMethod(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % References TSTOOL code cao, to determine the minimum embedding dimension for a
    % time series using Cao's method:
    % 
    % "Practical method for determining the minimum embedding dimension of a scalar
    % time series", L. Cao, Physica D 110(1-2) 43 (1997)
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % It computes the quantities E and E* for a range of embedding dimensions
    % m = 1, ..., m_{max}.
    % 
    %---INPUTS:
    % 
    % y, time series as a column vector
    % 
    % maxdim, maximum embedding dimension to consider
    % 
    % tau, time delay (can also be 'ac' or 'mi' for first zero-crossing of the
    %          autocorrelation function or the first minimum of the automutual information
    %          function)
    %          
    % NNR, number of nearest neighbours to use
    % 
    % Nref, number of reference points (can also be a fraction; of data length)
    % 
    % justanum [opt]: if not empty can just return a number, the embedding
    %                   dimension, based on the specified criterion:
    %                 (i) 'thresh', caoo1 passes above a threshold, th (given as
    %                               justanum = {'thresh',th})
    %                 (ii) 'mthresh', when gradient passes below a threshold (levels
    %                                 off), given as {'mthresh',mthresh}, where
    %                                 mthresh in the threshold
    %                 (iii) 'mmthresh', analyzes incremental differences to find
    %                                   level-off point
    %   
    % 
    %---OUTPUTS: statistics on the result, including when the output quantity first
    % passes a given threshold, and the m at which it levels off.
    % 
    ----------------------------------------
    """

    outnames = ('caoo1_1',
                'caoo1_10',
                'caoo1_2',
                'caoo1_3',
                'caoo1_4',
                'caoo1_5',
                'caoo1_6',
                'caoo1_7',
                'caoo1_8',
                'caoo1_9',
                'caoo2_1',
                'caoo2_10',
                'caoo2_2',
                'caoo2_3',
                'caoo2_4',
                'caoo2_5',
                'caoo2_6',
                'caoo2_7',
                'caoo2_8',
                'caoo2_9',
                'fm005_1',
                'fm005_2',
                'fm01_1',
                'fm01_2',
                'fm02_1',
                'fm02_2',
                'fmm10_1',
                'fmm10_2',
                'fmm20_1',
                'fmm20_2',
                'fmm40_1',
                'fmm40_2',
                'fmmmax_1',
                'fmmmax_2',
                'fp05_1',
                'fp05_2',
                'fp08_1',
                'fp08_2',
                'fp09_1',
                'fp09_2',
                'max1',
                'max2',
                'median1',
                'median2',
                'min1',
                'min2',
                'std1',
                'std2')

    def __init__(self, maxdim=10, tau='mi', NNR=2, Nref=100, justanum=None):
        super(NL_CaosMethod, self).__init__(add_descriptors=False)
        self.maxdim = maxdim
        self.tau = tau
        self.NNR = NNR
        self.Nref = Nref
        self.justanum = justanum

    def eval(self, engine, x):
        return HCTSA_NL_CaosMethod(engine,
                                   x,
                                   maxdim=self.maxdim,
                                   tau=self.tau,
                                   NNR=self.NNR,
                                   Nref=self.Nref,
                                   justanum=self.justanum)


def HCTSA_NL_MS_LZcomplexity(eng, x, n=7, preproc='diff'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the Lempel-Ziv complexity of a n-bit encoding of the time
    % series using Michael Small's code: 'complexity' (renamed MS_complexity here).
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Code is available at http://small.eie.polyu.edu.hk/matlab/
    % 
    % The code is a wrapper for Michael Small's original code and uses the
    % associated mex file compiled from complexitybs.c (renamed MS_complexitybs.c
    % here).
    % 
    %---INPUTS:
    % y, the input time series
    % n, the (integer) number of bits to encode the data into
    % preproc [opt], first apply a given preprocessing to the time series. For now,
    %               just 'diff' is implemented, which zscores incremental
    %               differences and then applies the complexity method.
    % 
    %---OUTPUT: the normalized Lempel-Ziv complexity: i.e., the number of distinct
    %           symbol sequences in the time series divided by the expected number
    %           of distinct symbols for a noise sequence.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if n is None:
        out = eng.run_function(1, 'NL_MS_LZcomplexity', x, )
    elif preproc is None:
        out = eng.run_function(1, 'NL_MS_LZcomplexity', x, n)
    else:
        out = eng.run_function(1, 'NL_MS_LZcomplexity', x, n, preproc)
    return outfunc(out)


class NL_MS_LZcomplexity(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the Lempel-Ziv complexity of a n-bit encoding of the time
    % series using Michael Small's code: 'complexity' (renamed MS_complexity here).
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Code is available at http://small.eie.polyu.edu.hk/matlab/
    % 
    % The code is a wrapper for Michael Small's original code and uses the
    % associated mex file compiled from complexitybs.c (renamed MS_complexitybs.c
    % here).
    % 
    %---INPUTS:
    % y, the input time series
    % n, the (integer) number of bits to encode the data into
    % preproc [opt], first apply a given preprocessing to the time series. For now,
    %               just 'diff' is implemented, which zscores incremental
    %               differences and then applies the complexity method.
    % 
    %---OUTPUT: the normalized Lempel-Ziv complexity: i.e., the number of distinct
    %           symbol sequences in the time series divided by the expected number
    %           of distinct symbols for a noise sequence.
    % 
    ----------------------------------------
    """

    def __init__(self, n=7, preproc='diff'):
        super(NL_MS_LZcomplexity, self).__init__(add_descriptors=False)
        self.n = n
        self.preproc = preproc

    def eval(self, engine, x):
        return HCTSA_NL_MS_LZcomplexity(engine,
                                        x,
                                        n=self.n,
                                        preproc=self.preproc)


def HCTSA_NL_MS_fnn(eng, x, de=MatlabSequence('1:10'), tau='mi', th=5, kth=1, justbest=None, bestp=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Determines the number of false nearest neighbors for the embedded time series
    % using Michael Small's false nearest neighbor code, fnn (renamed MS_fnn here)
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Code available at http://small.eie.polyu.edu.hk/matlab/
    % 
    % False nearest neighbors are judged using a ratio of the distances between the
    % next k points and the neighboring points of a given datapoint.
    % 
    %---INPUTS:
    % y, the input time series
    % de, the embedding dimensions to compare across (a vector)
    % tau, the time-delay (can be 'ac' or 'mi' to be the first zero-crossing of ACF,
    %                       or first minimum of AMI, respectively)
    % th, the distance threshold for neighbours
    % kth, the the distance to next points
    % [opt] justbest, can be set to 1 to just return the best embedding dimension, m_{best}
    % [opt] bestp, if justbest = 1, can set bestp as the proportion of false nearest
    %              neighbours at which the optimal embedding dimension is selected.
    % 
    % This function returns statistics on the proportion of false nearest neighbors
    % as a function of the embedding dimension m = m_{min}, m_{min}+1, ..., m_{max}
    % for a given time lag tau, and distance threshold for neighbors, d_{th}.
    % 
    %---OUTPUTS: include the proportion of false nearest neighbors at each m, the mean
    % and spread, and the smallest m at which the proportion of false nearest
    % neighbors drops below each of a set of fixed thresholds.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['firstunder001',
                                                      'firstunder002',
                                                      'firstunder005',
                                                      'firstunder01',
                                                      'firstunder02',
                                                      'max1stepchange',
                                                      'meanpfnn',
                                                      'pfnn_1',
                                                      'pfnn_10',
                                                      'pfnn_2',
                                                      'pfnn_3',
                                                      'pfnn_4',
                                                      'pfnn_5',
                                                      'pfnn_6',
                                                      'pfnn_7',
                                                      'pfnn_8',
                                                      'pfnn_9',
                                                      'stdpfnn']}
    if de is None:
        out = eng.run_function(1, 'NL_MS_fnn', x, )
    elif tau is None:
        out = eng.run_function(1, 'NL_MS_fnn', x, de)
    elif th is None:
        out = eng.run_function(1, 'NL_MS_fnn', x, de, tau)
    elif kth is None:
        out = eng.run_function(1, 'NL_MS_fnn', x, de, tau, th)
    elif justbest is None:
        out = eng.run_function(1, 'NL_MS_fnn', x, de, tau, th, kth)
    elif bestp is None:
        out = eng.run_function(1, 'NL_MS_fnn', x, de, tau, th, kth, justbest)
    else:
        out = eng.run_function(1, 'NL_MS_fnn', x, de, tau, th, kth, justbest, bestp)
    return outfunc(out)


class NL_MS_fnn(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Determines the number of false nearest neighbors for the embedded time series
    % using Michael Small's false nearest neighbor code, fnn (renamed MS_fnn here)
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % Code available at http://small.eie.polyu.edu.hk/matlab/
    % 
    % False nearest neighbors are judged using a ratio of the distances between the
    % next k points and the neighboring points of a given datapoint.
    % 
    %---INPUTS:
    % y, the input time series
    % de, the embedding dimensions to compare across (a vector)
    % tau, the time-delay (can be 'ac' or 'mi' to be the first zero-crossing of ACF,
    %                       or first minimum of AMI, respectively)
    % th, the distance threshold for neighbours
    % kth, the the distance to next points
    % [opt] justbest, can be set to 1 to just return the best embedding dimension, m_{best}
    % [opt] bestp, if justbest = 1, can set bestp as the proportion of false nearest
    %              neighbours at which the optimal embedding dimension is selected.
    % 
    % This function returns statistics on the proportion of false nearest neighbors
    % as a function of the embedding dimension m = m_{min}, m_{min}+1, ..., m_{max}
    % for a given time lag tau, and distance threshold for neighbors, d_{th}.
    % 
    %---OUTPUTS: include the proportion of false nearest neighbors at each m, the mean
    % and spread, and the smallest m at which the proportion of false nearest
    % neighbors drops below each of a set of fixed thresholds.
    % 
    ----------------------------------------
    """

    outnames = ('firstunder001',
                'firstunder002',
                'firstunder005',
                'firstunder01',
                'firstunder02',
                'max1stepchange',
                'meanpfnn',
                'pfnn_1',
                'pfnn_10',
                'pfnn_2',
                'pfnn_3',
                'pfnn_4',
                'pfnn_5',
                'pfnn_6',
                'pfnn_7',
                'pfnn_8',
                'pfnn_9',
                'stdpfnn')

    def __init__(self, de=MatlabSequence('1:10'), tau='mi', th=5, kth=1, justbest=None, bestp=None):
        super(NL_MS_fnn, self).__init__(add_descriptors=False)
        self.de = de
        self.tau = tau
        self.th = th
        self.kth = kth
        self.justbest = justbest
        self.bestp = bestp

    def eval(self, engine, x):
        return HCTSA_NL_MS_fnn(engine,
                               x,
                               de=self.de,
                               tau=self.tau,
                               th=self.th,
                               kth=self.kth,
                               justbest=self.justbest,
                               bestp=self.bestp)


def HCTSA_NL_MS_nlpe(eng, x, de='fnn', tau='mi', maxN=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the normalized 'drop-one-out' constant interpolation nonlinear
    % prediction error for a time-delay embedded time series using Michael Small's
    % code nlpe (renamed MS_nlpe here):
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % 
    % Michael Small's Matlab code is available at http://small.eie.polyu.edu.hk/matlab/
    % 
    %---INPUTS:
    % y, the input time series
    % de, the embedding dimension (can be an integer, or 'fnn' to select as the
    %       point where the proportion of false nearest neighbors falls below 5%
    %       using NL_MS_fnn)
    % tau, the time-delay (can be an integer or 'ac' to be the first zero-crossing
    %       of the ACF or 'mi' to be the first minimum of the automutual information
    %       function)
    % 
    %---OUTPUTS: include measures of the meanerror of the nonlinear predictor, and a
    % set of measures on the correlation, Gaussianity, etc. of the residuals.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac1n',
                                                      'ac2',
                                                      'ac2n',
                                                      'ac3',
                                                      'ac3n',
                                                      'acmnd0',
                                                      'acsnd0',
                                                      'dwts',
                                                      'ftbth',
                                                      'maxonmean',
                                                      'meanabs',
                                                      'meane',
                                                      'minfpe',
                                                      'minsbc',
                                                      'mms',
                                                      'msqerr',
                                                      'normksstat',
                                                      'normp',
                                                      'p1_5',
                                                      'p2_5',
                                                      'p3_5',
                                                      'p4_5',
                                                      'p5_5',
                                                      'popt',
                                                      'propbth',
                                                      'rmse',
                                                      'sbc1',
                                                      'stde']}
    if de is None:
        out = eng.run_function(1, 'NL_MS_nlpe', x, )
    elif tau is None:
        out = eng.run_function(1, 'NL_MS_nlpe', x, de)
    elif maxN is None:
        out = eng.run_function(1, 'NL_MS_nlpe', x, de, tau)
    else:
        out = eng.run_function(1, 'NL_MS_nlpe', x, de, tau, maxN)
    return outfunc(out)


class NL_MS_nlpe(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the normalized 'drop-one-out' constant interpolation nonlinear
    % prediction error for a time-delay embedded time series using Michael Small's
    % code nlpe (renamed MS_nlpe here):
    % 
    % cf. M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    % Physiology, and Finance (book) World Scientific, Nonlinear Science Series A,
    % Vol. 52 (2005)
    % 
    % Michael Small's Matlab code is available at http://small.eie.polyu.edu.hk/matlab/
    % 
    %---INPUTS:
    % y, the input time series
    % de, the embedding dimension (can be an integer, or 'fnn' to select as the
    %       point where the proportion of false nearest neighbors falls below 5%
    %       using NL_MS_fnn)
    % tau, the time-delay (can be an integer or 'ac' to be the first zero-crossing
    %       of the ACF or 'mi' to be the first minimum of the automutual information
    %       function)
    % 
    %---OUTPUTS: include measures of the meanerror of the nonlinear predictor, and a
    % set of measures on the correlation, Gaussianity, etc. of the residuals.
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac1n',
                'ac2',
                'ac2n',
                'ac3',
                'ac3n',
                'acmnd0',
                'acsnd0',
                'dwts',
                'ftbth',
                'maxonmean',
                'meanabs',
                'meane',
                'minfpe',
                'minsbc',
                'mms',
                'msqerr',
                'normksstat',
                'normp',
                'p1_5',
                'p2_5',
                'p3_5',
                'p4_5',
                'p5_5',
                'popt',
                'propbth',
                'rmse',
                'sbc1',
                'stde')

    def __init__(self, de='fnn', tau='mi', maxN=None):
        super(NL_MS_nlpe, self).__init__(add_descriptors=False)
        self.de = de
        self.tau = tau
        self.maxN = maxN

    def eval(self, engine, x):
        return HCTSA_NL_MS_nlpe(engine,
                                x,
                                de=self.de,
                                tau=self.tau,
                                maxN=self.maxN)


def HCTSA_NL_TISEAN_c1(eng, x, tau=1, mmm=(1, 7), tsep=0.02, Nref=0.5):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the c1 and c2d routines from the TISEAN nonlinear time-series
    % analysis package that compute curves for the fixed mass computation of the
    % information dimension.
    % 
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package" Hegger, R. and Kantz, H. and Schreiber, T., Chaos 9(2) 413 (1999)
    % 
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    % 
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    % 
    %---INPUTS:
    % 
    % y, the time series to analyze
    % 
    % tau, the time-delay (can be 'ac' or 'mi' for the first zero-crossing of the
    %           autocorrelation function or first minimum of the automutual
    %           information function)
    % 
    % mmm, a two-vector specifying the minimum and maximum embedding dimensions,
    %       e.g., [2,10] for m = 2 up to m = 10
    % 
    % tsep, time separation (can be between 0 and 1 for a proportion of the
    %       time-series length)
    %       
    % Nref, the number of reference points (can also be between 0 and 1 to specify a
    %       proportion of the time-series length)
    % 
    % 
    %---OUTPUTS: optimal scaling ranges and dimension estimates for a time delay,
    % tau, embedding dimensions, m, ranging from m_{min} to m_{max}, a time
    % separation, tsep, and a number of reference points, Nref.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['bestestd',
                                                      'bestestdstd',
                                                      'bestgoodness',
                                                      'bestscrd',
                                                      'longestscr',
                                                      'maxd',
                                                      'maxmd',
                                                      'meanstd',
                                                      'mediand',
                                                      'mind',
                                                      'ranged']}
    if tau is None:
        out = eng.run_function(1, 'NL_TISEAN_c1', x, )
    elif mmm is None:
        out = eng.run_function(1, 'NL_TISEAN_c1', x, tau)
    elif tsep is None:
        out = eng.run_function(1, 'NL_TISEAN_c1', x, tau, mmm)
    elif Nref is None:
        out = eng.run_function(1, 'NL_TISEAN_c1', x, tau, mmm, tsep)
    else:
        out = eng.run_function(1, 'NL_TISEAN_c1', x, tau, mmm, tsep, Nref)
    return outfunc(out)


class NL_TISEAN_c1(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the c1 and c2d routines from the TISEAN nonlinear time-series
    % analysis package that compute curves for the fixed mass computation of the
    % information dimension.
    % 
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package" Hegger, R. and Kantz, H. and Schreiber, T., Chaos 9(2) 413 (1999)
    % 
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    % 
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    % 
    %---INPUTS:
    % 
    % y, the time series to analyze
    % 
    % tau, the time-delay (can be 'ac' or 'mi' for the first zero-crossing of the
    %           autocorrelation function or first minimum of the automutual
    %           information function)
    % 
    % mmm, a two-vector specifying the minimum and maximum embedding dimensions,
    %       e.g., [2,10] for m = 2 up to m = 10
    % 
    % tsep, time separation (can be between 0 and 1 for a proportion of the
    %       time-series length)
    %       
    % Nref, the number of reference points (can also be between 0 and 1 to specify a
    %       proportion of the time-series length)
    % 
    % 
    %---OUTPUTS: optimal scaling ranges and dimension estimates for a time delay,
    % tau, embedding dimensions, m, ranging from m_{min} to m_{max}, a time
    % separation, tsep, and a number of reference points, Nref.
    % 
    ----------------------------------------
    """

    outnames = ('bestestd',
                'bestestdstd',
                'bestgoodness',
                'bestscrd',
                'longestscr',
                'maxd',
                'maxmd',
                'meanstd',
                'mediand',
                'mind',
                'ranged')

    def __init__(self, tau=1, mmm=(1, 7), tsep=0.02, Nref=0.5):
        super(NL_TISEAN_c1, self).__init__(add_descriptors=False)
        self.tau = tau
        self.mmm = mmm
        self.tsep = tsep
        self.Nref = Nref

    def eval(self, engine, x):
        return HCTSA_NL_TISEAN_c1(engine,
                                  x,
                                  tau=self.tau,
                                  mmm=self.mmm,
                                  tsep=self.tsep,
                                  Nref=self.Nref)


def HCTSA_NL_TISEAN_d2(eng, x, tau=1, maxm=10, theilerwin=0):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the d2 routine from the popular TISEAN package for
    % nonlinear time-series analysis:
    % 
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    % 
    % The TISEAN package is available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    % 
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    % 
    % The function estimates the correlation sum, the correlation dimension and
    % the correlation entropy of a given time series, y. Our code uses the outputs
    % from this algorithm to return a set of informative features about the results.
    % 
    %---INPUTS:
    % 
    % y, input time series
    % 
    % tau, time-delay (can be 'ac' or 'mi' for first zero-crossing of
    %       autocorrelation function, or first minimum of the automutual
    %       information)
    %       
    % maxm, the maximum embedding dimension
    % 
    % theilerwin, the Theiler window
    % 
    % cf. "Spurious dimension from correlation algorithms applied to limited
    % time-series data", J. Theiler, Phys. Rev. A, 34(3) 2427 (1986)
    % 
    % cf. "Nonlinear Time Series Analysis", Cambridge University Press, H. Kantz
    % and T. Schreiber (2004)
    % 
    % Taken's estimator is computed for the correlation dimension, as well as related
    % statistics, including other dimension estimates by finding appropriate scaling
    % ranges, and searching for a flat region in the output of TISEAN's h2
    % algorithm, which indicates determinism/deterministic chaos.
    % 
    % To find a suitable scaling range, a penalized regression procedure is used to
    % determine an optimal scaling range that simultaneously spans the greatest
    % range of scales and shows the best fit to the data, and return the range, a
    % goodness of fit statistic, and a dimension estimate.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['bend2_maxdim',
                                                      'bend2_meandim',
                                                      'bend2_meangoodness',
                                                      'bend2_mindim',
                                                      'bend2g_maxdim',
                                                      'bend2g_meandim',
                                                      'bend2g_meangoodness',
                                                      'bend2g_mindim',
                                                      'benmmind2_goodness',
                                                      'benmmind2_linrmserr',
                                                      'benmmind2_logminl',
                                                      'benmmind2_stabledim',
                                                      'benmmind2g_goodness',
                                                      'benmmind2g_linrmserr',
                                                      'benmmind2g_logminl',
                                                      'benmmind2g_stabledim',
                                                      'd2_dimest',
                                                      'd2_dimstd',
                                                      'd2_goodness',
                                                      'd2_logmaxscr',
                                                      'd2_logminscr',
                                                      'd2_logscr',
                                                      'd2g_dimest',
                                                      'd2g_dimstd',
                                                      'd2g_goodness',
                                                      'd2g_logmaxscr',
                                                      'd2g_logminscr',
                                                      'd2g_logscr',
                                                      'flatsh2min_goodness',
                                                      'flatsh2min_linrmserr',
                                                      'flatsh2min_ri1',
                                                      'flatsh2min_stabled',
                                                      'h2bestgoodness',
                                                      'h2besth2',
                                                      'h2meangoodness',
                                                      'meanh2',
                                                      'medianh2',
                                                      'slopesh2_goodness',
                                                      'slopesh2_linrmserr',
                                                      'slopesh2_ri1',
                                                      'slopesh2_stabled',
                                                      'takens05_iqr',
                                                      'takens05_max',
                                                      'takens05_mean',
                                                      'takens05_median',
                                                      'takens05_min',
                                                      'takens05_std',
                                                      'takens05mmin_goodness',
                                                      'takens05mmin_linrmserr',
                                                      'takens05mmin_ri',
                                                      'takens05mmin_stabled']}
    if tau is None:
        out = eng.run_function(1, 'NL_TISEAN_d2', x, )
    elif maxm is None:
        out = eng.run_function(1, 'NL_TISEAN_d2', x, tau)
    elif theilerwin is None:
        out = eng.run_function(1, 'NL_TISEAN_d2', x, tau, maxm)
    else:
        out = eng.run_function(1, 'NL_TISEAN_d2', x, tau, maxm, theilerwin)
    return outfunc(out)


class NL_TISEAN_d2(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the d2 routine from the popular TISEAN package for
    % nonlinear time-series analysis:
    % 
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    % 
    % The TISEAN package is available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    % 
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    % 
    % The function estimates the correlation sum, the correlation dimension and
    % the correlation entropy of a given time series, y. Our code uses the outputs
    % from this algorithm to return a set of informative features about the results.
    % 
    %---INPUTS:
    % 
    % y, input time series
    % 
    % tau, time-delay (can be 'ac' or 'mi' for first zero-crossing of
    %       autocorrelation function, or first minimum of the automutual
    %       information)
    %       
    % maxm, the maximum embedding dimension
    % 
    % theilerwin, the Theiler window
    % 
    % cf. "Spurious dimension from correlation algorithms applied to limited
    % time-series data", J. Theiler, Phys. Rev. A, 34(3) 2427 (1986)
    % 
    % cf. "Nonlinear Time Series Analysis", Cambridge University Press, H. Kantz
    % and T. Schreiber (2004)
    % 
    % Taken's estimator is computed for the correlation dimension, as well as related
    % statistics, including other dimension estimates by finding appropriate scaling
    % ranges, and searching for a flat region in the output of TISEAN's h2
    % algorithm, which indicates determinism/deterministic chaos.
    % 
    % To find a suitable scaling range, a penalized regression procedure is used to
    % determine an optimal scaling range that simultaneously spans the greatest
    % range of scales and shows the best fit to the data, and return the range, a
    % goodness of fit statistic, and a dimension estimate.
    % 
    ----------------------------------------
    """

    outnames = ('bend2_maxdim',
                'bend2_meandim',
                'bend2_meangoodness',
                'bend2_mindim',
                'bend2g_maxdim',
                'bend2g_meandim',
                'bend2g_meangoodness',
                'bend2g_mindim',
                'benmmind2_goodness',
                'benmmind2_linrmserr',
                'benmmind2_logminl',
                'benmmind2_stabledim',
                'benmmind2g_goodness',
                'benmmind2g_linrmserr',
                'benmmind2g_logminl',
                'benmmind2g_stabledim',
                'd2_dimest',
                'd2_dimstd',
                'd2_goodness',
                'd2_logmaxscr',
                'd2_logminscr',
                'd2_logscr',
                'd2g_dimest',
                'd2g_dimstd',
                'd2g_goodness',
                'd2g_logmaxscr',
                'd2g_logminscr',
                'd2g_logscr',
                'flatsh2min_goodness',
                'flatsh2min_linrmserr',
                'flatsh2min_ri1',
                'flatsh2min_stabled',
                'h2bestgoodness',
                'h2besth2',
                'h2meangoodness',
                'meanh2',
                'medianh2',
                'slopesh2_goodness',
                'slopesh2_linrmserr',
                'slopesh2_ri1',
                'slopesh2_stabled',
                'takens05_iqr',
                'takens05_max',
                'takens05_mean',
                'takens05_median',
                'takens05_min',
                'takens05_std',
                'takens05mmin_goodness',
                'takens05mmin_linrmserr',
                'takens05mmin_ri',
                'takens05mmin_stabled')

    def __init__(self, tau=1, maxm=10, theilerwin=0):
        super(NL_TISEAN_d2, self).__init__(add_descriptors=False)
        self.tau = tau
        self.maxm = maxm
        self.theilerwin = theilerwin

    def eval(self, engine, x):
        return HCTSA_NL_TISEAN_d2(engine,
                                  x,
                                  tau=self.tau,
                                  maxm=self.maxm,
                                  theilerwin=self.theilerwin)


def HCTSA_NL_TSTL_FractalDimensions(eng, x, kmin=2, kmax=100, Nref=0.2, gstart=1,
                                    gend=5, past=10, steps=32, embedparams=(1, 5)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the fractal dimension spectrum, D(q), using moments of neighbor
    % distances for time-delay embedded time series by referencing the code,
    % fracdims, from the TSTOOL package.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, column vector of time series data
    % 
    % kmin, minimum number of neighbours for each reference point
    % 
    % kmax, maximum number of neighbours for each reference point
    % 
    % Nref, number of randomly-chosen reference points (-1: use all points)
    % 
    % gstart, starting value for moments
    % 
    % gend, end value for moments
    % 
    % past [opt], number of samples to exclude before an after each reference
    %             index (default=0)
    % 
    % steps [opt], number of moments to calculate (default=32);
    % 
    % embedparams, how to embed the time series using a time-delay reconstruction
    % 
    % 
    %---OUTPUTS: include basic statistics of D(q) and q, statistics from a linear fit,
    % and an exponential fit of the form D(q) = Aexp(Bq) + C.
    % 
    %---HISTORY:
    % Ben Fulcher, November 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['expfit_a',
                                                      'expfit_adjr2',
                                                      'expfit_b',
                                                      'expfit_c',
                                                      'expfit_r2',
                                                      'expfit_rmse',
                                                      'linfit_a',
                                                      'linfit_b',
                                                      'linfit_rmsqres',
                                                      'maxDq',
                                                      'maxq',
                                                      'meanDq',
                                                      'meanq',
                                                      'minDq',
                                                      'minq',
                                                      'rangeDq',
                                                      'rangeq']}
    if kmin is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, )
    elif kmax is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin)
    elif Nref is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax)
    elif gstart is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax, Nref)
    elif gend is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax, Nref, gstart)
    elif past is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax, Nref, gstart, gend)
    elif steps is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax, Nref, gstart, gend, past)
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax, Nref, gstart, gend, past, steps)
    else:
        out = eng.run_function(1, 'NL_TSTL_FractalDimensions', x, kmin, kmax, Nref, gstart, gend, past, steps, embedparams)
    return outfunc(out)


class NL_TSTL_FractalDimensions(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the fractal dimension spectrum, D(q), using moments of neighbor
    % distances for time-delay embedded time series by referencing the code,
    % fracdims, from the TSTOOL package.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, column vector of time series data
    % 
    % kmin, minimum number of neighbours for each reference point
    % 
    % kmax, maximum number of neighbours for each reference point
    % 
    % Nref, number of randomly-chosen reference points (-1: use all points)
    % 
    % gstart, starting value for moments
    % 
    % gend, end value for moments
    % 
    % past [opt], number of samples to exclude before an after each reference
    %             index (default=0)
    % 
    % steps [opt], number of moments to calculate (default=32);
    % 
    % embedparams, how to embed the time series using a time-delay reconstruction
    % 
    % 
    %---OUTPUTS: include basic statistics of D(q) and q, statistics from a linear fit,
    % and an exponential fit of the form D(q) = Aexp(Bq) + C.
    % 
    %---HISTORY:
    % Ben Fulcher, November 2009
    % 
    ----------------------------------------
    """

    outnames = ('expfit_a',
                'expfit_adjr2',
                'expfit_b',
                'expfit_c',
                'expfit_r2',
                'expfit_rmse',
                'linfit_a',
                'linfit_b',
                'linfit_rmsqres',
                'maxDq',
                'maxq',
                'meanDq',
                'meanq',
                'minDq',
                'minq',
                'rangeDq',
                'rangeq')

    def __init__(self, kmin=2, kmax=100, Nref=0.2, gstart=1, gend=5, past=10, steps=32, embedparams=(1, 5)):
        super(NL_TSTL_FractalDimensions, self).__init__(add_descriptors=False)
        self.kmin = kmin
        self.kmax = kmax
        self.Nref = Nref
        self.gstart = gstart
        self.gend = gend
        self.past = past
        self.steps = steps
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_FractalDimensions(engine,
                                               x,
                                               kmin=self.kmin,
                                               kmax=self.kmax,
                                               Nref=self.Nref,
                                               gstart=self.gstart,
                                               gend=self.gend,
                                               past=self.past,
                                               steps=self.steps,
                                               embedparams=self.embedparams)


def HCTSA_NL_TSTL_GPCorrSum(eng, x, Nref=-1, r=0.1, thwin=40, nbins=20, embedparams=('ac', 'cao'), dotwo=2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses TSTOOL code corrsum (or corrsum2) to compute scaling of the correlation sum for a
    % time-delay reconstructed time series by the Grassberger-Proccacia algorithm
    % using fast nearest neighbor search.
    % 
    % cf. "Characterization of Strange Attractors", P. Grassberger and I. Procaccia,
    % Phys. Rev. Lett. 50(5) 346 (1983)
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time-series data
    % 
    % Nref, number of (randomly-chosen) reference points (-1: use all points,
    %       if a decimal, then use this fraction of the time series length)
    %       
    % r, maximum search radius relative to attractor size, 0 < r < 1
    % 
    % thwin, number of samples to exclude before and after each reference index
    %        (~ Theiler window)
    % 
    % nbins, number of partitioned bins
    % 
    % embedparams, embedding parameters to feed BF_embed.m for embedding the
    %               signal in the form {tau,m}
    % 
    % dotwo, if this is set to 1, will use corrsum, if set to 2, will use corrsum2.
    %           For corrsum2, n specifies the number of pairs per bin. Default is 1,
    %           to use corrsum.
    % 
    % 
    %---OUTPUTS: basic statistics on the outputs of corrsum, including iteratively
    % re-weighted least squares linear fits to log-log plots using the robustfit
    % function in Matlab's Statistics Toolbox.
    % 
    %---HISTORY:
    % Ben Fulcher, November 2009
    %
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['maxlnCr',
                                                      'maxlnr',
                                                      'meanlnCr',
                                                      'minlnCr',
                                                      'minlnr',
                                                      'rangelnCr',
                                                      'robfit_a1',
                                                      'robfit_a2',
                                                      'robfit_s',
                                                      'robfit_sea1',
                                                      'robfit_sea2',
                                                      'robfit_sigrat',
                                                      'robfitresac1',
                                                      'robfitresmeanabs',
                                                      'robfitresmeansq']}
    if Nref is None:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, )
    elif r is None:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, Nref)
    elif thwin is None:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, Nref, r)
    elif nbins is None:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, Nref, r, thwin)
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, Nref, r, thwin, nbins)
    elif dotwo is None:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, Nref, r, thwin, nbins, embedparams)
    else:
        out = eng.run_function(1, 'NL_TSTL_GPCorrSum', x, Nref, r, thwin, nbins, embedparams, dotwo)
    return outfunc(out)


class NL_TSTL_GPCorrSum(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses TSTOOL code corrsum (or corrsum2) to compute scaling of the correlation sum for a
    % time-delay reconstructed time series by the Grassberger-Proccacia algorithm
    % using fast nearest neighbor search.
    % 
    % cf. "Characterization of Strange Attractors", P. Grassberger and I. Procaccia,
    % Phys. Rev. Lett. 50(5) 346 (1983)
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time-series data
    % 
    % Nref, number of (randomly-chosen) reference points (-1: use all points,
    %       if a decimal, then use this fraction of the time series length)
    %       
    % r, maximum search radius relative to attractor size, 0 < r < 1
    % 
    % thwin, number of samples to exclude before and after each reference index
    %        (~ Theiler window)
    % 
    % nbins, number of partitioned bins
    % 
    % embedparams, embedding parameters to feed BF_embed.m for embedding the
    %               signal in the form {tau,m}
    % 
    % dotwo, if this is set to 1, will use corrsum, if set to 2, will use corrsum2.
    %           For corrsum2, n specifies the number of pairs per bin. Default is 1,
    %           to use corrsum.
    % 
    % 
    %---OUTPUTS: basic statistics on the outputs of corrsum, including iteratively
    % re-weighted least squares linear fits to log-log plots using the robustfit
    % function in Matlab's Statistics Toolbox.
    % 
    %---HISTORY:
    % Ben Fulcher, November 2009
    %
    ----------------------------------------
    """

    outnames = ('maxlnCr',
                'maxlnr',
                'meanlnCr',
                'minlnCr',
                'minlnr',
                'rangelnCr',
                'robfit_a1',
                'robfit_a2',
                'robfit_s',
                'robfit_sea1',
                'robfit_sea2',
                'robfit_sigrat',
                'robfitresac1',
                'robfitresmeanabs',
                'robfitresmeansq')

    def __init__(self, Nref=-1, r=0.1, thwin=40, nbins=20, embedparams=('ac', 'cao'), dotwo=2):
        super(NL_TSTL_GPCorrSum, self).__init__(add_descriptors=False)
        self.Nref = Nref
        self.r = r
        self.thwin = thwin
        self.nbins = nbins
        self.embedparams = embedparams
        self.dotwo = dotwo

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_GPCorrSum(engine,
                                       x,
                                       Nref=self.Nref,
                                       r=self.r,
                                       thwin=self.thwin,
                                       nbins=self.nbins,
                                       embedparams=self.embedparams,
                                       dotwo=self.dotwo)


def HCTSA_NL_TSTL_LargestLyap(eng, x, Nref=0.5, maxtstep=0.1, past=0.01, NNR=3, embedparams=('mi', 'cao')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the largest Lyapunov exponent of a time-delay reconstructed time
    % series using the TSTOOL code 'largelyap'.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % The algorithm used (using formula (1.5) in Parlitz Nonlinear Time Series
    % Analysis book) is very similar to the Wolf algorithm:
    % "Determining Lyapunov exponents from a time series", A. Wolf et al., Physica D
    % 16(3) 285 (1985)
    % 
    %---INPUTS:
    % 
    % y, the time series to analyze
    % 
    % Nref, number of randomly-chosen reference points (-1 == all)
    % 
    % maxtstep, maximum prediction length (samples)
    % 
    % past, exclude -- Theiler window idea
    % 
    % NNR, number of nearest neighbours
    % 
    % embedparams, input to BF_embed, how to time-delay-embed the time series, in
    %               the form {tau,m}, where string specifiers can indicate standard
    %               methods of determining tau or m.
    % 
    %---OUTPUTS: a range of statistics on the outputs from this function, including
    % a penalized linear regression to the scaling range in an attempt to fit to as
    % much of the range of scales as possible while simultaneously achieving the
    % best possible linear fit.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['expfit_a',
                                                      'expfit_adjr2',
                                                      'expfit_b',
                                                      'expfit_r2',
                                                      'expfit_rmse',
                                                      'maxp',
                                                      'ncross08max',
                                                      'ncross09max',
                                                      'p1',
                                                      'p2',
                                                      'p3',
                                                      'p4',
                                                      'p5',
                                                      'pcross08max',
                                                      'pcross09max',
                                                      'to05max',
                                                      'to07max',
                                                      'to08max',
                                                      'to095max',
                                                      'to09max',
                                                      've_gradient',
                                                      've_intercept',
                                                      've_meanabsres',
                                                      've_minbad',
                                                      've_rmsres',
                                                      'vse_gradient',
                                                      'vse_intercept',
                                                      'vse_meanabsres',
                                                      'vse_minbad',
                                                      'vse_rmsres']}
    if Nref is None:
        out = eng.run_function(1, 'NL_TSTL_LargestLyap', x, )
    elif maxtstep is None:
        out = eng.run_function(1, 'NL_TSTL_LargestLyap', x, Nref)
    elif past is None:
        out = eng.run_function(1, 'NL_TSTL_LargestLyap', x, Nref, maxtstep)
    elif NNR is None:
        out = eng.run_function(1, 'NL_TSTL_LargestLyap', x, Nref, maxtstep, past)
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_LargestLyap', x, Nref, maxtstep, past, NNR)
    else:
        out = eng.run_function(1, 'NL_TSTL_LargestLyap', x, Nref, maxtstep, past, NNR, embedparams)
    return outfunc(out)


class NL_TSTL_LargestLyap(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the largest Lyapunov exponent of a time-delay reconstructed time
    % series using the TSTOOL code 'largelyap'.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % The algorithm used (using formula (1.5) in Parlitz Nonlinear Time Series
    % Analysis book) is very similar to the Wolf algorithm:
    % "Determining Lyapunov exponents from a time series", A. Wolf et al., Physica D
    % 16(3) 285 (1985)
    % 
    %---INPUTS:
    % 
    % y, the time series to analyze
    % 
    % Nref, number of randomly-chosen reference points (-1 == all)
    % 
    % maxtstep, maximum prediction length (samples)
    % 
    % past, exclude -- Theiler window idea
    % 
    % NNR, number of nearest neighbours
    % 
    % embedparams, input to BF_embed, how to time-delay-embed the time series, in
    %               the form {tau,m}, where string specifiers can indicate standard
    %               methods of determining tau or m.
    % 
    %---OUTPUTS: a range of statistics on the outputs from this function, including
    % a penalized linear regression to the scaling range in an attempt to fit to as
    % much of the range of scales as possible while simultaneously achieving the
    % best possible linear fit.
    % 
    ----------------------------------------
    """

    outnames = ('expfit_a',
                'expfit_adjr2',
                'expfit_b',
                'expfit_r2',
                'expfit_rmse',
                'maxp',
                'ncross08max',
                'ncross09max',
                'p1',
                'p2',
                'p3',
                'p4',
                'p5',
                'pcross08max',
                'pcross09max',
                'to05max',
                'to07max',
                'to08max',
                'to095max',
                'to09max',
                've_gradient',
                've_intercept',
                've_meanabsres',
                've_minbad',
                've_rmsres',
                'vse_gradient',
                'vse_intercept',
                'vse_meanabsres',
                'vse_minbad',
                'vse_rmsres')

    def __init__(self, Nref=0.5, maxtstep=0.1, past=0.01, NNR=3, embedparams=('mi', 'cao')):
        super(NL_TSTL_LargestLyap, self).__init__(add_descriptors=False)
        self.Nref = Nref
        self.maxtstep = maxtstep
        self.past = past
        self.NNR = NNR
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_LargestLyap(engine,
                                         x,
                                         Nref=self.Nref,
                                         maxtstep=self.maxtstep,
                                         past=self.past,
                                         NNR=self.NNR,
                                         embedparams=self.embedparams)


def HCTSA_NL_TSTL_PoincareSection(eng, x, ref='max', embedparams=('ac', 3)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Obtains a Poincare section of the time-delay embedded time series, producing a
    % set of vector points projected orthogonal to the tangential vector at the
    % specified index using TSTOOL code 'poincare'. This function then tries to
    % obtain interesting structural measures from this output.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % ref: the reference point. Can be an absolute number (2 takes the second point
    %      in the (embedded) time series) or a string like 'max' or 'min' that takes
    %      the first maximum, ... in the (scalar) time series, ...
    % 
    % embedparams: the usual thing to give BF_embed for the time-delay embedding, as
    %               {tau,m}. A common choice for m is 3 -- i.e., embed in a 3
    %               dimensional space so that the Poincare section is 2-dimensional.
    % 
    % 
    %---OUTPUTS: include statistics on the x- and y- components of these vectors on the
    % Poincare surface, on distances between adjacent points, distances from the
    % mean position, and the entropy of the vector cloud.
    % 
    % Another thing that could be cool to do is to analyze variation in the plots as
    % ref changes... (not done here)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1D',
                                                      'ac1x',
                                                      'ac1y',
                                                      'ac2D',
                                                      'ac2x',
                                                      'ac2y',
                                                      'boxarea',
                                                      'hboxcounts10',
                                                      'hboxcounts5',
                                                      'iqrD',
                                                      'iqrds',
                                                      'iqrx',
                                                      'iqry',
                                                      'maxD',
                                                      'maxds',
                                                      'maxpbox10',
                                                      'maxpbox5',
                                                      'maxx',
                                                      'maxy',
                                                      'meanD',
                                                      'meands',
                                                      'meanpbox10',
                                                      'meanpbox5',
                                                      'meanx',
                                                      'meany',
                                                      'minD',
                                                      'minds',
                                                      'minpbox10',
                                                      'minpbox5',
                                                      'minx',
                                                      'miny',
                                                      'pcross',
                                                      'pwithin02',
                                                      'pwithin03',
                                                      'pwithin05',
                                                      'pwithin1',
                                                      'pwithin2',
                                                      'pwithinr01',
                                                      'rangepbox10',
                                                      'rangepbox5',
                                                      'stdD',
                                                      'stdx',
                                                      'stdy',
                                                      'tauacD',
                                                      'tauacx',
                                                      'tauacy',
                                                      'tracepbox10',
                                                      'tracepbox5',
                                                      'zerospbox10',
                                                      'zerospbox5']}
    if ref is None:
        out = eng.run_function(1, 'NL_TSTL_PoincareSection', x, )
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_PoincareSection', x, ref)
    else:
        out = eng.run_function(1, 'NL_TSTL_PoincareSection', x, ref, embedparams)
    return outfunc(out)


class NL_TSTL_PoincareSection(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Obtains a Poincare section of the time-delay embedded time series, producing a
    % set of vector points projected orthogonal to the tangential vector at the
    % specified index using TSTOOL code 'poincare'. This function then tries to
    % obtain interesting structural measures from this output.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % ref: the reference point. Can be an absolute number (2 takes the second point
    %      in the (embedded) time series) or a string like 'max' or 'min' that takes
    %      the first maximum, ... in the (scalar) time series, ...
    % 
    % embedparams: the usual thing to give BF_embed for the time-delay embedding, as
    %               {tau,m}. A common choice for m is 3 -- i.e., embed in a 3
    %               dimensional space so that the Poincare section is 2-dimensional.
    % 
    % 
    %---OUTPUTS: include statistics on the x- and y- components of these vectors on the
    % Poincare surface, on distances between adjacent points, distances from the
    % mean position, and the entropy of the vector cloud.
    % 
    % Another thing that could be cool to do is to analyze variation in the plots as
    % ref changes... (not done here)
    % 
    ----------------------------------------
    """

    outnames = ('ac1D',
                'ac1x',
                'ac1y',
                'ac2D',
                'ac2x',
                'ac2y',
                'boxarea',
                'hboxcounts10',
                'hboxcounts5',
                'iqrD',
                'iqrds',
                'iqrx',
                'iqry',
                'maxD',
                'maxds',
                'maxpbox10',
                'maxpbox5',
                'maxx',
                'maxy',
                'meanD',
                'meands',
                'meanpbox10',
                'meanpbox5',
                'meanx',
                'meany',
                'minD',
                'minds',
                'minpbox10',
                'minpbox5',
                'minx',
                'miny',
                'pcross',
                'pwithin02',
                'pwithin03',
                'pwithin05',
                'pwithin1',
                'pwithin2',
                'pwithinr01',
                'rangepbox10',
                'rangepbox5',
                'stdD',
                'stdx',
                'stdy',
                'tauacD',
                'tauacx',
                'tauacy',
                'tracepbox10',
                'tracepbox5',
                'zerospbox10',
                'zerospbox5')

    def __init__(self, ref='max', embedparams=('ac', 3)):
        super(NL_TSTL_PoincareSection, self).__init__(add_descriptors=False)
        self.ref = ref
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_PoincareSection(engine,
                                             x,
                                             ref=self.ref,
                                             embedparams=self.embedparams)


def HCTSA_NL_TSTL_ReturnTime(eng, x, NNR=0.05, maxT=1, past=0.05, Nref=-1, embedparams=(1, 3)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes a histogram of return times, the time taken for the time series to
    % return to a similar location in phase space for a given reference point using
    % the code return_time from TSTOOL.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % Strong peaks in the histogram are indicative of periodicities in the data.
    % 
    %---INPUTS:
    % 
    % y, scalar time series as a column vector
    % NNR, number of nearest neighbours
    % maxT, maximum return time to consider
    % past, Theiler window
    % Nref, number of reference indicies
    % embedparams, to feed into BF_embed
    % 
    %---OUTPUTS: include basic measures from the histogram, including the occurrence of
    % peaks, spread, proportion of zeros, and the distributional entropy.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['hcgdist',
                                                      'hhist',
                                                      'hhisthist',
                                                      'iqr',
                                                      'max',
                                                      'maxhisthist',
                                                      'maxpeaksep',
                                                      'meanpeaksep',
                                                      'minpeaksep',
                                                      'pg05',
                                                      'phisthistmin',
                                                      'pzeros',
                                                      'pzeroscgdist',
                                                      'rangecgdist',
                                                      'rangepeaksep',
                                                      'statrtym',
                                                      'statrtys',
                                                      'std',
                                                      'stdpeaksep']}
    if NNR is None:
        out = eng.run_function(1, 'NL_TSTL_ReturnTime', x, )
    elif maxT is None:
        out = eng.run_function(1, 'NL_TSTL_ReturnTime', x, NNR)
    elif past is None:
        out = eng.run_function(1, 'NL_TSTL_ReturnTime', x, NNR, maxT)
    elif Nref is None:
        out = eng.run_function(1, 'NL_TSTL_ReturnTime', x, NNR, maxT, past)
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_ReturnTime', x, NNR, maxT, past, Nref)
    else:
        out = eng.run_function(1, 'NL_TSTL_ReturnTime', x, NNR, maxT, past, Nref, embedparams)
    return outfunc(out)


class NL_TSTL_ReturnTime(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes a histogram of return times, the time taken for the time series to
    % return to a similar location in phase space for a given reference point using
    % the code return_time from TSTOOL.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    % Strong peaks in the histogram are indicative of periodicities in the data.
    % 
    %---INPUTS:
    % 
    % y, scalar time series as a column vector
    % NNR, number of nearest neighbours
    % maxT, maximum return time to consider
    % past, Theiler window
    % Nref, number of reference indicies
    % embedparams, to feed into BF_embed
    % 
    %---OUTPUTS: include basic measures from the histogram, including the occurrence of
    % peaks, spread, proportion of zeros, and the distributional entropy.
    % 
    ----------------------------------------
    """

    outnames = ('hcgdist',
                'hhist',
                'hhisthist',
                'iqr',
                'max',
                'maxhisthist',
                'maxpeaksep',
                'meanpeaksep',
                'minpeaksep',
                'pg05',
                'phisthistmin',
                'pzeros',
                'pzeroscgdist',
                'rangecgdist',
                'rangepeaksep',
                'statrtym',
                'statrtys',
                'std',
                'stdpeaksep')

    def __init__(self, NNR=0.05, maxT=1, past=0.05, Nref=-1, embedparams=(1, 3)):
        super(NL_TSTL_ReturnTime, self).__init__(add_descriptors=False)
        self.NNR = NNR
        self.maxT = maxT
        self.past = past
        self.Nref = Nref
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_ReturnTime(engine,
                                        x,
                                        NNR=self.NNR,
                                        maxT=self.maxT,
                                        past=self.past,
                                        Nref=self.Nref,
                                        embedparams=self.embedparams)


def HCTSA_NL_TSTL_TakensEstimator(eng, x, Nref=-1, rad=0.05, past=0.05, embedparams=('ac', 3)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the Taken's estimator for correlation dimension using the
    % TSTOOL code takens_estimator.
    %
    % cf. "Detecting strange attractors in turbulence", F. Takens.
    % Lect. Notes Math. 898 p366 (1981)
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, the input time series
    % Nref, the number of reference points (can be -1 to use all points)
    % rad, the maximum search radius (as a proportion of the attractor size)
    % past, the Theiler window
    % embedparams, the embedding parameters for BF_embed, in the form {tau,m}
    % 
    %---OUTPUT: the Taken's estimator of the correlation dimension, d2.
    % 
    %---HISTORY:
    % Ben Fulcher, 14/11/2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if Nref is None:
        out = eng.run_function(1, 'NL_TSTL_TakensEstimator', x, )
    elif rad is None:
        out = eng.run_function(1, 'NL_TSTL_TakensEstimator', x, Nref)
    elif past is None:
        out = eng.run_function(1, 'NL_TSTL_TakensEstimator', x, Nref, rad)
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_TakensEstimator', x, Nref, rad, past)
    else:
        out = eng.run_function(1, 'NL_TSTL_TakensEstimator', x, Nref, rad, past, embedparams)
    return outfunc(out)


class NL_TSTL_TakensEstimator(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the Taken's estimator for correlation dimension using the
    % TSTOOL code takens_estimator.
    %
    % cf. "Detecting strange attractors in turbulence", F. Takens.
    % Lect. Notes Math. 898 p366 (1981)
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, the input time series
    % Nref, the number of reference points (can be -1 to use all points)
    % rad, the maximum search radius (as a proportion of the attractor size)
    % past, the Theiler window
    % embedparams, the embedding parameters for BF_embed, in the form {tau,m}
    % 
    %---OUTPUT: the Taken's estimator of the correlation dimension, d2.
    % 
    %---HISTORY:
    % Ben Fulcher, 14/11/2009
    % 
    ----------------------------------------
    """

    def __init__(self, Nref=-1, rad=0.05, past=0.05, embedparams=('ac', 3)):
        super(NL_TSTL_TakensEstimator, self).__init__(add_descriptors=False)
        self.Nref = Nref
        self.rad = rad
        self.past = past
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_TakensEstimator(engine,
                                             x,
                                             Nref=self.Nref,
                                             rad=self.rad,
                                             past=self.past,
                                             embedparams=self.embedparams)


def HCTSA_NL_TSTL_acp(eng, x, tau='mi', past=1, maxdelay=((), 10, ()), maxdim=None, Nref=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the TSTOOL routine acp using a time lag, tau, a Theiler window,
    % past, maximum delay, maxdelay, maximum embedding dimension, maxdim, and number
    % of reference points, Nref.
    % 
    % The documentation isn't crystal clear, but I think this function has to do
    % with cross-prediction.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    % 
    % y, time series
    % 
    % tau, delay time
    % 
    % past, number of samples to exclude before and after each index (to avoid
    %               correlation effects ~ Theiler window)
    % 
    % maxdelay, maximal delay (<< length(y))
    % 
    % maxdim, maximal dimension to use
    % 
    % Nref, number of reference points
    % 
    %---OUTPUTS: statistics summarizing the output of the routine.
    % 
    % May in future want to also make outputs normalized by first value; so get
    % metrics on both absolute values at each dimension but also some
    % indication of the shape
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1_acpf_1',
                                                      'ac1_acpf_10',
                                                      'ac1_acpf_2',
                                                      'ac1_acpf_3',
                                                      'ac1_acpf_4',
                                                      'ac1_acpf_5',
                                                      'ac1_acpf_6',
                                                      'ac1_acpf_7',
                                                      'ac1_acpf_8',
                                                      'ac1_acpf_9',
                                                      'iqracpf_1',
                                                      'iqracpf_10',
                                                      'iqracpf_2',
                                                      'iqracpf_3',
                                                      'iqracpf_4',
                                                      'iqracpf_5',
                                                      'iqracpf_6',
                                                      'iqracpf_7',
                                                      'iqracpf_8',
                                                      'iqracpf_9',
                                                      'macpf_1',
                                                      'macpf_10',
                                                      'macpf_2',
                                                      'macpf_3',
                                                      'macpf_4',
                                                      'macpf_5',
                                                      'macpf_6',
                                                      'macpf_7',
                                                      'macpf_8',
                                                      'macpf_9',
                                                      'macpfdrop_1',
                                                      'macpfdrop_2',
                                                      'macpfdrop_3',
                                                      'macpfdrop_4',
                                                      'macpfdrop_5',
                                                      'macpfdrop_6',
                                                      'macpfdrop_7',
                                                      'macpfdrop_8',
                                                      'macpfdrop_9',
                                                      'mmacpfdiff',
                                                      'propdecmacpf',
                                                      'sacpf_1',
                                                      'sacpf_10',
                                                      'sacpf_2',
                                                      'sacpf_3',
                                                      'sacpf_4',
                                                      'sacpf_5',
                                                      'sacpf_6',
                                                      'sacpf_7',
                                                      'sacpf_8',
                                                      'sacpf_9',
                                                      'stdmacpfdiff']}
    if tau is None:
        out = eng.run_function(1, 'NL_TSTL_acp', x, )
    elif past is None:
        out = eng.run_function(1, 'NL_TSTL_acp', x, tau)
    elif maxdelay is None:
        out = eng.run_function(1, 'NL_TSTL_acp', x, tau, past)
    elif maxdim is None:
        out = eng.run_function(1, 'NL_TSTL_acp', x, tau, past, maxdelay)
    elif Nref is None:
        out = eng.run_function(1, 'NL_TSTL_acp', x, tau, past, maxdelay, maxdim)
    else:
        out = eng.run_function(1, 'NL_TSTL_acp', x, tau, past, maxdelay, maxdim, Nref)
    return outfunc(out)


class NL_TSTL_acp(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements the TSTOOL routine acp using a time lag, tau, a Theiler window,
    % past, maximum delay, maxdelay, maximum embedding dimension, maxdim, and number
    % of reference points, Nref.
    % 
    % The documentation isn't crystal clear, but I think this function has to do
    % with cross-prediction.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    %
    %---INPUTS:
    % 
    % y, time series
    % 
    % tau, delay time
    % 
    % past, number of samples to exclude before and after each index (to avoid
    %               correlation effects ~ Theiler window)
    % 
    % maxdelay, maximal delay (<< length(y))
    % 
    % maxdim, maximal dimension to use
    % 
    % Nref, number of reference points
    % 
    %---OUTPUTS: statistics summarizing the output of the routine.
    % 
    % May in future want to also make outputs normalized by first value; so get
    % metrics on both absolute values at each dimension but also some
    % indication of the shape
    % 
    ----------------------------------------
    """

    outnames = ('ac1_acpf_1',
                'ac1_acpf_10',
                'ac1_acpf_2',
                'ac1_acpf_3',
                'ac1_acpf_4',
                'ac1_acpf_5',
                'ac1_acpf_6',
                'ac1_acpf_7',
                'ac1_acpf_8',
                'ac1_acpf_9',
                'iqracpf_1',
                'iqracpf_10',
                'iqracpf_2',
                'iqracpf_3',
                'iqracpf_4',
                'iqracpf_5',
                'iqracpf_6',
                'iqracpf_7',
                'iqracpf_8',
                'iqracpf_9',
                'macpf_1',
                'macpf_10',
                'macpf_2',
                'macpf_3',
                'macpf_4',
                'macpf_5',
                'macpf_6',
                'macpf_7',
                'macpf_8',
                'macpf_9',
                'macpfdrop_1',
                'macpfdrop_2',
                'macpfdrop_3',
                'macpfdrop_4',
                'macpfdrop_5',
                'macpfdrop_6',
                'macpfdrop_7',
                'macpfdrop_8',
                'macpfdrop_9',
                'mmacpfdiff',
                'propdecmacpf',
                'sacpf_1',
                'sacpf_10',
                'sacpf_2',
                'sacpf_3',
                'sacpf_4',
                'sacpf_5',
                'sacpf_6',
                'sacpf_7',
                'sacpf_8',
                'sacpf_9',
                'stdmacpfdiff')

    def __init__(self, tau='mi', past=1, maxdelay=((), 10, ()), maxdim=None, Nref=None):
        super(NL_TSTL_acp, self).__init__(add_descriptors=False)
        self.tau = tau
        self.past = past
        self.maxdelay = maxdelay
        self.maxdim = maxdim
        self.Nref = Nref

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_acp(engine,
                                 x,
                                 tau=self.tau,
                                 past=self.past,
                                 maxdelay=self.maxdelay,
                                 maxdim=self.maxdim,
                                 Nref=self.Nref)


def HCTSA_NL_TSTL_dimensions(eng, x, nbins=50, embedparams=('ac', 'cao')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the box counting, information, and correlation dimension of a
    % time-delay embedded time series using the TSTOOL code 'dimensions'.
    % This function contains extensive code for estimating the best scaling range to
    % estimate the dimension using a penalized regression procedure.
    % 
    % TSTOOL, http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, column vector of time series data
    % 
    % nbins, maximum number of partitions per axis.
    % 
    % embedparams, embedding parameters to feed BF_embed.m for embedding the
    %              signal in the form {tau,m}
    % 
    % 
    %---OUTPUTS:
    % A range of statistics are returned about how each dimension estimate changes
    % with m, the scaling range in r, and the embedding dimension at which the best
    % fit is obtained.
    % 
    %---HISTORY:
    % Ben Fulcher, November 2009
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['bc_lfitb1',
                                                      'bc_lfitb2',
                                                      'bc_lfitb3',
                                                      'bc_lfitbmax',
                                                      'bc_lfitm1',
                                                      'bc_lfitm2',
                                                      'bc_lfitm3',
                                                      'bc_lfitmeansqdev1',
                                                      'bc_lfitmeansqdev2',
                                                      'bc_lfitmeansqdev3',
                                                      'bc_lfitmeansqdevmax',
                                                      'bc_lfitmmax',
                                                      'bc_maxscalingexp',
                                                      'bc_mbestfit',
                                                      'bc_meandiff',
                                                      'bc_meanm1',
                                                      'bc_meanm2',
                                                      'bc_meanm3',
                                                      'bc_meanmmax',
                                                      'bc_meanscalingexp',
                                                      'bc_mindiff',
                                                      'bc_minm1',
                                                      'bc_minm2',
                                                      'bc_minm3',
                                                      'bc_minmmax',
                                                      'bc_minscalingexp',
                                                      'bc_range1',
                                                      'bc_range2',
                                                      'bc_range3',
                                                      'bc_rangemmax',
                                                      'co_lfitb1',
                                                      'co_lfitb2',
                                                      'co_lfitb3',
                                                      'co_lfitbmax',
                                                      'co_lfitm1',
                                                      'co_lfitm2',
                                                      'co_lfitm3',
                                                      'co_lfitmeansqdev1',
                                                      'co_lfitmeansqdev2',
                                                      'co_lfitmeansqdev3',
                                                      'co_lfitmeansqdevmax',
                                                      'co_lfitmmax',
                                                      'co_maxscalingexp',
                                                      'co_mbestfit',
                                                      'co_meandiff',
                                                      'co_meanm1',
                                                      'co_meanm2',
                                                      'co_meanm3',
                                                      'co_meanmmax',
                                                      'co_meanscalingexp',
                                                      'co_mindiff',
                                                      'co_minm1',
                                                      'co_minm2',
                                                      'co_minm3',
                                                      'co_minmmax',
                                                      'co_minscalingexp',
                                                      'co_range1',
                                                      'co_range2',
                                                      'co_range3',
                                                      'co_rangemmax',
                                                      'in_lfitb1',
                                                      'in_lfitb2',
                                                      'in_lfitb3',
                                                      'in_lfitbmax',
                                                      'in_lfitm1',
                                                      'in_lfitm2',
                                                      'in_lfitm3',
                                                      'in_lfitmeansqdev1',
                                                      'in_lfitmeansqdev2',
                                                      'in_lfitmeansqdev3',
                                                      'in_lfitmeansqdevmax',
                                                      'in_lfitmmax',
                                                      'in_maxscalingexp',
                                                      'in_mbestfit',
                                                      'in_meandiff',
                                                      'in_meanm1',
                                                      'in_meanm2',
                                                      'in_meanm3',
                                                      'in_meanmmax',
                                                      'in_meanscalingexp',
                                                      'in_mindiff',
                                                      'in_minm1',
                                                      'in_minm2',
                                                      'in_minm3',
                                                      'in_minmmax',
                                                      'in_minscalingexp',
                                                      'in_range1',
                                                      'in_range2',
                                                      'in_range3',
                                                      'in_rangemmax',
                                                      'scr_bc_m1_logrmax',
                                                      'scr_bc_m1_logrmin',
                                                      'scr_bc_m1_logrrange',
                                                      'scr_bc_m1_meanabsres',
                                                      'scr_bc_m1_meansqres',
                                                      'scr_bc_m1_minbad',
                                                      'scr_bc_m1_pgone',
                                                      'scr_bc_m1_scaling_exp',
                                                      'scr_bc_m1_scaling_int',
                                                      'scr_bc_m2_logrmax',
                                                      'scr_bc_m2_logrmin',
                                                      'scr_bc_m2_logrrange',
                                                      'scr_bc_m2_meanabsres',
                                                      'scr_bc_m2_meansqres',
                                                      'scr_bc_m2_minbad',
                                                      'scr_bc_m2_pgone',
                                                      'scr_bc_m2_scaling_exp',
                                                      'scr_bc_m2_scaling_int',
                                                      'scr_bc_m3_logrmax',
                                                      'scr_bc_m3_logrmin',
                                                      'scr_bc_m3_logrrange',
                                                      'scr_bc_m3_meanabsres',
                                                      'scr_bc_m3_meansqres',
                                                      'scr_bc_m3_minbad',
                                                      'scr_bc_m3_pgone',
                                                      'scr_bc_m3_scaling_exp',
                                                      'scr_bc_m3_scaling_int',
                                                      'scr_bc_mopt_logrmax',
                                                      'scr_bc_mopt_logrmin',
                                                      'scr_bc_mopt_logrrange',
                                                      'scr_bc_mopt_meanabsres',
                                                      'scr_bc_mopt_meansqres',
                                                      'scr_bc_mopt_minbad',
                                                      'scr_bc_mopt_pgone',
                                                      'scr_bc_mopt_scaling_exp',
                                                      'scr_bc_mopt_scaling_int',
                                                      'scr_co_m1_logrmax',
                                                      'scr_co_m1_logrmin',
                                                      'scr_co_m1_logrrange',
                                                      'scr_co_m1_meanabsres',
                                                      'scr_co_m1_meansqres',
                                                      'scr_co_m1_minbad',
                                                      'scr_co_m1_pgone',
                                                      'scr_co_m1_scaling_exp',
                                                      'scr_co_m1_scaling_int',
                                                      'scr_co_m2_logrmax',
                                                      'scr_co_m2_logrmin',
                                                      'scr_co_m2_logrrange',
                                                      'scr_co_m2_meanabsres',
                                                      'scr_co_m2_meansqres',
                                                      'scr_co_m2_minbad',
                                                      'scr_co_m2_pgone',
                                                      'scr_co_m2_scaling_exp',
                                                      'scr_co_m2_scaling_int',
                                                      'scr_co_m3_logrmax',
                                                      'scr_co_m3_logrmin',
                                                      'scr_co_m3_logrrange',
                                                      'scr_co_m3_meanabsres',
                                                      'scr_co_m3_meansqres',
                                                      'scr_co_m3_minbad',
                                                      'scr_co_m3_pgone',
                                                      'scr_co_m3_scaling_exp',
                                                      'scr_co_m3_scaling_int',
                                                      'scr_co_mopt_logrmax',
                                                      'scr_co_mopt_logrmin',
                                                      'scr_co_mopt_logrrange',
                                                      'scr_co_mopt_meanabsres',
                                                      'scr_co_mopt_meansqres',
                                                      'scr_co_mopt_minbad',
                                                      'scr_co_mopt_pgone',
                                                      'scr_co_mopt_scaling_exp',
                                                      'scr_co_mopt_scaling_int',
                                                      'scr_in_m1_logrmax',
                                                      'scr_in_m1_logrmin',
                                                      'scr_in_m1_logrrange',
                                                      'scr_in_m1_meanabsres',
                                                      'scr_in_m1_meansqres',
                                                      'scr_in_m1_minbad',
                                                      'scr_in_m1_pgone',
                                                      'scr_in_m1_scaling_exp',
                                                      'scr_in_m1_scaling_int',
                                                      'scr_in_m2_logrmax',
                                                      'scr_in_m2_logrmin',
                                                      'scr_in_m2_logrrange',
                                                      'scr_in_m2_meanabsres',
                                                      'scr_in_m2_meansqres',
                                                      'scr_in_m2_minbad',
                                                      'scr_in_m2_pgone',
                                                      'scr_in_m2_scaling_exp',
                                                      'scr_in_m2_scaling_int',
                                                      'scr_in_m3_logrmax',
                                                      'scr_in_m3_logrmin',
                                                      'scr_in_m3_logrrange',
                                                      'scr_in_m3_meanabsres',
                                                      'scr_in_m3_meansqres',
                                                      'scr_in_m3_minbad',
                                                      'scr_in_m3_pgone',
                                                      'scr_in_m3_scaling_exp',
                                                      'scr_in_m3_scaling_int',
                                                      'scr_in_mopt_logrmax',
                                                      'scr_in_mopt_logrmin',
                                                      'scr_in_mopt_logrrange',
                                                      'scr_in_mopt_meanabsres',
                                                      'scr_in_mopt_meansqres',
                                                      'scr_in_mopt_minbad',
                                                      'scr_in_mopt_pgone',
                                                      'scr_in_mopt_scaling_exp',
                                                      'scr_in_mopt_scaling_int']}
    if nbins is None:
        out = eng.run_function(1, 'NL_TSTL_dimensions', x, )
    elif embedparams is None:
        out = eng.run_function(1, 'NL_TSTL_dimensions', x, nbins)
    else:
        out = eng.run_function(1, 'NL_TSTL_dimensions', x, nbins, embedparams)
    return outfunc(out)


class NL_TSTL_dimensions(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes the box counting, information, and correlation dimension of a
    % time-delay embedded time series using the TSTOOL code 'dimensions'.
    % This function contains extensive code for estimating the best scaling range to
    % estimate the dimension using a penalized regression procedure.
    % 
    % TSTOOL, http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, column vector of time series data
    % 
    % nbins, maximum number of partitions per axis.
    % 
    % embedparams, embedding parameters to feed BF_embed.m for embedding the
    %              signal in the form {tau,m}
    % 
    % 
    %---OUTPUTS:
    % A range of statistics are returned about how each dimension estimate changes
    % with m, the scaling range in r, and the embedding dimension at which the best
    % fit is obtained.
    % 
    %---HISTORY:
    % Ben Fulcher, November 2009
    ----------------------------------------
    """

    outnames = ('bc_lfitb1',
                'bc_lfitb2',
                'bc_lfitb3',
                'bc_lfitbmax',
                'bc_lfitm1',
                'bc_lfitm2',
                'bc_lfitm3',
                'bc_lfitmeansqdev1',
                'bc_lfitmeansqdev2',
                'bc_lfitmeansqdev3',
                'bc_lfitmeansqdevmax',
                'bc_lfitmmax',
                'bc_maxscalingexp',
                'bc_mbestfit',
                'bc_meandiff',
                'bc_meanm1',
                'bc_meanm2',
                'bc_meanm3',
                'bc_meanmmax',
                'bc_meanscalingexp',
                'bc_mindiff',
                'bc_minm1',
                'bc_minm2',
                'bc_minm3',
                'bc_minmmax',
                'bc_minscalingexp',
                'bc_range1',
                'bc_range2',
                'bc_range3',
                'bc_rangemmax',
                'co_lfitb1',
                'co_lfitb2',
                'co_lfitb3',
                'co_lfitbmax',
                'co_lfitm1',
                'co_lfitm2',
                'co_lfitm3',
                'co_lfitmeansqdev1',
                'co_lfitmeansqdev2',
                'co_lfitmeansqdev3',
                'co_lfitmeansqdevmax',
                'co_lfitmmax',
                'co_maxscalingexp',
                'co_mbestfit',
                'co_meandiff',
                'co_meanm1',
                'co_meanm2',
                'co_meanm3',
                'co_meanmmax',
                'co_meanscalingexp',
                'co_mindiff',
                'co_minm1',
                'co_minm2',
                'co_minm3',
                'co_minmmax',
                'co_minscalingexp',
                'co_range1',
                'co_range2',
                'co_range3',
                'co_rangemmax',
                'in_lfitb1',
                'in_lfitb2',
                'in_lfitb3',
                'in_lfitbmax',
                'in_lfitm1',
                'in_lfitm2',
                'in_lfitm3',
                'in_lfitmeansqdev1',
                'in_lfitmeansqdev2',
                'in_lfitmeansqdev3',
                'in_lfitmeansqdevmax',
                'in_lfitmmax',
                'in_maxscalingexp',
                'in_mbestfit',
                'in_meandiff',
                'in_meanm1',
                'in_meanm2',
                'in_meanm3',
                'in_meanmmax',
                'in_meanscalingexp',
                'in_mindiff',
                'in_minm1',
                'in_minm2',
                'in_minm3',
                'in_minmmax',
                'in_minscalingexp',
                'in_range1',
                'in_range2',
                'in_range3',
                'in_rangemmax',
                'scr_bc_m1_logrmax',
                'scr_bc_m1_logrmin',
                'scr_bc_m1_logrrange',
                'scr_bc_m1_meanabsres',
                'scr_bc_m1_meansqres',
                'scr_bc_m1_minbad',
                'scr_bc_m1_pgone',
                'scr_bc_m1_scaling_exp',
                'scr_bc_m1_scaling_int',
                'scr_bc_m2_logrmax',
                'scr_bc_m2_logrmin',
                'scr_bc_m2_logrrange',
                'scr_bc_m2_meanabsres',
                'scr_bc_m2_meansqres',
                'scr_bc_m2_minbad',
                'scr_bc_m2_pgone',
                'scr_bc_m2_scaling_exp',
                'scr_bc_m2_scaling_int',
                'scr_bc_m3_logrmax',
                'scr_bc_m3_logrmin',
                'scr_bc_m3_logrrange',
                'scr_bc_m3_meanabsres',
                'scr_bc_m3_meansqres',
                'scr_bc_m3_minbad',
                'scr_bc_m3_pgone',
                'scr_bc_m3_scaling_exp',
                'scr_bc_m3_scaling_int',
                'scr_bc_mopt_logrmax',
                'scr_bc_mopt_logrmin',
                'scr_bc_mopt_logrrange',
                'scr_bc_mopt_meanabsres',
                'scr_bc_mopt_meansqres',
                'scr_bc_mopt_minbad',
                'scr_bc_mopt_pgone',
                'scr_bc_mopt_scaling_exp',
                'scr_bc_mopt_scaling_int',
                'scr_co_m1_logrmax',
                'scr_co_m1_logrmin',
                'scr_co_m1_logrrange',
                'scr_co_m1_meanabsres',
                'scr_co_m1_meansqres',
                'scr_co_m1_minbad',
                'scr_co_m1_pgone',
                'scr_co_m1_scaling_exp',
                'scr_co_m1_scaling_int',
                'scr_co_m2_logrmax',
                'scr_co_m2_logrmin',
                'scr_co_m2_logrrange',
                'scr_co_m2_meanabsres',
                'scr_co_m2_meansqres',
                'scr_co_m2_minbad',
                'scr_co_m2_pgone',
                'scr_co_m2_scaling_exp',
                'scr_co_m2_scaling_int',
                'scr_co_m3_logrmax',
                'scr_co_m3_logrmin',
                'scr_co_m3_logrrange',
                'scr_co_m3_meanabsres',
                'scr_co_m3_meansqres',
                'scr_co_m3_minbad',
                'scr_co_m3_pgone',
                'scr_co_m3_scaling_exp',
                'scr_co_m3_scaling_int',
                'scr_co_mopt_logrmax',
                'scr_co_mopt_logrmin',
                'scr_co_mopt_logrrange',
                'scr_co_mopt_meanabsres',
                'scr_co_mopt_meansqres',
                'scr_co_mopt_minbad',
                'scr_co_mopt_pgone',
                'scr_co_mopt_scaling_exp',
                'scr_co_mopt_scaling_int',
                'scr_in_m1_logrmax',
                'scr_in_m1_logrmin',
                'scr_in_m1_logrrange',
                'scr_in_m1_meanabsres',
                'scr_in_m1_meansqres',
                'scr_in_m1_minbad',
                'scr_in_m1_pgone',
                'scr_in_m1_scaling_exp',
                'scr_in_m1_scaling_int',
                'scr_in_m2_logrmax',
                'scr_in_m2_logrmin',
                'scr_in_m2_logrrange',
                'scr_in_m2_meanabsres',
                'scr_in_m2_meansqres',
                'scr_in_m2_minbad',
                'scr_in_m2_pgone',
                'scr_in_m2_scaling_exp',
                'scr_in_m2_scaling_int',
                'scr_in_m3_logrmax',
                'scr_in_m3_logrmin',
                'scr_in_m3_logrrange',
                'scr_in_m3_meanabsres',
                'scr_in_m3_meansqres',
                'scr_in_m3_minbad',
                'scr_in_m3_pgone',
                'scr_in_m3_scaling_exp',
                'scr_in_m3_scaling_int',
                'scr_in_mopt_logrmax',
                'scr_in_mopt_logrmin',
                'scr_in_mopt_logrrange',
                'scr_in_mopt_meanabsres',
                'scr_in_mopt_meansqres',
                'scr_in_mopt_minbad',
                'scr_in_mopt_pgone',
                'scr_in_mopt_scaling_exp',
                'scr_in_mopt_scaling_int')

    def __init__(self, nbins=50, embedparams=('ac', 'cao')):
        super(NL_TSTL_dimensions, self).__init__(add_descriptors=False)
        self.nbins = nbins
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_NL_TSTL_dimensions(engine,
                                        x,
                                        nbins=self.nbins,
                                        embedparams=self.embedparams)


def HCTSA_NL_crptool_fnn(eng, x, maxm=10, r=2, taum='ac', th=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes and analyzes the false-nearest neighbours statistic.
    % 
    % Computation is done by referencing N. Marwan's code from the CRP Toolbox:
    % http://tocsy.pik-potsdam.de/CRPtoolbox/
    % 
    %---INPUTS:
    % y, the input time series
    % maxm, the maximum embedding dimension to consider
    % r, the threshold; neighbourhood criterion
    % taum, the method of determining the time delay, 'corr' for first zero-crossing
    %       of autocorrelation function, or 'mi' for the first minimum of the mutual
    %       information
    % 
    % th [opt], returns the first time the number of false nearest neighbours drops
    %           under this threshold
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['fnn10',
                                                      'fnn2',
                                                      'fnn3',
                                                      'fnn4',
                                                      'fnn5',
                                                      'fnn6',
                                                      'fnn7',
                                                      'fnn8',
                                                      'fnn9',
                                                      'm005',
                                                      'm01',
                                                      'm02',
                                                      'm05',
                                                      'mdrop',
                                                      'pdrop']}
    if maxm is None:
        out = eng.run_function(1, 'NL_crptool_fnn', x, )
    elif r is None:
        out = eng.run_function(1, 'NL_crptool_fnn', x, maxm)
    elif taum is None:
        out = eng.run_function(1, 'NL_crptool_fnn', x, maxm, r)
    elif th is None:
        out = eng.run_function(1, 'NL_crptool_fnn', x, maxm, r, taum)
    else:
        out = eng.run_function(1, 'NL_crptool_fnn', x, maxm, r, taum, th)
    return outfunc(out)


class NL_crptool_fnn(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Computes and analyzes the false-nearest neighbours statistic.
    % 
    % Computation is done by referencing N. Marwan's code from the CRP Toolbox:
    % http://tocsy.pik-potsdam.de/CRPtoolbox/
    % 
    %---INPUTS:
    % y, the input time series
    % maxm, the maximum embedding dimension to consider
    % r, the threshold; neighbourhood criterion
    % taum, the method of determining the time delay, 'corr' for first zero-crossing
    %       of autocorrelation function, or 'mi' for the first minimum of the mutual
    %       information
    % 
    % th [opt], returns the first time the number of false nearest neighbours drops
    %           under this threshold
    % 
    ----------------------------------------
    """

    outnames = ('fnn10',
                'fnn2',
                'fnn3',
                'fnn4',
                'fnn5',
                'fnn6',
                'fnn7',
                'fnn8',
                'fnn9',
                'm005',
                'm01',
                'm02',
                'm05',
                'mdrop',
                'pdrop')

    def __init__(self, maxm=10, r=2, taum='ac', th=None):
        super(NL_crptool_fnn, self).__init__(add_descriptors=False)
        self.maxm = maxm
        self.r = r
        self.taum = taum
        self.th = th

    def eval(self, engine, x):
        return HCTSA_NL_crptool_fnn(engine,
                                    x,
                                    maxm=self.maxm,
                                    r=self.r,
                                    taum=self.taum,
                                    th=self.th)


def HCTSA_NL_embed_PCA(eng, x, tau='mi', m=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Reconstructs the time series as a time-delay embedding, and performs Principal
    % Components Analysis on the result using princomp code from
    % Matlab's Bioinformatics Toolbox.
    % 
    % This technique is known as singular spectrum analysis
    % 
    % "Extracting qualitative dynamics from experimental data"
    % D. S. Broomhead and G. P. King, Physica D 20(2-3) 217 (1986)
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % tau, the time-delay, can be an integer or 'ac', or 'mi' for first
    %               zero-crossing of the autocorrelation function or first minimum
    %               of the automutual information, respectively
    %               
    % m, the embedding dimension
    % 
    % OUTPUTS: Various statistics summarizing the obtained eigenvalue distribution.
    % 
    % The suggestion to implement this idea was provided by Siddarth Arora.
    % (Siddharth Arora, <arora@maths.ox.ac.uk>)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['fb001',
                                                      'fb01',
                                                      'max',
                                                      'min',
                                                      'nto80',
                                                      'nto90',
                                                      'range',
                                                      'std',
                                                      'top2']}
    if tau is None:
        out = eng.run_function(1, 'NL_embed_PCA', x, )
    elif m is None:
        out = eng.run_function(1, 'NL_embed_PCA', x, tau)
    else:
        out = eng.run_function(1, 'NL_embed_PCA', x, tau, m)
    return outfunc(out)


class NL_embed_PCA(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Reconstructs the time series as a time-delay embedding, and performs Principal
    % Components Analysis on the result using princomp code from
    % Matlab's Bioinformatics Toolbox.
    % 
    % This technique is known as singular spectrum analysis
    % 
    % "Extracting qualitative dynamics from experimental data"
    % D. S. Broomhead and G. P. King, Physica D 20(2-3) 217 (1986)
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % tau, the time-delay, can be an integer or 'ac', or 'mi' for first
    %               zero-crossing of the autocorrelation function or first minimum
    %               of the automutual information, respectively
    %               
    % m, the embedding dimension
    % 
    % OUTPUTS: Various statistics summarizing the obtained eigenvalue distribution.
    % 
    % The suggestion to implement this idea was provided by Siddarth Arora.
    % (Siddharth Arora, <arora@maths.ox.ac.uk>)
    % 
    ----------------------------------------
    """

    outnames = ('fb001',
                'fb01',
                'max',
                'min',
                'nto80',
                'nto90',
                'range',
                'std',
                'top2')

    def __init__(self, tau='mi', m=10):
        super(NL_embed_PCA, self).__init__(add_descriptors=False)
        self.tau = tau
        self.m = m

    def eval(self, engine, x):
        return HCTSA_NL_embed_PCA(engine,
                                  x,
                                  tau=self.tau,
                                  m=self.m)


def HCTSA_NW_VisibilityGraph(eng, x, meth='horiz', maxL=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Constructs a visibility graph of the time series and returns various
    % statistics on the properties of the resulting network.
    % 
    % cf.: "From time series to complex networks: The visibility graph"
    % Lacasa, Lucas and Luque, Bartolo and Ballesteros, Fernando and Luque, Jordi
    % and Nuno, Juan Carlos P. Natl. Acad. Sci. USA. 105(13) 4972 (2008)
    % 
    % "Horizontal visibility graphs: Exact results for random time series"
    % Luque, B. and Lacasa, L. and Ballesteros, F. and Luque, J.
    % Phys. Rev. E. 80(4) 046103 (2009)
    % 
    % The normal visibility graph may not be implemented correctly, we focused only
    % on the horizontal visibility graph.
    % 
    %---INPUTS:
    % y, the time series (a column vector)
    % 
    % meth, the method for constructing:
    % 			(i) 'norm': the normal visibility definition
    % 			(ii) 'horiz': uses only horizonatal lines to link nodes/datums
    %             
    % maxL, the maximum number of samples to consider. Due to memory constraints,
    %               only the first maxL (6000 by default) points of time series are
    %               analyzed. Longer time series are reduced to their first maxL
    %               samples.
    % 
    % 
    %---OUTPUTS: statistics on the degree distribution, including the mode, mean,
    % spread, histogram entropy, and fits to gaussian, exponential, and powerlaw
    % distributions.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['dexpk_adjr2',
                                                      'dexpk_r2',
                                                      'dexpk_resAC1',
                                                      'dexpk_resAC2',
                                                      'dexpk_resruns',
                                                      'dexpk_rmse',
                                                      'dgaussk_adjr2',
                                                      'dgaussk_r2',
                                                      'dgaussk_resAC1',
                                                      'dgaussk_resAC2',
                                                      'dgaussk_resruns',
                                                      'dgaussk_rmse',
                                                      'dpowerk_adjr2',
                                                      'dpowerk_r2',
                                                      'dpowerk_resAC1',
                                                      'dpowerk_resAC2',
                                                      'dpowerk_resruns',
                                                      'dpowerk_rmse',
                                                      'evnlogL',
                                                      'evparm1',
                                                      'evparm2',
                                                      'explambda',
                                                      'expmu',
                                                      'expnlogL',
                                                      'gaussmu',
                                                      'gaussnlogL',
                                                      'gausssigma',
                                                      'iqrk',
                                                      'kac1',
                                                      'kac2',
                                                      'kac3',
                                                      'ktau',
                                                      'maxent',
                                                      'maxk',
                                                      'maxonmedian',
                                                      'meanchent',
                                                      'meanent',
                                                      'meank',
                                                      'mediank',
                                                      'mink',
                                                      'minnbinmaxent',
                                                      'modek',
                                                      'ol90',
                                                      'olu90',
                                                      'propmode',
                                                      'rangek',
                                                      'stdk']}
    if meth is None:
        out = eng.run_function(1, 'NW_VisibilityGraph', x, )
    elif maxL is None:
        out = eng.run_function(1, 'NW_VisibilityGraph', x, meth)
    else:
        out = eng.run_function(1, 'NW_VisibilityGraph', x, meth, maxL)
    return outfunc(out)


class NW_VisibilityGraph(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Constructs a visibility graph of the time series and returns various
    % statistics on the properties of the resulting network.
    % 
    % cf.: "From time series to complex networks: The visibility graph"
    % Lacasa, Lucas and Luque, Bartolo and Ballesteros, Fernando and Luque, Jordi
    % and Nuno, Juan Carlos P. Natl. Acad. Sci. USA. 105(13) 4972 (2008)
    % 
    % "Horizontal visibility graphs: Exact results for random time series"
    % Luque, B. and Lacasa, L. and Ballesteros, F. and Luque, J.
    % Phys. Rev. E. 80(4) 046103 (2009)
    % 
    % The normal visibility graph may not be implemented correctly, we focused only
    % on the horizontal visibility graph.
    % 
    %---INPUTS:
    % y, the time series (a column vector)
    % 
    % meth, the method for constructing:
    % 			(i) 'norm': the normal visibility definition
    % 			(ii) 'horiz': uses only horizonatal lines to link nodes/datums
    %             
    % maxL, the maximum number of samples to consider. Due to memory constraints,
    %               only the first maxL (6000 by default) points of time series are
    %               analyzed. Longer time series are reduced to their first maxL
    %               samples.
    % 
    % 
    %---OUTPUTS: statistics on the degree distribution, including the mode, mean,
    % spread, histogram entropy, and fits to gaussian, exponential, and powerlaw
    % distributions.
    % 
    ----------------------------------------
    """

    outnames = ('dexpk_adjr2',
                'dexpk_r2',
                'dexpk_resAC1',
                'dexpk_resAC2',
                'dexpk_resruns',
                'dexpk_rmse',
                'dgaussk_adjr2',
                'dgaussk_r2',
                'dgaussk_resAC1',
                'dgaussk_resAC2',
                'dgaussk_resruns',
                'dgaussk_rmse',
                'dpowerk_adjr2',
                'dpowerk_r2',
                'dpowerk_resAC1',
                'dpowerk_resAC2',
                'dpowerk_resruns',
                'dpowerk_rmse',
                'evnlogL',
                'evparm1',
                'evparm2',
                'explambda',
                'expmu',
                'expnlogL',
                'gaussmu',
                'gaussnlogL',
                'gausssigma',
                'iqrk',
                'kac1',
                'kac2',
                'kac3',
                'ktau',
                'maxent',
                'maxk',
                'maxonmedian',
                'meanchent',
                'meanent',
                'meank',
                'mediank',
                'mink',
                'minnbinmaxent',
                'modek',
                'ol90',
                'olu90',
                'propmode',
                'rangek',
                'stdk')

    def __init__(self, meth='horiz', maxL=None):
        super(NW_VisibilityGraph, self).__init__(add_descriptors=False)
        self.meth = meth
        self.maxL = maxL

    def eval(self, engine, x):
        return HCTSA_NW_VisibilityGraph(engine,
                                        x,
                                        meth=self.meth,
                                        maxL=self.maxL)


def HCTSA_PD_PeriodicityWang(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements an idea based on the periodicity extraction measure proposed in:
    % 
    % "Structure-based Statistical Features and Multivariate Time Series Clustering"
    % Wang, X. and Wirth, A. and Wang, L.
    % Seventh IEEE International Conference on Data Mining, 351--360 (2007)
    % DOI: 10.1109/ICDM.2007.103
    %
    % This function detrends the time series using a single-knot cubic regression
    % spline, and then computes autocorrelations up to one third of the length of
    % the time series. The frequency is the first peak in the autocorrelation
    % function satisfying a set of conditions.
    % 
    %---INPUTS:
    % y, the input time series.
    % 
    % The single threshold of 0.01 was considered in the original paper, this code
    % uses a range of thresholds: 0, 0.01, 0.1, 0.2, 1\sqrt{N}, 5\sqrt{N}, and
    % 10\sqrt{N}, where N is the length of the time series.
    % 
    %---HISTORY:
    % Ben Fulcher, 9/7/09
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['th1',
                                                      'th2',
                                                      'th3',
                                                      'th4',
                                                      'th5',
                                                      'th6',
                                                      'th7']}
    out = eng.run_function(1, 'PD_PeriodicityWang', x, )
    return outfunc(out)


class PD_PeriodicityWang(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements an idea based on the periodicity extraction measure proposed in:
    % 
    % "Structure-based Statistical Features and Multivariate Time Series Clustering"
    % Wang, X. and Wirth, A. and Wang, L.
    % Seventh IEEE International Conference on Data Mining, 351--360 (2007)
    % DOI: 10.1109/ICDM.2007.103
    %
    % This function detrends the time series using a single-knot cubic regression
    % spline, and then computes autocorrelations up to one third of the length of
    % the time series. The frequency is the first peak in the autocorrelation
    % function satisfying a set of conditions.
    % 
    %---INPUTS:
    % y, the input time series.
    % 
    % The single threshold of 0.01 was considered in the original paper, this code
    % uses a range of thresholds: 0, 0.01, 0.1, 0.2, 1\sqrt{N}, 5\sqrt{N}, and
    % 10\sqrt{N}, where N is the length of the time series.
    % 
    %---HISTORY:
    % Ben Fulcher, 9/7/09
    % 
    ----------------------------------------
    """

    outnames = ('th1',
                'th2',
                'th3',
                'th4',
                'th5',
                'th6',
                'th7')

    def __init__(self, ):
        super(PD_PeriodicityWang, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_PD_PeriodicityWang(engine, x)


def HCTSA_PH_ForcePotential(eng, x, whatpot='dblwell', params=(1, 0.2, 0.1)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Couples the values of the time series to a given dynamical system. The input
    % time series forces a particle in the given potential well.
    % 
    % The time series contributes to a forcing term on a simulated particle in a:
    % 
    % (i) Quartic double-well potential with potential energy V(x) = x^4/4 - alpha^2
    %           x^2/2, or a force F(x) = -x^3 + alpha^2 x
    % 
    % (ii) Sinusoidal potential with V(x) = -cos(x/alpha), or F(x) = sin(x/alpha)/alpha
    %
    %---INPUTS:
    % y, the input time series
    % 
    % whatpot, the potential function to simulate:
    %               (i) 'dblwell' (a double well potential function)
    %               (ii) 'sine' (a sinusoidal potential function)
    % 
    % params, the parameters for simulation, should be in the form:
    %                   params = [alpha, kappa, deltat]
    %                   
    %           (i) The double-well potential has three parameters:
    %               * alpha controls the relative positions of the wells,
    %               * kappa is the coefficient of friction,
    %               * deltat sets the time step for the simulation.
    %           
    %           (ii) The sinusoidal potential also has three parameters:
    %               * alpha controls the period of oscillations in the potential
    %               * kappa is the coefficient of friction,
    %               * deltat sets the time step for the simulation.
    % 
    %---OUTPUTS: statistics summarizing the trajectory of the simulated particle,
    % including its mean, the range, proportion positive, proportion of times it
    % crosses zero, its autocorrelation, final position, and standard deviation.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac10',
                                                      'ac50',
                                                      'finaldev',
                                                      'mean',
                                                      'median',
                                                      'pcross',
                                                      'pcrossdown',
                                                      'pcrossup',
                                                      'proppos',
                                                      'range',
                                                      'std',
                                                      'tau']}
    if whatpot is None:
        out = eng.run_function(1, 'PH_ForcePotential', x, )
    elif params is None:
        out = eng.run_function(1, 'PH_ForcePotential', x, whatpot)
    else:
        out = eng.run_function(1, 'PH_ForcePotential', x, whatpot, params)
    return outfunc(out)


class PH_ForcePotential(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Couples the values of the time series to a given dynamical system. The input
    % time series forces a particle in the given potential well.
    % 
    % The time series contributes to a forcing term on a simulated particle in a:
    % 
    % (i) Quartic double-well potential with potential energy V(x) = x^4/4 - alpha^2
    %           x^2/2, or a force F(x) = -x^3 + alpha^2 x
    % 
    % (ii) Sinusoidal potential with V(x) = -cos(x/alpha), or F(x) = sin(x/alpha)/alpha
    %
    %---INPUTS:
    % y, the input time series
    % 
    % whatpot, the potential function to simulate:
    %               (i) 'dblwell' (a double well potential function)
    %               (ii) 'sine' (a sinusoidal potential function)
    % 
    % params, the parameters for simulation, should be in the form:
    %                   params = [alpha, kappa, deltat]
    %                   
    %           (i) The double-well potential has three parameters:
    %               * alpha controls the relative positions of the wells,
    %               * kappa is the coefficient of friction,
    %               * deltat sets the time step for the simulation.
    %           
    %           (ii) The sinusoidal potential also has three parameters:
    %               * alpha controls the period of oscillations in the potential
    %               * kappa is the coefficient of friction,
    %               * deltat sets the time step for the simulation.
    % 
    %---OUTPUTS: statistics summarizing the trajectory of the simulated particle,
    % including its mean, the range, proportion positive, proportion of times it
    % crosses zero, its autocorrelation, final position, and standard deviation.
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac10',
                'ac50',
                'finaldev',
                'mean',
                'median',
                'pcross',
                'pcrossdown',
                'pcrossup',
                'proppos',
                'range',
                'std',
                'tau')

    def __init__(self, whatpot='dblwell', params=(1, 0.2, 0.1)):
        super(PH_ForcePotential, self).__init__(add_descriptors=False)
        self.whatpot = whatpot
        self.params = params

    def eval(self, engine, x):
        return HCTSA_PH_ForcePotential(engine,
                                       x,
                                       whatpot=self.whatpot,
                                       params=self.params)


def HCTSA_PH_Walker(eng, x, walkerrule='prop', wparam=0.9):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This operation simulates a hypothetical particle (or 'walker'), that moves in
    % the time domain in response to values of the time series at each point.
    % 
    % Outputs from this operation are summaries of the walker's motion, and
    % comparisons of it to the original time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % walkerrule, the kinematic rule by which the walker moves in response to the
    %             time series over time:
    %             
    %             (i) 'prop': the walker narrows the gap between its value and that
    %                         of the time series by a given proportion p.
    %                         wparam = p;
    %                         
    %             (ii) 'biasprop': the walker is biased to move more in one
    %                          direction; when it is being pushed up by the time
    %                          series, it narrows the gap by a proportion p_{up},
    %                          and when it is being pushed down by the time series,
    %                          it narrows the gap by a (potentially different)
    %                          proportion p_{down}. wparam = [pup,pdown].
    %                          
    %             (iii) 'momentum': the walker moves as if it has mass m and inertia
    %                          from the previous time step and the time series acts
    %                          as a force altering its motion in a classical
    %                          Newtonian dynamics framework. [wparam = m], the mass.
    %                          
    %              (iv) 'runningvar': the walker moves with inertia as above, but
    %                          its values are also adjusted so as to match the local
    %                          variance of time series by a multiplicative factor.
    %                          wparam = [m,wl], where m is the inertial mass and wl
    %                          is the window length.
    % 
    % wparam, the parameters for the specified walkerrule, explained above.
    % 
    %---OUTPUTS: include the mean, spread, maximum, minimum, and autocorrelation of the
    % walker's trajectory, the number of crossings between the walker and the
    % original time series, the ratio or difference of some basic summary statistics
    % between the original time series and the walker, an Ansari-Bradley test
    % comparing the distributions of the walker and original time series, and
    % various statistics summarizing properties of the residuals between the
    % walker's trajectory and the original time series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['res_ac1',
                                                      'res_runstest',
                                                      'res_swss5_1',
                                                      'sw_ac1rat',
                                                      'sw_ansarib_pval',
                                                      'sw_distdiff',
                                                      'sw_maxrat',
                                                      'sw_meanabsdiff',
                                                      'sw_minrat',
                                                      'sw_propcross',
                                                      'sw_stdrat',
                                                      'sw_taudiff',
                                                      'w_ac1',
                                                      'w_ac2',
                                                      'w_max',
                                                      'w_mean',
                                                      'w_median',
                                                      'w_min',
                                                      'w_propzcross',
                                                      'w_std',
                                                      'w_tau']}
    if walkerrule is None:
        out = eng.run_function(1, 'PH_Walker', x, )
    elif wparam is None:
        out = eng.run_function(1, 'PH_Walker', x, walkerrule)
    else:
        out = eng.run_function(1, 'PH_Walker', x, walkerrule, wparam)
    return outfunc(out)


class PH_Walker(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This operation simulates a hypothetical particle (or 'walker'), that moves in
    % the time domain in response to values of the time series at each point.
    % 
    % Outputs from this operation are summaries of the walker's motion, and
    % comparisons of it to the original time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % walkerrule, the kinematic rule by which the walker moves in response to the
    %             time series over time:
    %             
    %             (i) 'prop': the walker narrows the gap between its value and that
    %                         of the time series by a given proportion p.
    %                         wparam = p;
    %                         
    %             (ii) 'biasprop': the walker is biased to move more in one
    %                          direction; when it is being pushed up by the time
    %                          series, it narrows the gap by a proportion p_{up},
    %                          and when it is being pushed down by the time series,
    %                          it narrows the gap by a (potentially different)
    %                          proportion p_{down}. wparam = [pup,pdown].
    %                          
    %             (iii) 'momentum': the walker moves as if it has mass m and inertia
    %                          from the previous time step and the time series acts
    %                          as a force altering its motion in a classical
    %                          Newtonian dynamics framework. [wparam = m], the mass.
    %                          
    %              (iv) 'runningvar': the walker moves with inertia as above, but
    %                          its values are also adjusted so as to match the local
    %                          variance of time series by a multiplicative factor.
    %                          wparam = [m,wl], where m is the inertial mass and wl
    %                          is the window length.
    % 
    % wparam, the parameters for the specified walkerrule, explained above.
    % 
    %---OUTPUTS: include the mean, spread, maximum, minimum, and autocorrelation of the
    % walker's trajectory, the number of crossings between the walker and the
    % original time series, the ratio or difference of some basic summary statistics
    % between the original time series and the walker, an Ansari-Bradley test
    % comparing the distributions of the walker and original time series, and
    % various statistics summarizing properties of the residuals between the
    % walker's trajectory and the original time series.
    % 
    ----------------------------------------
    """

    outnames = ('res_ac1',
                'res_runstest',
                'res_swss5_1',
                'sw_ac1rat',
                'sw_ansarib_pval',
                'sw_distdiff',
                'sw_maxrat',
                'sw_meanabsdiff',
                'sw_minrat',
                'sw_propcross',
                'sw_stdrat',
                'sw_taudiff',
                'w_ac1',
                'w_ac2',
                'w_max',
                'w_mean',
                'w_median',
                'w_min',
                'w_propzcross',
                'w_std',
                'w_tau')

    def __init__(self, walkerrule='prop', wparam=0.9):
        super(PH_Walker, self).__init__(add_descriptors=False)
        self.walkerrule = walkerrule
        self.wparam = wparam

    def eval(self, engine, x):
        return HCTSA_PH_Walker(engine,
                               x,
                               walkerrule=self.walkerrule,
                               wparam=self.wparam)


def HCTSA_PP_Compare(eng, x, detrndmeth='medianf4'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Applies a given pre-processing transformation to the time series, and returns
    % statistics on how various time-series properties change as a result.
    % 
    % Inputs are structured in a clunky way, unfortunately...
    % 
    %---INPUTS:
    % y, the input time series
    % detrndmeth, the method to use for detrending:
    %      (i) 'poly': polynomial detrendings, both linear and quadratic. Can
    %                  be of the following forms:
    %            (a) polynomial of given order: 'poly1', 'poly2', 'poly3',
    %                'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'poly9'
    %            (b) fit best polynomial: 'polybest' determines 'best' by
    %                            various tests (e.g., whiteness of residuals,
    %                            etc.)
    %            (c) 'fitstrong' only fits if a 'strong' trend.
    %      (ii) 'sin': sinusoidal detrending with either one or two frequency
    %                components,
    %            (a) fit a sine series of a given order
    %               Fits a form like: a1*sin(b1*x+c1) + a2*sin(b2*x+c2) + ...
    %               Additional number determines how many terms to include in the
    %               series: 'sin1', 'sin2', 'sin3', 'sin4', 'sin5', 'sin6', 'sin7',
    %               'sin8'
    %            (b) 'sin_st1': fit only if a strong trend (i.e., if the amplitudes
    %                                           are above a given threshold)
    %      (iii) 'spline': removes a least squares spline using Matlab's
    %                      Spline Toolbox function spap2
    %                      Input of the form 'spline<nknots><interpolant_order>'
    %                      e.g., 'spline45' uses four knots and 5th order
    %                      interpolants (Implements a least squares spline via the
    %                      spline toolbox function spap2)
    %      (iv) 'diff': takes incremental differences of the time series. Of form
    %                 'diff<ndiff>' where ndiff is the number of differencings to
    %                 perform. e.g., 'diff3' performs three recursive differences
    %      (v) 'medianf': a running median filter using a given window lengths
    %                   Of form 'medianf<n>' where n is the window length.
    %                   e.g., 'medianf3' takes a running median using the median of
    %                     every 3 consecutive values as a point in the filtered
    %                     time series. Uses the Signal Processing Toolbox
    %                     function medfilt1
    %      (vi) 'rav': running mean filter of a given window length.
    %                  Uses Matlab's filter function to perform a running
    %                  average of order n. Of form 'rav<n>' where n is the order of
    %                  the running average.
    %      (vii) 'resample': resamples the data by a given ratio using the resample
    %                        function in Matlab.
    %                        Of form 'resample_<p>_<q>', where the ratio p/q is the
    %                        new sampling rate e.g., 'resample_1_2' will downsample
    %                        the signal by one half e.g., resample_10_1' will
    %                        resample the signal to 10 times its original length
    %      (viii) 'logr': takes log returns of the data. Only valid for positive
    %                       data, else returns a NaN.
    %      (ix) 'boxcox': makes a Box-Cox transformation of the data. Only valid for
    %                     positive only data; otherwise returns a NaN.
    % 
    % If multiple detrending methods are specified, they should be in a cell of
    % strings; the methods will be executed in order, e.g., {'poly1','sin1'} does a
    % linear polynomial then a simple one-frequency seasonal detrend (in that order)
    % 
    %---OUTPUTS: include comparisons of stationarity and distributional measures
    % between the original and transformed time series.
    % 
    %---HISTORY:
    % Ben Fulcher, 9/7/09
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['gauss1_h10_adjr2',
                                                      'gauss1_h10_r2',
                                                      'gauss1_h10_resAC1',
                                                      'gauss1_h10_resAC2',
                                                      'gauss1_h10_resruns',
                                                      'gauss1_h10_rmse',
                                                      'gauss1_kd_adjr2',
                                                      'gauss1_kd_r2',
                                                      'gauss1_kd_resAC1',
                                                      'gauss1_kd_resAC2',
                                                      'gauss1_kd_resruns',
                                                      'gauss1_kd_rmse',
                                                      'htdt_chi2n',
                                                      'htdt_ksn',
                                                      'htdt_llfn',
                                                      'kscn_adiff',
                                                      'kscn_olapint',
                                                      'kscn_peaksepx',
                                                      'kscn_peaksepy',
                                                      'kscn_relent',
                                                      'olbt_m2',
                                                      'olbt_m5',
                                                      'olbt_s2',
                                                      'olbt_s5',
                                                      'statav10',
                                                      'statav2',
                                                      'statav4',
                                                      'statav6',
                                                      'statav8',
                                                      'swms10_1',
                                                      'swms2_1',
                                                      'swms2_2',
                                                      'swms5_1',
                                                      'swms5_2',
                                                      'swss10_1',
                                                      'swss10_2',
                                                      'swss2_1',
                                                      'swss2_2',
                                                      'swss5_1',
                                                      'swss5_2']}
    if detrndmeth is None:
        out = eng.run_function(1, 'PP_Compare', x, )
    else:
        out = eng.run_function(1, 'PP_Compare', x, detrndmeth)
    return outfunc(out)


class PP_Compare(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Applies a given pre-processing transformation to the time series, and returns
    % statistics on how various time-series properties change as a result.
    % 
    % Inputs are structured in a clunky way, unfortunately...
    % 
    %---INPUTS:
    % y, the input time series
    % detrndmeth, the method to use for detrending:
    %      (i) 'poly': polynomial detrendings, both linear and quadratic. Can
    %                  be of the following forms:
    %            (a) polynomial of given order: 'poly1', 'poly2', 'poly3',
    %                'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'poly9'
    %            (b) fit best polynomial: 'polybest' determines 'best' by
    %                            various tests (e.g., whiteness of residuals,
    %                            etc.)
    %            (c) 'fitstrong' only fits if a 'strong' trend.
    %      (ii) 'sin': sinusoidal detrending with either one or two frequency
    %                components,
    %            (a) fit a sine series of a given order
    %               Fits a form like: a1*sin(b1*x+c1) + a2*sin(b2*x+c2) + ...
    %               Additional number determines how many terms to include in the
    %               series: 'sin1', 'sin2', 'sin3', 'sin4', 'sin5', 'sin6', 'sin7',
    %               'sin8'
    %            (b) 'sin_st1': fit only if a strong trend (i.e., if the amplitudes
    %                                           are above a given threshold)
    %      (iii) 'spline': removes a least squares spline using Matlab's
    %                      Spline Toolbox function spap2
    %                      Input of the form 'spline<nknots><interpolant_order>'
    %                      e.g., 'spline45' uses four knots and 5th order
    %                      interpolants (Implements a least squares spline via the
    %                      spline toolbox function spap2)
    %      (iv) 'diff': takes incremental differences of the time series. Of form
    %                 'diff<ndiff>' where ndiff is the number of differencings to
    %                 perform. e.g., 'diff3' performs three recursive differences
    %      (v) 'medianf': a running median filter using a given window lengths
    %                   Of form 'medianf<n>' where n is the window length.
    %                   e.g., 'medianf3' takes a running median using the median of
    %                     every 3 consecutive values as a point in the filtered
    %                     time series. Uses the Signal Processing Toolbox
    %                     function medfilt1
    %      (vi) 'rav': running mean filter of a given window length.
    %                  Uses Matlab's filter function to perform a running
    %                  average of order n. Of form 'rav<n>' where n is the order of
    %                  the running average.
    %      (vii) 'resample': resamples the data by a given ratio using the resample
    %                        function in Matlab.
    %                        Of form 'resample_<p>_<q>', where the ratio p/q is the
    %                        new sampling rate e.g., 'resample_1_2' will downsample
    %                        the signal by one half e.g., resample_10_1' will
    %                        resample the signal to 10 times its original length
    %      (viii) 'logr': takes log returns of the data. Only valid for positive
    %                       data, else returns a NaN.
    %      (ix) 'boxcox': makes a Box-Cox transformation of the data. Only valid for
    %                     positive only data; otherwise returns a NaN.
    % 
    % If multiple detrending methods are specified, they should be in a cell of
    % strings; the methods will be executed in order, e.g., {'poly1','sin1'} does a
    % linear polynomial then a simple one-frequency seasonal detrend (in that order)
    % 
    %---OUTPUTS: include comparisons of stationarity and distributional measures
    % between the original and transformed time series.
    % 
    %---HISTORY:
    % Ben Fulcher, 9/7/09
    % 
    ----------------------------------------
    """

    outnames = ('gauss1_h10_adjr2',
                'gauss1_h10_r2',
                'gauss1_h10_resAC1',
                'gauss1_h10_resAC2',
                'gauss1_h10_resruns',
                'gauss1_h10_rmse',
                'gauss1_kd_adjr2',
                'gauss1_kd_r2',
                'gauss1_kd_resAC1',
                'gauss1_kd_resAC2',
                'gauss1_kd_resruns',
                'gauss1_kd_rmse',
                'htdt_chi2n',
                'htdt_ksn',
                'htdt_llfn',
                'kscn_adiff',
                'kscn_olapint',
                'kscn_peaksepx',
                'kscn_peaksepy',
                'kscn_relent',
                'olbt_m2',
                'olbt_m5',
                'olbt_s2',
                'olbt_s5',
                'statav10',
                'statav2',
                'statav4',
                'statav6',
                'statav8',
                'swms10_1',
                'swms2_1',
                'swms2_2',
                'swms5_1',
                'swms5_2',
                'swss10_1',
                'swss10_2',
                'swss2_1',
                'swss2_2',
                'swss5_1',
                'swss5_2')

    def __init__(self, detrndmeth='medianf4'):
        super(PP_Compare, self).__init__(add_descriptors=False)
        self.detrndmeth = detrndmeth

    def eval(self, engine, x):
        return HCTSA_PP_Compare(engine,
                                x,
                                detrndmeth=self.detrndmeth)


def HCTSA_PP_Iterate(eng, x, detrndmeth='diff'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Iteratively applies a transformation to the time series and measures how
    % various properties of the time series change as the transformation is
    % iteratively applied to it.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % detrndmeth, the detrending method to apply:
    %           (i) 'spline' removes a spine fit,
    %           (ii) 'diff' takes incremental differences,
    %           (iii) 'medianf' applies a median filter,
    %           (iv) 'rav' applies a running mean filter,
    %           (v) 'resampleup' progressively upsamples the time series,
    %           (vi) 'resampledown' progressively downsamples the time series.
    %
    %---HISTORY:
    % Ben Fulcher, 10/7/09
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['gauss1_h10_exp',
                                                      'gauss1_h10_jump',
                                                      'gauss1_h10_lin',
                                                      'gauss1_h10_trend',
                                                      'gauss1_kd_exp',
                                                      'gauss1_kd_jump',
                                                      'gauss1_kd_lin',
                                                      'gauss1_kd_trend',
                                                      'norm_kscomp_exp',
                                                      'norm_kscomp_jump',
                                                      'norm_kscomp_lin',
                                                      'norm_kscomp_trend',
                                                      'normdiff_exp',
                                                      'normdiff_jump',
                                                      'normdiff_lin',
                                                      'normdiff_trend',
                                                      'ol_exp',
                                                      'ol_jump',
                                                      'ol_lin',
                                                      'ol_trend',
                                                      'statav5_exp',
                                                      'statav5_jump',
                                                      'statav5_lin',
                                                      'statav5_trend',
                                                      'swms5_2_exp',
                                                      'swms5_2_jump',
                                                      'swms5_2_lin',
                                                      'swms5_2_trend',
                                                      'swss5_2_exp',
                                                      'swss5_2_jump',
                                                      'swss5_2_lin',
                                                      'swss5_2_trend',
                                                      'xc1_exp',
                                                      'xc1_jump',
                                                      'xc1_lin',
                                                      'xc1_trend',
                                                      'xcn1_exp',
                                                      'xcn1_jump',
                                                      'xcn1_lin',
                                                      'xcn1_trend']}
    if detrndmeth is None:
        out = eng.run_function(1, 'PP_Iterate', x, )
    else:
        out = eng.run_function(1, 'PP_Iterate', x, detrndmeth)
    return outfunc(out)


class PP_Iterate(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Iteratively applies a transformation to the time series and measures how
    % various properties of the time series change as the transformation is
    % iteratively applied to it.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % detrndmeth, the detrending method to apply:
    %           (i) 'spline' removes a spine fit,
    %           (ii) 'diff' takes incremental differences,
    %           (iii) 'medianf' applies a median filter,
    %           (iv) 'rav' applies a running mean filter,
    %           (v) 'resampleup' progressively upsamples the time series,
    %           (vi) 'resampledown' progressively downsamples the time series.
    %
    %---HISTORY:
    % Ben Fulcher, 10/7/09
    % 
    ----------------------------------------
    """

    outnames = ('gauss1_h10_exp',
                'gauss1_h10_jump',
                'gauss1_h10_lin',
                'gauss1_h10_trend',
                'gauss1_kd_exp',
                'gauss1_kd_jump',
                'gauss1_kd_lin',
                'gauss1_kd_trend',
                'norm_kscomp_exp',
                'norm_kscomp_jump',
                'norm_kscomp_lin',
                'norm_kscomp_trend',
                'normdiff_exp',
                'normdiff_jump',
                'normdiff_lin',
                'normdiff_trend',
                'ol_exp',
                'ol_jump',
                'ol_lin',
                'ol_trend',
                'statav5_exp',
                'statav5_jump',
                'statav5_lin',
                'statav5_trend',
                'swms5_2_exp',
                'swms5_2_jump',
                'swms5_2_lin',
                'swms5_2_trend',
                'swss5_2_exp',
                'swss5_2_jump',
                'swss5_2_lin',
                'swss5_2_trend',
                'xc1_exp',
                'xc1_jump',
                'xc1_lin',
                'xc1_trend',
                'xcn1_exp',
                'xcn1_jump',
                'xcn1_lin',
                'xcn1_trend')

    def __init__(self, detrndmeth='diff'):
        super(PP_Iterate, self).__init__(add_descriptors=False)
        self.detrndmeth = detrndmeth

    def eval(self, engine, x):
        return HCTSA_PP_Iterate(engine,
                                x,
                                detrndmeth=self.detrndmeth)


def HCTSA_PP_ModelFit(eng, x, model='ar', order=2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Carries out a variety of preprocessings to look at improvement of fit to
    % an AR model.
    % 
    % After performing the range of transformations to the time series, returns the
    % in-sample root-mean-square (RMS) prediction errors for an AR model on each
    % transformed time series as a ratio of the RMS prediction error of the original
    % time series.
    % 
    % PP_PreProcess.m is used to perform the preprocessings
    % 
    % The AR model is fitted using the function ar and pe from Matlab's System
    % Identification Toolbox
    % 
    % Transformations performed include:
    % (i) incremental differencing,
    % (ii) filtering of the power spectral density function,
    % (iii) removal of piece-wise polynomial trends, and
    % (iv) rank mapping the values of the time series to a Gaussian distribution.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % model, the time-series model to fit to the transformed time series (currently
    %           'ar' is the only option)
    %           
    % order, the order of the AR model to fit to the data
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['rmserrrat_d1',
                                                      'rmserrrat_d2',
                                                      'rmserrrat_d3',
                                                      'rmserrrat_lf_02',
                                                      'rmserrrat_lf_02_d1',
                                                      'rmserrrat_p1_10',
                                                      'rmserrrat_p1_20',
                                                      'rmserrrat_p1_40',
                                                      'rmserrrat_p1_5',
                                                      'rmserrrat_p2_10',
                                                      'rmserrrat_p2_20',
                                                      'rmserrrat_p2_40',
                                                      'rmserrrat_p2_5',
                                                      'rmserrrat_peaks_08',
                                                      'rmserrrat_peaks_08_d1',
                                                      'rmserrrat_rmgd']}
    if model is None:
        out = eng.run_function(1, 'PP_ModelFit', x, )
    elif order is None:
        out = eng.run_function(1, 'PP_ModelFit', x, model)
    else:
        out = eng.run_function(1, 'PP_ModelFit', x, model, order)
    return outfunc(out)


class PP_ModelFit(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Carries out a variety of preprocessings to look at improvement of fit to
    % an AR model.
    % 
    % After performing the range of transformations to the time series, returns the
    % in-sample root-mean-square (RMS) prediction errors for an AR model on each
    % transformed time series as a ratio of the RMS prediction error of the original
    % time series.
    % 
    % PP_PreProcess.m is used to perform the preprocessings
    % 
    % The AR model is fitted using the function ar and pe from Matlab's System
    % Identification Toolbox
    % 
    % Transformations performed include:
    % (i) incremental differencing,
    % (ii) filtering of the power spectral density function,
    % (iii) removal of piece-wise polynomial trends, and
    % (iv) rank mapping the values of the time series to a Gaussian distribution.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % model, the time-series model to fit to the transformed time series (currently
    %           'ar' is the only option)
    %           
    % order, the order of the AR model to fit to the data
    % 
    ----------------------------------------
    """

    outnames = ('rmserrrat_d1',
                'rmserrrat_d2',
                'rmserrrat_d3',
                'rmserrrat_lf_02',
                'rmserrrat_lf_02_d1',
                'rmserrrat_p1_10',
                'rmserrrat_p1_20',
                'rmserrrat_p1_40',
                'rmserrrat_p1_5',
                'rmserrrat_p2_10',
                'rmserrrat_p2_20',
                'rmserrrat_p2_40',
                'rmserrrat_p2_5',
                'rmserrrat_peaks_08',
                'rmserrrat_peaks_08_d1',
                'rmserrrat_rmgd')

    def __init__(self, model='ar', order=2):
        super(PP_ModelFit, self).__init__(add_descriptors=False)
        self.model = model
        self.order = order

    def eval(self, engine, x):
        return HCTSA_PP_ModelFit(engine,
                                 x,
                                 model=self.model,
                                 order=self.order)


def HCTSA_PP_PreProcess(eng, x, choosebest=None, order=None, beatthis=None, dospectral=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a bunch of time series in the structure yp that have been
    % preprocessed in a number of different ways.
    % 
    %---INPUTS:
    % y, the input time series
    % choosebest: (i) '' (empty): the function returns a structure, yp, of all
    %                   preprocessings [default]
    %             (ii) 'ar': returns the pre-processing with the worst fit to an AR
    %                        model of specified order
    %             (iii) 'ac': returns the least autocorrelated pre-processing
    % order [opt], extra parameter for above
    % beatthis, can specify the preprocessed version has to be some percentage
    %           better than the unprocessed input
    % dospectral, whether to include spectral processing
    % 
    % 
    % If second argument is specified, will choose amongst the preprocessings
    % for the 'best' one according to the given criterion.
    % Based on (really improvement/development of) PP_ModelFit.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if choosebest is None:
        out = eng.run_function(1, 'PP_PreProcess', x, )
    elif order is None:
        out = eng.run_function(1, 'PP_PreProcess', x, choosebest)
    elif beatthis is None:
        out = eng.run_function(1, 'PP_PreProcess', x, choosebest, order)
    elif dospectral is None:
        out = eng.run_function(1, 'PP_PreProcess', x, choosebest, order, beatthis)
    else:
        out = eng.run_function(1, 'PP_PreProcess', x, choosebest, order, beatthis, dospectral)
    return outfunc(out)


class PP_PreProcess(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a bunch of time series in the structure yp that have been
    % preprocessed in a number of different ways.
    % 
    %---INPUTS:
    % y, the input time series
    % choosebest: (i) '' (empty): the function returns a structure, yp, of all
    %                   preprocessings [default]
    %             (ii) 'ar': returns the pre-processing with the worst fit to an AR
    %                        model of specified order
    %             (iii) 'ac': returns the least autocorrelated pre-processing
    % order [opt], extra parameter for above
    % beatthis, can specify the preprocessed version has to be some percentage
    %           better than the unprocessed input
    % dospectral, whether to include spectral processing
    % 
    % 
    % If second argument is specified, will choose amongst the preprocessings
    % for the 'best' one according to the given criterion.
    % Based on (really improvement/development of) PP_ModelFit.
    % 
    ----------------------------------------
    """

    def __init__(self, choosebest=None, order=None, beatthis=None, dospectral=None):
        super(PP_PreProcess, self).__init__(add_descriptors=False)
        self.choosebest = choosebest
        self.order = order
        self.beatthis = beatthis
        self.dospectral = dospectral

    def eval(self, engine, x):
        return HCTSA_PP_PreProcess(engine,
                                   x,
                                   choosebest=self.choosebest,
                                   order=self.order,
                                   beatthis=self.beatthis,
                                   dospectral=self.dospectral)


def HCTSA_RN_Gaussian(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a random number drawn from a normal distribution.
    % 
    % INPUT:
    % y, the input time series (but this input isn't used so it could be anything)
    % 
    % This operation was sometimes used as a control, as a kind of 'null operation'
    % with which to compare to the performance of other operations that use
    % informative properties of the time-series data, but is otherwise useless for
    % any real analysis. It is NOT included in the set of default operations.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'RN_Gaussian', x, )
    return outfunc(out)


class RN_Gaussian(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a random number drawn from a normal distribution.
    % 
    % INPUT:
    % y, the input time series (but this input isn't used so it could be anything)
    % 
    % This operation was sometimes used as a control, as a kind of 'null operation'
    % with which to compare to the performance of other operations that use
    % informative properties of the time-series data, but is otherwise useless for
    % any real analysis. It is NOT included in the set of default operations.
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(RN_Gaussian, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_RN_Gaussian(engine, x)


def HCTSA_SB_BinaryStats(eng, x, binarymeth='diff'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on a binary symbolization of the time series (to a symbolic
    % string of 0s and 1s).
    % 
    % Provides information about the coarse-grained behavior of the time series
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % binarymeth, the symbolization rule:
    %         (i) 'diff': by whether incremental differences of the time series are
    %                      positive (1), or negative (0),
    %          (ii) 'mean': by whether each point is above (1) or below the mean (0)
    %          (iii) 'iqr': by whether the time series is within the interquartile range
    %                      (1), or not (0).
    % 
    %---OUTPUTS: include the Shannon entropy of the string, the longest stretches of 0s
    % or 1s, the mean length of consecutive 0s or 1s, and the spread of consecutive
    % strings of 0s or 1s.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['h',
                                                      'longstretch0',
                                                      'longstretch1',
                                                      'meanstretch0',
                                                      'meanstretch1',
                                                      'meanstretchrat',
                                                      'pstretch0',
                                                      'pstretch1',
                                                      'pstretches',
                                                      'pup',
                                                      'pupstat2',
                                                      'rat21stretch0',
                                                      'rat21stretch1',
                                                      'stdstretch0',
                                                      'stdstretch1',
                                                      'stdstretchrat']}
    if binarymeth is None:
        out = eng.run_function(1, 'SB_BinaryStats', x, )
    else:
        out = eng.run_function(1, 'SB_BinaryStats', x, binarymeth)
    return outfunc(out)


class SB_BinaryStats(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns statistics on a binary symbolization of the time series (to a symbolic
    % string of 0s and 1s).
    % 
    % Provides information about the coarse-grained behavior of the time series
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % binarymeth, the symbolization rule:
    %         (i) 'diff': by whether incremental differences of the time series are
    %                      positive (1), or negative (0),
    %          (ii) 'mean': by whether each point is above (1) or below the mean (0)
    %          (iii) 'iqr': by whether the time series is within the interquartile range
    %                      (1), or not (0).
    % 
    %---OUTPUTS: include the Shannon entropy of the string, the longest stretches of 0s
    % or 1s, the mean length of consecutive 0s or 1s, and the spread of consecutive
    % strings of 0s or 1s.
    % 
    ----------------------------------------
    """

    outnames = ('h',
                'longstretch0',
                'longstretch1',
                'meanstretch0',
                'meanstretch1',
                'meanstretchrat',
                'pstretch0',
                'pstretch1',
                'pstretches',
                'pup',
                'pupstat2',
                'rat21stretch0',
                'rat21stretch1',
                'stdstretch0',
                'stdstretch1',
                'stdstretchrat')

    def __init__(self, binarymeth='diff'):
        super(SB_BinaryStats, self).__init__(add_descriptors=False)
        self.binarymeth = binarymeth

    def eval(self, engine, x):
        return HCTSA_SB_BinaryStats(engine,
                                    x,
                                    binarymeth=self.binarymeth)


def HCTSA_SB_BinaryStretch(eng, x, stretchwhat='lseq0'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the longest stretch of consecutive zeros or ones in a symbolized time
    % series as a proportion of the time-series length.
    % 
    % The time series is symbolized to a binary string by whether it's above (1) or
    % below (0) its mean.
    % 
    % It doesn't actually measure this correctly, due to an error in the code, but
    % it's still kind of an interesting operation...?!
    % 
    %---INPUTS:
    % x, the input time series
    % stretchwhat, (i) 'lseq1', measures something related to consecutive 1s
    %              (ii) 'lseq0', measures something related to consecutive 0s
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if stretchwhat is None:
        out = eng.run_function(1, 'SB_BinaryStretch', x, )
    else:
        out = eng.run_function(1, 'SB_BinaryStretch', x, stretchwhat)
    return outfunc(out)


class SB_BinaryStretch(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the longest stretch of consecutive zeros or ones in a symbolized time
    % series as a proportion of the time-series length.
    % 
    % The time series is symbolized to a binary string by whether it's above (1) or
    % below (0) its mean.
    % 
    % It doesn't actually measure this correctly, due to an error in the code, but
    % it's still kind of an interesting operation...?!
    % 
    %---INPUTS:
    % x, the input time series
    % stretchwhat, (i) 'lseq1', measures something related to consecutive 1s
    %              (ii) 'lseq0', measures something related to consecutive 0s
    % 
    ----------------------------------------
    """

    def __init__(self, stretchwhat='lseq0'):
        super(SB_BinaryStretch, self).__init__(add_descriptors=False)
        self.stretchwhat = stretchwhat

    def eval(self, engine, x):
        return HCTSA_SB_BinaryStretch(engine,
                                      x,
                                      stretchwhat=self.stretchwhat)


def HCTSA_SB_CoarseGrain(eng, x, howtocg=None, ng=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Coarse-grains the continuous time series to a discrete alphabet
    % by a given method.
    % 
    %---INPUTS:
    % howtocg, the method of coarse-graining
    % 
    % ng, either specifies the size of the alphabet for 'quantile' and 'updown'
    %       or sets the timedelay for the embedding subroutines
    % 
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if howtocg is None:
        out = eng.run_function(1, 'SB_CoarseGrain', x, )
    elif ng is None:
        out = eng.run_function(1, 'SB_CoarseGrain', x, howtocg)
    else:
        out = eng.run_function(1, 'SB_CoarseGrain', x, howtocg, ng)
    return outfunc(out)


class SB_CoarseGrain(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Coarse-grains the continuous time series to a discrete alphabet
    % by a given method.
    % 
    %---INPUTS:
    % howtocg, the method of coarse-graining
    % 
    % ng, either specifies the size of the alphabet for 'quantile' and 'updown'
    %       or sets the timedelay for the embedding subroutines
    % 
    % 
    ----------------------------------------
    """

    def __init__(self, howtocg=None, ng=None):
        super(SB_CoarseGrain, self).__init__(add_descriptors=False)
        self.howtocg = howtocg
        self.ng = ng

    def eval(self, engine, x):
        return HCTSA_SB_CoarseGrain(engine,
                                    x,
                                    howtocg=self.howtocg,
                                    ng=self.ng)


def HCTSA_SB_MotifThree(eng, x, trit='diffquant'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % As for SB_MotifTwo, but using an alphabet of three letters, i.e., looks for
    % motifs in a course-graining of the time series to an alphabet of three letters
    % 
    %---INPUTS:
    % y, time series to analyze
    % trit, the coarse-graining method to use:
    %       (i) 'quantile': equiprobable alphabet by time-series value
    %       (ii) 'diffquant': equiprobably alphabet by time-series increments
    % 
    %---OUTPUTS:
    % Statistics on words of length 1, 2, 3, and 4.
    % 
    %---HISTORY:
    % This code was laboriously written by Ben Fulcher, 2009.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['aa',
                                                      'aaa',
                                                      'aaaa',
                                                      'aaab',
                                                      'aaac',
                                                      'aab',
                                                      'aaba',
                                                      'aabb',
                                                      'aabc',
                                                      'aac',
                                                      'aaca',
                                                      'aacb',
                                                      'aacc',
                                                      'ab',
                                                      'aba',
                                                      'abaa',
                                                      'abab',
                                                      'abac',
                                                      'abb',
                                                      'abba',
                                                      'abbb',
                                                      'abbc',
                                                      'abc',
                                                      'abca',
                                                      'abcb',
                                                      'abcc',
                                                      'ac',
                                                      'aca',
                                                      'acaa',
                                                      'acab',
                                                      'acac',
                                                      'acb',
                                                      'acba',
                                                      'acbb',
                                                      'acbc',
                                                      'acc',
                                                      'acca',
                                                      'accb',
                                                      'accc',
                                                      'ba',
                                                      'baa',
                                                      'baaa',
                                                      'baab',
                                                      'baac',
                                                      'bab',
                                                      'baba',
                                                      'babb',
                                                      'babc',
                                                      'bac',
                                                      'baca',
                                                      'bacb',
                                                      'bacc',
                                                      'bb',
                                                      'bba',
                                                      'bbaa',
                                                      'bbab',
                                                      'bbac',
                                                      'bbb',
                                                      'bbba',
                                                      'bbbb',
                                                      'bbbc',
                                                      'bbc',
                                                      'bbca',
                                                      'bbcb',
                                                      'bbcc',
                                                      'bc',
                                                      'bca',
                                                      'bcaa',
                                                      'bcab',
                                                      'bcac',
                                                      'bcb',
                                                      'bcba',
                                                      'bcbb',
                                                      'bcbc',
                                                      'bcc',
                                                      'bcca',
                                                      'bccb',
                                                      'bccc',
                                                      'ca',
                                                      'caa',
                                                      'caaa',
                                                      'caab',
                                                      'caac',
                                                      'cab',
                                                      'caba',
                                                      'cabb',
                                                      'cabc',
                                                      'cac',
                                                      'caca',
                                                      'cacb',
                                                      'cacc',
                                                      'cb',
                                                      'cba',
                                                      'cbaa',
                                                      'cbab',
                                                      'cbac',
                                                      'cbb',
                                                      'cbba',
                                                      'cbbb',
                                                      'cbbc',
                                                      'cbc',
                                                      'cbca',
                                                      'cbcb',
                                                      'cbcc',
                                                      'cc',
                                                      'cca',
                                                      'ccaa',
                                                      'ccab',
                                                      'ccac',
                                                      'ccb',
                                                      'ccba',
                                                      'ccbb',
                                                      'ccbc',
                                                      'ccc',
                                                      'ccca',
                                                      'cccb',
                                                      'cccc',
                                                      'hh',
                                                      'hhh',
                                                      'hhhh']}
    if trit is None:
        out = eng.run_function(1, 'SB_MotifThree', x, )
    else:
        out = eng.run_function(1, 'SB_MotifThree', x, trit)
    return outfunc(out)


class SB_MotifThree(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % As for SB_MotifTwo, but using an alphabet of three letters, i.e., looks for
    % motifs in a course-graining of the time series to an alphabet of three letters
    % 
    %---INPUTS:
    % y, time series to analyze
    % trit, the coarse-graining method to use:
    %       (i) 'quantile': equiprobable alphabet by time-series value
    %       (ii) 'diffquant': equiprobably alphabet by time-series increments
    % 
    %---OUTPUTS:
    % Statistics on words of length 1, 2, 3, and 4.
    % 
    %---HISTORY:
    % This code was laboriously written by Ben Fulcher, 2009.
    % 
    ----------------------------------------
    """

    outnames = ('aa',
                'aaa',
                'aaaa',
                'aaab',
                'aaac',
                'aab',
                'aaba',
                'aabb',
                'aabc',
                'aac',
                'aaca',
                'aacb',
                'aacc',
                'ab',
                'aba',
                'abaa',
                'abab',
                'abac',
                'abb',
                'abba',
                'abbb',
                'abbc',
                'abc',
                'abca',
                'abcb',
                'abcc',
                'ac',
                'aca',
                'acaa',
                'acab',
                'acac',
                'acb',
                'acba',
                'acbb',
                'acbc',
                'acc',
                'acca',
                'accb',
                'accc',
                'ba',
                'baa',
                'baaa',
                'baab',
                'baac',
                'bab',
                'baba',
                'babb',
                'babc',
                'bac',
                'baca',
                'bacb',
                'bacc',
                'bb',
                'bba',
                'bbaa',
                'bbab',
                'bbac',
                'bbb',
                'bbba',
                'bbbb',
                'bbbc',
                'bbc',
                'bbca',
                'bbcb',
                'bbcc',
                'bc',
                'bca',
                'bcaa',
                'bcab',
                'bcac',
                'bcb',
                'bcba',
                'bcbb',
                'bcbc',
                'bcc',
                'bcca',
                'bccb',
                'bccc',
                'ca',
                'caa',
                'caaa',
                'caab',
                'caac',
                'cab',
                'caba',
                'cabb',
                'cabc',
                'cac',
                'caca',
                'cacb',
                'cacc',
                'cb',
                'cba',
                'cbaa',
                'cbab',
                'cbac',
                'cbb',
                'cbba',
                'cbbb',
                'cbbc',
                'cbc',
                'cbca',
                'cbcb',
                'cbcc',
                'cc',
                'cca',
                'ccaa',
                'ccab',
                'ccac',
                'ccb',
                'ccba',
                'ccbb',
                'ccbc',
                'ccc',
                'ccca',
                'cccb',
                'cccc',
                'hh',
                'hhh',
                'hhhh')

    def __init__(self, trit='diffquant'):
        super(SB_MotifThree, self).__init__(add_descriptors=False)
        self.trit = trit

    def eval(self, engine, x):
        return HCTSA_SB_MotifThree(engine,
                                   x,
                                   trit=self.trit)


def HCTSA_SB_MotifTwo(eng, x, bint='mean'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Looks at local motifs in a binary symbolization of the time series, which is
    % performed by:
    % 
    %---INPUTS:
    %
    % y, the input time series
    % 
    % bint, the binary transformation method:
    %       (i) 'diff': incremental time-series increases are encoded as 1, and
    %                   decreases as 0,
    %       (ii) 'mean': time-series values above its mean are given 1, and those
    %                    below the mean are 0,
    %       (iii) 'median': time-series values above the median are given 1, and
    %       those below the median 0.
    % 
    %---OUTPUTS:
    % Probabilities of words in the binary alphabet of lengths 1, 2, 3, and 4, and
    % their entropies.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['d',
                                                      'dd',
                                                      'ddd',
                                                      'dddd',
                                                      'dddu',
                                                      'ddu',
                                                      'ddud',
                                                      'dduu',
                                                      'du',
                                                      'dud',
                                                      'dudd',
                                                      'dudu',
                                                      'duu',
                                                      'duud',
                                                      'duuu',
                                                      'h',
                                                      'hh',
                                                      'hhh',
                                                      'hhhh',
                                                      'u',
                                                      'ud',
                                                      'udd',
                                                      'uddd',
                                                      'uddu',
                                                      'udu',
                                                      'udud',
                                                      'uduu',
                                                      'uu',
                                                      'uud',
                                                      'uudd',
                                                      'uudu',
                                                      'uuu',
                                                      'uuud',
                                                      'uuuu']}
    if bint is None:
        out = eng.run_function(1, 'SB_MotifTwo', x, )
    else:
        out = eng.run_function(1, 'SB_MotifTwo', x, bint)
    return outfunc(out)


class SB_MotifTwo(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Looks at local motifs in a binary symbolization of the time series, which is
    % performed by:
    % 
    %---INPUTS:
    %
    % y, the input time series
    % 
    % bint, the binary transformation method:
    %       (i) 'diff': incremental time-series increases are encoded as 1, and
    %                   decreases as 0,
    %       (ii) 'mean': time-series values above its mean are given 1, and those
    %                    below the mean are 0,
    %       (iii) 'median': time-series values above the median are given 1, and
    %       those below the median 0.
    % 
    %---OUTPUTS:
    % Probabilities of words in the binary alphabet of lengths 1, 2, 3, and 4, and
    % their entropies.
    % 
    ----------------------------------------
    """

    outnames = ('d',
                'dd',
                'ddd',
                'dddd',
                'dddu',
                'ddu',
                'ddud',
                'dduu',
                'du',
                'dud',
                'dudd',
                'dudu',
                'duu',
                'duud',
                'duuu',
                'h',
                'hh',
                'hhh',
                'hhhh',
                'u',
                'ud',
                'udd',
                'uddd',
                'uddu',
                'udu',
                'udud',
                'uduu',
                'uu',
                'uud',
                'uudd',
                'uudu',
                'uuu',
                'uuud',
                'uuuu')

    def __init__(self, bint='mean'):
        super(SB_MotifTwo, self).__init__(add_descriptors=False)
        self.bint = bint

    def eval(self, engine, x):
        return HCTSA_SB_MotifTwo(engine,
                                 x,
                                 bint=self.bint)


def HCTSA_SB_TransitionMatrix(eng, x, discmeth='quantile', ng=2, tau=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the transition probabilities between different states of the time
    % series given a method to symbolize or coarse-grain the time series.
    % 
    % The input time series is transformed into a symbolic string using an
    % equiprobable alphabet of ng letters. The transition probabilities are
    % calculated at a lag tau.
    % 
    %---INPUTS:
    % y, the input time series
    %
    % discmeth, the method of discretization (currently 'quantile' is the only
    %           option; could incorporate SB_CoarseGrain for more options in future)
    %
    % ng: number of groups in the course-graining
    %
    % tau: analyze transition matricies corresponding to this lag. We
    %      could either downsample the time series at this lag and then do the
    %      discretization as normal, or do the discretization and then just
    %      look at this dicrete lag. Here we do the former. Can also set tau to 'ac'
    %      to set tau to the first zero-crossing of the autocorrelation function.
    % 
    %---OUTPUTS: include the transition probabilities themselves, as well as the trace
    % of the transition matrix, measures of asymmetry, and eigenvalues of the
    % transition matrix.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['T1',
                                                      'T2',
                                                      'T3',
                                                      'T4',
                                                      'T5',
                                                      'T6',
                                                      'T7',
                                                      'T8',
                                                      'T9',
                                                      'TD1',
                                                      'TD2',
                                                      'TD3',
                                                      'TD4',
                                                      'TD5',
                                                      'maxeig',
                                                      'maxeigcov',
                                                      'maximeig',
                                                      'meaneig',
                                                      'meaneigcov',
                                                      'mineig',
                                                      'mineigcov',
                                                      'ondiag',
                                                      'stddiag',
                                                      'stdeig',
                                                      'stdeigcov',
                                                      'sumdiagcov',
                                                      'symdiff',
                                                      'symsumdiff']}
    if discmeth is None:
        out = eng.run_function(1, 'SB_TransitionMatrix', x, )
    elif ng is None:
        out = eng.run_function(1, 'SB_TransitionMatrix', x, discmeth)
    elif tau is None:
        out = eng.run_function(1, 'SB_TransitionMatrix', x, discmeth, ng)
    else:
        out = eng.run_function(1, 'SB_TransitionMatrix', x, discmeth, ng, tau)
    return outfunc(out)


class SB_TransitionMatrix(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the transition probabilities between different states of the time
    % series given a method to symbolize or coarse-grain the time series.
    % 
    % The input time series is transformed into a symbolic string using an
    % equiprobable alphabet of ng letters. The transition probabilities are
    % calculated at a lag tau.
    % 
    %---INPUTS:
    % y, the input time series
    %
    % discmeth, the method of discretization (currently 'quantile' is the only
    %           option; could incorporate SB_CoarseGrain for more options in future)
    %
    % ng: number of groups in the course-graining
    %
    % tau: analyze transition matricies corresponding to this lag. We
    %      could either downsample the time series at this lag and then do the
    %      discretization as normal, or do the discretization and then just
    %      look at this dicrete lag. Here we do the former. Can also set tau to 'ac'
    %      to set tau to the first zero-crossing of the autocorrelation function.
    % 
    %---OUTPUTS: include the transition probabilities themselves, as well as the trace
    % of the transition matrix, measures of asymmetry, and eigenvalues of the
    % transition matrix.
    % 
    ----------------------------------------
    """

    outnames = ('T1',
                'T2',
                'T3',
                'T4',
                'T5',
                'T6',
                'T7',
                'T8',
                'T9',
                'TD1',
                'TD2',
                'TD3',
                'TD4',
                'TD5',
                'maxeig',
                'maxeigcov',
                'maximeig',
                'meaneig',
                'meaneigcov',
                'mineig',
                'mineigcov',
                'ondiag',
                'stddiag',
                'stdeig',
                'stdeigcov',
                'sumdiagcov',
                'symdiff',
                'symsumdiff')

    def __init__(self, discmeth='quantile', ng=2, tau=1):
        super(SB_TransitionMatrix, self).__init__(add_descriptors=False)
        self.discmeth = discmeth
        self.ng = ng
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_SB_TransitionMatrix(engine,
                                         x,
                                         discmeth=self.discmeth,
                                         ng=self.ng,
                                         tau=self.tau)


def HCTSA_SB_TransitionpAlphabet(eng, x, ng=MatlabSequence('2:40'), tau=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the transition probabilities and measures how they change as the
    % size of the alphabet increases.
    % 
    % Discretization is done by quantile separation.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % ng, the number of groups in the coarse-graining (scalar for constant, or a
    %       vector of ng to compare across this range)
    % 
    % tau: the time-delay; transition matricies corresponding to this time-delay. We
    %      can either downsample the time series at this lag and then do the
    %      discretization as normal, or do the discretization and then just
    %      look at this dicrete lag. Here we do the former. (scalar for
    %      constant tau, vector for range to vary across)
    % 
    %---OUTPUTS: include the decay rate of the sum, mean, and maximum of diagonal
    % elements of the transition matrices, changes in symmetry, and the eigenvalues
    % of the transition matrix.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['maxdiagfexp_a',
                                                      'maxdiagfexp_adjr2',
                                                      'maxdiagfexp_b',
                                                      'maxdiagfexp_r2',
                                                      'maxdiagfexp_rmse',
                                                      'maxeig_fexpa',
                                                      'maxeig_fexpadjr2',
                                                      'maxeig_fexpb',
                                                      'maxeig_fexpr2',
                                                      'maxeig_fexprmse',
                                                      'meandiagfexp_a',
                                                      'meandiagfexp_adjr2',
                                                      'meandiagfexp_b',
                                                      'meandiagfexp_r2',
                                                      'meandiagfexp_rmse',
                                                      'meaneigfexp_a',
                                                      'meaneigfexp_adjr2',
                                                      'meaneigfexp_b',
                                                      'meaneigfexp_r2',
                                                      'meaneigfexp_rmse',
                                                      'mineigfexp_a',
                                                      'mineigfexp_adjr2',
                                                      'mineigfexp_b',
                                                      'mineigfexp_r2',
                                                      'mineigfexp_rmse',
                                                      'stdeigfexp_a',
                                                      'stdeigfexp_adjr2',
                                                      'stdeigfexp_b',
                                                      'stdeigfexp_r2',
                                                      'stdeigfexp_rmse',
                                                      'symd_a',
                                                      'symd_risept',
                                                      'trcov_jump',
                                                      'trcovfexp_a',
                                                      'trcovfexp_adjr2',
                                                      'trcovfexp_b',
                                                      'trcovfexp_r2',
                                                      'trcovfexp_rmse',
                                                      'trfexp_a',
                                                      'trfexp_adjr2',
                                                      'trfexp_b',
                                                      'trfexp_r2',
                                                      'trfexp_rmse',
                                                      'trflin10adjr2',
                                                      'trflin5_adjr2']}
    if ng is None:
        out = eng.run_function(1, 'SB_TransitionpAlphabet', x, )
    elif tau is None:
        out = eng.run_function(1, 'SB_TransitionpAlphabet', x, ng)
    else:
        out = eng.run_function(1, 'SB_TransitionpAlphabet', x, ng, tau)
    return outfunc(out)


class SB_TransitionpAlphabet(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates the transition probabilities and measures how they change as the
    % size of the alphabet increases.
    % 
    % Discretization is done by quantile separation.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % ng, the number of groups in the coarse-graining (scalar for constant, or a
    %       vector of ng to compare across this range)
    % 
    % tau: the time-delay; transition matricies corresponding to this time-delay. We
    %      can either downsample the time series at this lag and then do the
    %      discretization as normal, or do the discretization and then just
    %      look at this dicrete lag. Here we do the former. (scalar for
    %      constant tau, vector for range to vary across)
    % 
    %---OUTPUTS: include the decay rate of the sum, mean, and maximum of diagonal
    % elements of the transition matrices, changes in symmetry, and the eigenvalues
    % of the transition matrix.
    % 
    ----------------------------------------
    """

    outnames = ('maxdiagfexp_a',
                'maxdiagfexp_adjr2',
                'maxdiagfexp_b',
                'maxdiagfexp_r2',
                'maxdiagfexp_rmse',
                'maxeig_fexpa',
                'maxeig_fexpadjr2',
                'maxeig_fexpb',
                'maxeig_fexpr2',
                'maxeig_fexprmse',
                'meandiagfexp_a',
                'meandiagfexp_adjr2',
                'meandiagfexp_b',
                'meandiagfexp_r2',
                'meandiagfexp_rmse',
                'meaneigfexp_a',
                'meaneigfexp_adjr2',
                'meaneigfexp_b',
                'meaneigfexp_r2',
                'meaneigfexp_rmse',
                'mineigfexp_a',
                'mineigfexp_adjr2',
                'mineigfexp_b',
                'mineigfexp_r2',
                'mineigfexp_rmse',
                'stdeigfexp_a',
                'stdeigfexp_adjr2',
                'stdeigfexp_b',
                'stdeigfexp_r2',
                'stdeigfexp_rmse',
                'symd_a',
                'symd_risept',
                'trcov_jump',
                'trcovfexp_a',
                'trcovfexp_adjr2',
                'trcovfexp_b',
                'trcovfexp_r2',
                'trcovfexp_rmse',
                'trfexp_a',
                'trfexp_adjr2',
                'trfexp_b',
                'trfexp_r2',
                'trfexp_rmse',
                'trflin10adjr2',
                'trflin5_adjr2')

    def __init__(self, ng=MatlabSequence('2:40'), tau=1):
        super(SB_TransitionpAlphabet, self).__init__(add_descriptors=False)
        self.ng = ng
        self.tau = tau

    def eval(self, engine, x):
        return HCTSA_SB_TransitionpAlphabet(engine,
                                            x,
                                            ng=self.ng,
                                            tau=self.tau)


def HCTSA_SC_FluctAnal(eng, x, q=2, wtf='dfa', taustep=2, k=2, lag=(), loginc=0):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements fluctuation analysis by a variety of methods.
    % 
    % Much of our implementation is based on the well-explained discussion of
    % scaling methods in:
    % "Power spectrum and detrended fluctuation analysis: Application to daily
    % temperatures" P. Talkner and R. O. Weber, Phys. Rev. E 62(1) 150 (2000)
    % 
    % The main difference between algorithms for estimating scaling exponents amount to
    % differences in how fluctuations, F, are quantified in time-series segments.
    % Many alternatives are implemented in this function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % q, the parameter in the fluctuation function q = 2 (usual) gives RMS fluctuations.
    % 
    % wtf, (what to fluctuate)
    %       (i) 'endptdiff', calculates the differences in end points in each segment
    %       (ii) 'range' calculates the range in each segment
    %       (iii) 'std' takes the standard deviation in each segment
    %       
    %           cf. "Evaluating scaled windowed variance methods for estimating the
    %               Hurst coefficient of time series", M. J. Cannon et al. Physica A
    %               241(3-4) 606 (1997)
    %       
    %       (iv) 'iqr' takes the interquartile range in each segment
    %       (v) 'dfa' removes a polynomial trend of order k in each segment,
    %       (vi) 'rsrange' returns the range after removing a straight line fit
    %       
    %           cf. "Analyzing exact fractal time series: evaluating dispersional
    %           analysis and rescaled range methods",  D. C. Caccia et al., Physica
    %           A 246(3-4) 609 (1997)
    %       
    %       (vii) 'rsrangefit' fits a polynomial of order k and then returns the
    %           range. The parameter q controls the order of fluctuations, for which
    %           we mostly use the standard choice, q = 2, corresponding to root mean
    %           square fluctuations.
    %           An optional input parameter to this operation is a timelag for
    %           computing the cumulative sum (or integrated profile), as suggested
    %           by: "Using detrended fluctuation analysis for lagged correlation
    %           analysis of nonstationary signals" J. Alvarez-Ramirez et al. Phys.
    %           Rev. E 79(5) 057202 (2009)
    % 
    % taustep, increments in tau for linear range (i.e., if loginc = 0), or number of tau
    %           steps in logarithmic range if login = 1
    %           The spacing of time scales, tau, is commonly logarithmic through a range from
    %           5 samples to a quarter of the length of the time series, as suggested in
    %           "Statistical properties of DNA sequences", C.-K. Peng et al. Physica A
    %           221(1-3) 180 (1995)
    %           
    %           Max A. Little's fractal paper used L = 4 to L = N/2:
    %           "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection"
    %           M. A. Little et al. Biomed. Eng. Online 6(1) 23 (2007)
    %           
    % k, polynomial order of detrending for 'dfa', 'rsrangefit'
    % 
    % lag, optional time-lag, as in Alvarez-Ramirez (see (vii) above)
    % 
    % loginc, whether to use logarithmic increments in tau (it should be logarithmic).
    % 
    %---OUTPUTS: include statistics of fitting a linear function to a plot of log(F) as
    % a function of log(tau), and for fitting two straight lines to the same data,
    % choosing the split point at tau = tau_{split} as that which minimizes the
    % combined fitting errors.
    % 
    % This function can also be applied to the absolute deviations of the time
    % series from its mean, and also for just the sign of deviations from the mean
    % (i.e., converting the time series into a series of +1, when the time series is
    % above its mean, and -1 when the time series is below its mean).
    % 
    % All results are obtained with both linearly, and logarithmically-spaced time
    % scales tau.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['alpha',
                                                      'alpharat',
                                                      'linfitint',
                                                      'logtausplit',
                                                      'r1_alpha',
                                                      'r1_linfitint',
                                                      'r1_resac1',
                                                      'r1_se1',
                                                      'r1_se2',
                                                      'r1_ssr',
                                                      'r1_stats_coeffcorr',
                                                      'r2_alpha',
                                                      'r2_linfitint',
                                                      'r2_resac1',
                                                      'r2_se1',
                                                      'r2_se2',
                                                      'r2_ssr',
                                                      'r2_stats_coeffcorr',
                                                      'ratsplitminerr',
                                                      'resac1',
                                                      'se1',
                                                      'se2',
                                                      'ssr',
                                                      'stats_coeffcorr']}
    if q is None:
        out = eng.run_function(1, 'SC_FluctAnal', x, )
    elif wtf is None:
        out = eng.run_function(1, 'SC_FluctAnal', x, q)
    elif taustep is None:
        out = eng.run_function(1, 'SC_FluctAnal', x, q, wtf)
    elif k is None:
        out = eng.run_function(1, 'SC_FluctAnal', x, q, wtf, taustep)
    elif lag is None:
        out = eng.run_function(1, 'SC_FluctAnal', x, q, wtf, taustep, k)
    elif loginc is None:
        out = eng.run_function(1, 'SC_FluctAnal', x, q, wtf, taustep, k, lag)
    else:
        out = eng.run_function(1, 'SC_FluctAnal', x, q, wtf, taustep, k, lag, loginc)
    return outfunc(out)


class SC_FluctAnal(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements fluctuation analysis by a variety of methods.
    % 
    % Much of our implementation is based on the well-explained discussion of
    % scaling methods in:
    % "Power spectrum and detrended fluctuation analysis: Application to daily
    % temperatures" P. Talkner and R. O. Weber, Phys. Rev. E 62(1) 150 (2000)
    % 
    % The main difference between algorithms for estimating scaling exponents amount to
    % differences in how fluctuations, F, are quantified in time-series segments.
    % Many alternatives are implemented in this function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % q, the parameter in the fluctuation function q = 2 (usual) gives RMS fluctuations.
    % 
    % wtf, (what to fluctuate)
    %       (i) 'endptdiff', calculates the differences in end points in each segment
    %       (ii) 'range' calculates the range in each segment
    %       (iii) 'std' takes the standard deviation in each segment
    %       
    %           cf. "Evaluating scaled windowed variance methods for estimating the
    %               Hurst coefficient of time series", M. J. Cannon et al. Physica A
    %               241(3-4) 606 (1997)
    %       
    %       (iv) 'iqr' takes the interquartile range in each segment
    %       (v) 'dfa' removes a polynomial trend of order k in each segment,
    %       (vi) 'rsrange' returns the range after removing a straight line fit
    %       
    %           cf. "Analyzing exact fractal time series: evaluating dispersional
    %           analysis and rescaled range methods",  D. C. Caccia et al., Physica
    %           A 246(3-4) 609 (1997)
    %       
    %       (vii) 'rsrangefit' fits a polynomial of order k and then returns the
    %           range. The parameter q controls the order of fluctuations, for which
    %           we mostly use the standard choice, q = 2, corresponding to root mean
    %           square fluctuations.
    %           An optional input parameter to this operation is a timelag for
    %           computing the cumulative sum (or integrated profile), as suggested
    %           by: "Using detrended fluctuation analysis for lagged correlation
    %           analysis of nonstationary signals" J. Alvarez-Ramirez et al. Phys.
    %           Rev. E 79(5) 057202 (2009)
    % 
    % taustep, increments in tau for linear range (i.e., if loginc = 0), or number of tau
    %           steps in logarithmic range if login = 1
    %           The spacing of time scales, tau, is commonly logarithmic through a range from
    %           5 samples to a quarter of the length of the time series, as suggested in
    %           "Statistical properties of DNA sequences", C.-K. Peng et al. Physica A
    %           221(1-3) 180 (1995)
    %           
    %           Max A. Little's fractal paper used L = 4 to L = N/2:
    %           "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection"
    %           M. A. Little et al. Biomed. Eng. Online 6(1) 23 (2007)
    %           
    % k, polynomial order of detrending for 'dfa', 'rsrangefit'
    % 
    % lag, optional time-lag, as in Alvarez-Ramirez (see (vii) above)
    % 
    % loginc, whether to use logarithmic increments in tau (it should be logarithmic).
    % 
    %---OUTPUTS: include statistics of fitting a linear function to a plot of log(F) as
    % a function of log(tau), and for fitting two straight lines to the same data,
    % choosing the split point at tau = tau_{split} as that which minimizes the
    % combined fitting errors.
    % 
    % This function can also be applied to the absolute deviations of the time
    % series from its mean, and also for just the sign of deviations from the mean
    % (i.e., converting the time series into a series of +1, when the time series is
    % above its mean, and -1 when the time series is below its mean).
    % 
    % All results are obtained with both linearly, and logarithmically-spaced time
    % scales tau.
    % 
    ----------------------------------------
    """

    outnames = ('alpha',
                'alpharat',
                'linfitint',
                'logtausplit',
                'r1_alpha',
                'r1_linfitint',
                'r1_resac1',
                'r1_se1',
                'r1_se2',
                'r1_ssr',
                'r1_stats_coeffcorr',
                'r2_alpha',
                'r2_linfitint',
                'r2_resac1',
                'r2_se1',
                'r2_se2',
                'r2_ssr',
                'r2_stats_coeffcorr',
                'ratsplitminerr',
                'resac1',
                'se1',
                'se2',
                'ssr',
                'stats_coeffcorr')

    def __init__(self, q=2, wtf='dfa', taustep=2, k=2, lag=(), loginc=0):
        super(SC_FluctAnal, self).__init__(add_descriptors=False)
        self.q = q
        self.wtf = wtf
        self.taustep = taustep
        self.k = k
        self.lag = lag
        self.loginc = loginc

    def eval(self, engine, x):
        return HCTSA_SC_FluctAnal(engine,
                                  x,
                                  q=self.q,
                                  wtf=self.wtf,
                                  taustep=self.taustep,
                                  k=self.k,
                                  lag=self.lag,
                                  loginc=self.loginc)


def HCTSA_SC_HurstExponent(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % Calculate the Hurst exponent of the input time series, y
    % 
    % Code by Bill Davidson (quellen@yahoo.com) that estimates the Hurst Exponent of
    % an input time series.
    % 
    % Original code: hurst_exponent.m (renamed: BD_hurst_exponent.m).
    % 
    % Code was obtained from http://www.mathworks.com/matlabcentral/fileexchange/9842
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'SC_HurstExponent', x, )
    return outfunc(out)


class SC_HurstExponent(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % Calculate the Hurst exponent of the input time series, y
    % 
    % Code by Bill Davidson (quellen@yahoo.com) that estimates the Hurst Exponent of
    % an input time series.
    % 
    % Original code: hurst_exponent.m (renamed: BD_hurst_exponent.m).
    % 
    % Code was obtained from http://www.mathworks.com/matlabcentral/fileexchange/9842
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(SC_HurstExponent, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_SC_HurstExponent(engine, x)


def HCTSA_SC_fastdfa(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the scaling exponent of the time series using a fast implementation
    % of detrended fluctuation analysis (DFA).
    % 
    % The original fastdfa code is by Max A. Little and publicly-available at
    % http://www.maxlittle.net/software/index.php
    %
    %---INPUT:
    % y, the input time series, is fed straight into the fastdfa script.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'SC_fastdfa', x, )
    return outfunc(out)


class SC_fastdfa(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the scaling exponent of the time series using a fast implementation
    % of detrended fluctuation analysis (DFA).
    % 
    % The original fastdfa code is by Max A. Little and publicly-available at
    % http://www.maxlittle.net/software/index.php
    %
    %---INPUT:
    % y, the input time series, is fed straight into the fastdfa script.
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(SC_fastdfa, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_SC_fastdfa(engine, x)


def HCTSA_SD_MakeSurrogates(eng, x, surrmethod=None, nsurrs=None, extrap=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Generates surrogate time series given a method (surrogates), number of
    % surrogates (nsurrs), and any extra parameters (extrap)
    % 
    % Method described relatively clearly in Guarin Lopez et al. (arXiv, 2010)
    % Used bits of aaft code that references (and presumably was obtained from)
    % "ìSurrogate data test for nonlinearity including monotonic
    % transformations", D. Kugiumtzis, Phys. Rev. E, vol. 62, no. 1, 2000.
    % 
    % Note that many other surrogate data methods exist that could later be
    % implemented, cf. references in "Improvements to surrogate data methods for
    % nonstationary time series", J. H. Lucio et al., Phys. Rev. E 85, 056202 (2012)
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % surrmethod, the method for generating surrogates:
    %             (i) 'RP' -- random phase surrogates
    %             (ii) 'AAFT' -- amplitude adjusted Fourier transform
    %             (iii) 'TFT' -- truncated Fourier transform
    % 
    % nsurrs, the number of surrogates to generate
    % 
    % extrap, extra parameters required by the selected surrogate generation method
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if surrmethod is None:
        out = eng.run_function(1, 'SD_MakeSurrogates', x, )
    elif nsurrs is None:
        out = eng.run_function(1, 'SD_MakeSurrogates', x, surrmethod)
    elif extrap is None:
        out = eng.run_function(1, 'SD_MakeSurrogates', x, surrmethod, nsurrs)
    else:
        out = eng.run_function(1, 'SD_MakeSurrogates', x, surrmethod, nsurrs, extrap)
    return outfunc(out)


class SD_MakeSurrogates(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Generates surrogate time series given a method (surrogates), number of
    % surrogates (nsurrs), and any extra parameters (extrap)
    % 
    % Method described relatively clearly in Guarin Lopez et al. (arXiv, 2010)
    % Used bits of aaft code that references (and presumably was obtained from)
    % "ìSurrogate data test for nonlinearity including monotonic
    % transformations", D. Kugiumtzis, Phys. Rev. E, vol. 62, no. 1, 2000.
    % 
    % Note that many other surrogate data methods exist that could later be
    % implemented, cf. references in "Improvements to surrogate data methods for
    % nonstationary time series", J. H. Lucio et al., Phys. Rev. E 85, 056202 (2012)
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % surrmethod, the method for generating surrogates:
    %             (i) 'RP' -- random phase surrogates
    %             (ii) 'AAFT' -- amplitude adjusted Fourier transform
    %             (iii) 'TFT' -- truncated Fourier transform
    % 
    % nsurrs, the number of surrogates to generate
    % 
    % extrap, extra parameters required by the selected surrogate generation method
    % 
    ----------------------------------------
    """

    def __init__(self, surrmethod=None, nsurrs=None, extrap=None):
        super(SD_MakeSurrogates, self).__init__(add_descriptors=False)
        self.surrmethod = surrmethod
        self.nsurrs = nsurrs
        self.extrap = extrap

    def eval(self, engine, x):
        return HCTSA_SD_MakeSurrogates(engine,
                                       x,
                                       surrmethod=self.surrmethod,
                                       nsurrs=self.nsurrs,
                                       extrap=self.extrap)


def HCTSA_SD_SurrogateTest(eng, x, surrmeth='RP', nsurrs=99, extrap=(), teststat=('ami1', 'fmmi', 'o3', 'tc3')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes the test statistics obtained from surrogate time series compared to
    % those measured from the given time series.
    % 
    % This code was based on information found in:
    % "Surrogate data test for nonlinearity including nonmonotonic transforms"
    % D. Kugiumtzis Phys. Rev. E 62(1) R25 (2000)
    % 
    % The generation of surrogates is done by the periphery function,
    % SD_MakeSurrogates
    % 
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % surrmeth, the method for generating surrogate time series:
    %       (i) 'RP': random phase surrogates that maintain linear correlations in
    %                 the data but destroy any nonlinear structure through phase
    %                 randomization
    %       (ii) 'AAFT': the amplitude-adjusted Fourier transform method maintains
    %                    linear correlations but destroys nonlinear structure
    %                    through phase randomization, yet preserves the approximate
    %                    amplitude distribution,
    %       (iii) 'TFT': preserves low-frequency phases but randomizes high-frequency phases (as a way of dealing
    %                    with non-stationarity, cf.:
    %               "A new surrogate data method for nonstationary time series",
    %                   D. L. Guarin Lopez et al., arXiv 1008.1804 (2010)
    % 
    % nsurrs, the number of surrogates to compute (default is 99 for a 0.01
    %         significance level 1-sided test)
    % 
    % extrap, extra parameter, the cut-off frequency for 'TFT'
    % 
    % teststat, the test statistic to evalute on all surrogates and the original
    %           time series. Can specify multiple options in a cell and will return
    %           output for each specified test statistic:
    %           (i) 'ami': the automutual information at lag 1, cf.
    %                 "Testing for nonlinearity in irregular fluctuations with
    %                 long-term trends" T. Nakamura and M. Small and Y. Hirata,
    %                 Phys. Rev. E 74(2) 026205 (2006)
    %           (ii) 'fmmi': the first minimum of the automutual information
    %                       function
    %           (iii) 'o3': a third-order statistic used in: "Surrogate time
    %                 series", T. Schreiber and A. Schmitz, Physica D 142(3-4) 346
    %                 (2000)
    %           (iv) 'tc3': a time-reversal asymmetry measure. Outputs of the
    %                 function include a z-test between the two distributions, and
    %                 some comparative rank-based statistics.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ami_f',
                                                      'ami_mediqr',
                                                      'ami_p',
                                                      'ami_prank',
                                                      'ami_zscore',
                                                      'fmmi_f',
                                                      'fmmi_mediqr',
                                                      'fmmi_p',
                                                      'fmmi_prank',
                                                      'fmmi_zscore',
                                                      'o3_f',
                                                      'o3_mediqr',
                                                      'o3_p',
                                                      'o3_prank',
                                                      'o3_zscore',
                                                      'tc3_f',
                                                      'tc3_mediqr',
                                                      'tc3_p',
                                                      'tc3_prank',
                                                      'tc3_zscore']}
    if surrmeth is None:
        out = eng.run_function(1, 'SD_SurrogateTest', x, )
    elif nsurrs is None:
        out = eng.run_function(1, 'SD_SurrogateTest', x, surrmeth)
    elif extrap is None:
        out = eng.run_function(1, 'SD_SurrogateTest', x, surrmeth, nsurrs)
    elif teststat is None:
        out = eng.run_function(1, 'SD_SurrogateTest', x, surrmeth, nsurrs, extrap)
    else:
        out = eng.run_function(1, 'SD_SurrogateTest', x, surrmeth, nsurrs, extrap, teststat)
    return outfunc(out)


class SD_SurrogateTest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes the test statistics obtained from surrogate time series compared to
    % those measured from the given time series.
    % 
    % This code was based on information found in:
    % "Surrogate data test for nonlinearity including nonmonotonic transforms"
    % D. Kugiumtzis Phys. Rev. E 62(1) R25 (2000)
    % 
    % The generation of surrogates is done by the periphery function,
    % SD_MakeSurrogates
    % 
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % surrmeth, the method for generating surrogate time series:
    %       (i) 'RP': random phase surrogates that maintain linear correlations in
    %                 the data but destroy any nonlinear structure through phase
    %                 randomization
    %       (ii) 'AAFT': the amplitude-adjusted Fourier transform method maintains
    %                    linear correlations but destroys nonlinear structure
    %                    through phase randomization, yet preserves the approximate
    %                    amplitude distribution,
    %       (iii) 'TFT': preserves low-frequency phases but randomizes high-frequency phases (as a way of dealing
    %                    with non-stationarity, cf.:
    %               "A new surrogate data method for nonstationary time series",
    %                   D. L. Guarin Lopez et al., arXiv 1008.1804 (2010)
    % 
    % nsurrs, the number of surrogates to compute (default is 99 for a 0.01
    %         significance level 1-sided test)
    % 
    % extrap, extra parameter, the cut-off frequency for 'TFT'
    % 
    % teststat, the test statistic to evalute on all surrogates and the original
    %           time series. Can specify multiple options in a cell and will return
    %           output for each specified test statistic:
    %           (i) 'ami': the automutual information at lag 1, cf.
    %                 "Testing for nonlinearity in irregular fluctuations with
    %                 long-term trends" T. Nakamura and M. Small and Y. Hirata,
    %                 Phys. Rev. E 74(2) 026205 (2006)
    %           (ii) 'fmmi': the first minimum of the automutual information
    %                       function
    %           (iii) 'o3': a third-order statistic used in: "Surrogate time
    %                 series", T. Schreiber and A. Schmitz, Physica D 142(3-4) 346
    %                 (2000)
    %           (iv) 'tc3': a time-reversal asymmetry measure. Outputs of the
    %                 function include a z-test between the two distributions, and
    %                 some comparative rank-based statistics.
    % 
    ----------------------------------------
    """

    outnames = ('ami_f',
                'ami_mediqr',
                'ami_p',
                'ami_prank',
                'ami_zscore',
                'fmmi_f',
                'fmmi_mediqr',
                'fmmi_p',
                'fmmi_prank',
                'fmmi_zscore',
                'o3_f',
                'o3_mediqr',
                'o3_p',
                'o3_prank',
                'o3_zscore',
                'tc3_f',
                'tc3_mediqr',
                'tc3_p',
                'tc3_prank',
                'tc3_zscore')

    def __init__(self, surrmeth='RP', nsurrs=99, extrap=(), teststat=('ami1', 'fmmi', 'o3', 'tc3')):
        super(SD_SurrogateTest, self).__init__(add_descriptors=False)
        self.surrmeth = surrmeth
        self.nsurrs = nsurrs
        self.extrap = extrap
        self.teststat = teststat

    def eval(self, engine, x):
        return HCTSA_SD_SurrogateTest(engine,
                                      x,
                                      surrmeth=self.surrmeth,
                                      nsurrs=self.nsurrs,
                                      extrap=self.extrap,
                                      teststat=self.teststat)


def HCTSA_SD_TSTL_surrogates(eng, x, tau='mi', nsurr=100, surrmethod=1, surrfn='tc3'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Generates surrogate time series and tests them against the original time
    % series according to some test statistics: T_{C3}, using the
    % TSTOOL code tc3 or T_{rev}, using TSTOOL code trev.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % tau, the autocorrelation lag length <x_n x_{n-tau} x_{n-2tau)>/abs(<x_n
    %                                                   x_{n-tau}|^3/2
    % nsurr, the number of surrogates to generate
    % 
    % surrmeth, the method of generating surrogates:
    %               (i) 1: randomizes phases of fourier spectrum
    %               (ii) 2:  (see Theiler algorithm II)
    %               (iii) 3: permutes samples randomly
    % 
    % 
    % surrfn, the surrogate statistic to evaluate on all surrogates, either 'tc3' or
    %           'trev'
    % 
    %---OUTPUTS: include the Gaussianity of the test statistics, a z-test, and
    % various tests based on fitted kernel densities.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['iqrsfrommedian',
                                                      'ksiqrsfrommode',
                                                      'ksphereonmax',
                                                      'kspminfromext',
                                                      'meansurr',
                                                      'normpatponmax',
                                                      'stdfrommean',
                                                      'stdsurr',
                                                      'ztestp']}
    if tau is None:
        out = eng.run_function(1, 'SD_TSTL_surrogates', x, )
    elif nsurr is None:
        out = eng.run_function(1, 'SD_TSTL_surrogates', x, tau)
    elif surrmethod is None:
        out = eng.run_function(1, 'SD_TSTL_surrogates', x, tau, nsurr)
    elif surrfn is None:
        out = eng.run_function(1, 'SD_TSTL_surrogates', x, tau, nsurr, surrmethod)
    else:
        out = eng.run_function(1, 'SD_TSTL_surrogates', x, tau, nsurr, surrmethod, surrfn)
    return outfunc(out)


class SD_TSTL_surrogates(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Generates surrogate time series and tests them against the original time
    % series according to some test statistics: T_{C3}, using the
    % TSTOOL code tc3 or T_{rev}, using TSTOOL code trev.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % tau, the autocorrelation lag length <x_n x_{n-tau} x_{n-2tau)>/abs(<x_n
    %                                                   x_{n-tau}|^3/2
    % nsurr, the number of surrogates to generate
    % 
    % surrmeth, the method of generating surrogates:
    %               (i) 1: randomizes phases of fourier spectrum
    %               (ii) 2:  (see Theiler algorithm II)
    %               (iii) 3: permutes samples randomly
    % 
    % 
    % surrfn, the surrogate statistic to evaluate on all surrogates, either 'tc3' or
    %           'trev'
    % 
    %---OUTPUTS: include the Gaussianity of the test statistics, a z-test, and
    % various tests based on fitted kernel densities.
    % 
    ----------------------------------------
    """

    outnames = ('iqrsfrommedian',
                'ksiqrsfrommode',
                'ksphereonmax',
                'kspminfromext',
                'meansurr',
                'normpatponmax',
                'stdfrommean',
                'stdsurr',
                'ztestp')

    def __init__(self, tau='mi', nsurr=100, surrmethod=1, surrfn='tc3'):
        super(SD_TSTL_surrogates, self).__init__(add_descriptors=False)
        self.tau = tau
        self.nsurr = nsurr
        self.surrmethod = surrmethod
        self.surrfn = surrfn

    def eval(self, engine, x):
        return HCTSA_SD_TSTL_surrogates(engine,
                                        x,
                                        tau=self.tau,
                                        nsurr=self.nsurr,
                                        surrmethod=self.surrmethod,
                                        surrfn=self.surrfn)


def HCTSA_SP_Summaries(eng, x, psdmeth='periodogram', wmeth='hamming', nf=(), dologabs=1, dopower=0):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a set of measures summarizing an estimate of the Fourier transform of
    % the signal.
    % 
    % The estimation can be done using a periodogram, using the periodogram code in
    % Matlab's Signal Processing Toolbox, or a fast fourier transform, implemented
    % using Matlab's fft code.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % psdmeth, the method of obtaining the spectrum from the signal:
    %               (i) 'periodogram': periodogram
    %               (ii) 'fft': fast fourier transform
    % 
    % wmeth, the window to use:
    %               (i) 'boxcar'
    %               (iii) 'bartlett'
    %               (iv) 'hann'
    %               (v) 'hamming'
    %               (vi) 'none'
    %               
    % nf, the number of frequency components to include, if
    %           empty (default), it's approx length(y)
    %       
    % dologabs, if 1, takes log amplitude of the signal before
    %           transforming to the frequency domain.
    % 
    % dopower, analyzes the power spectrum rather than amplitudes of a Fourier
    %          transform
    % 
    %---OUTPUTS: statistics summarizing various properties of the spectrum,
    % including its maximum, minimum, spread, correlation, centroid, area in certain
    % (normalized) frequency bands, moments of the spectrum, Shannon spectral
    % entropy, a spectral flatness measure, power-law fits, and the number of
    % crossings of the spectrum at various amplitude thresholds.
    % 
    %---HISTORY:
    % Ben Fulcher, August 2009
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'ac2',
                                                      'ac3',
                                                      'ac4',
                                                      'area_2_1',
                                                      'area_2_2',
                                                      'area_3_1',
                                                      'area_3_2',
                                                      'area_3_3',
                                                      'area_4_1',
                                                      'area_4_2',
                                                      'area_4_3',
                                                      'area_4_4',
                                                      'area_5_1',
                                                      'area_5_2',
                                                      'area_5_3',
                                                      'area_5_4',
                                                      'area_5_5',
                                                      'areatopeak',
                                                      'centroid',
                                                      'fpoly2_adjr2',
                                                      'fpoly2_r2',
                                                      'fpoly2_rmse',
                                                      'fpoly2_sse',
                                                      'fpoly2csS_p1',
                                                      'fpoly2csS_p2',
                                                      'fpoly2csS_p3',
                                                      'fpolysat_a',
                                                      'fpolysat_adjr2',
                                                      'fpolysat_b',
                                                      'fpolysat_r2',
                                                      'fpolysat_rmse',
                                                      'iqr',
                                                      'linfitloglog_all_a1',
                                                      'linfitloglog_all_a2',
                                                      'linfitloglog_all_s',
                                                      'linfitloglog_all_sea1',
                                                      'linfitloglog_all_sea2',
                                                      'linfitloglog_all_sigrat',
                                                      'linfitloglog_hf_a1',
                                                      'linfitloglog_hf_a2',
                                                      'linfitloglog_hf_s',
                                                      'linfitloglog_hf_sea1',
                                                      'linfitloglog_hf_sea2',
                                                      'linfitloglog_hf_sigrat',
                                                      'linfitloglog_lf_a1',
                                                      'linfitloglog_lf_a2',
                                                      'linfitloglog_lf_s',
                                                      'linfitloglog_lf_sea1',
                                                      'linfitloglog_lf_sea2',
                                                      'linfitloglog_lf_sigrat',
                                                      'linfitloglog_mf_a1',
                                                      'linfitloglog_mf_a2',
                                                      'linfitloglog_mf_s',
                                                      'linfitloglog_mf_sea1',
                                                      'linfitloglog_mf_sea2',
                                                      'linfitloglog_mf_sigrat',
                                                      'logac1',
                                                      'logac2',
                                                      'logac3',
                                                      'logac4',
                                                      'logarea_2_1',
                                                      'logarea_2_2',
                                                      'logarea_3_1',
                                                      'logarea_3_2',
                                                      'logarea_3_3',
                                                      'logarea_4_1',
                                                      'logarea_4_2',
                                                      'logarea_4_3',
                                                      'logarea_4_4',
                                                      'logarea_5_1',
                                                      'logarea_5_2',
                                                      'logarea_5_3',
                                                      'logarea_5_4',
                                                      'logarea_5_5',
                                                      'logiqr',
                                                      'logmaxonlogmean1e',
                                                      'logmean',
                                                      'logstd',
                                                      'logtau',
                                                      'maxS',
                                                      'maxSlog',
                                                      'maxw',
                                                      'maxwidth',
                                                      'mean',
                                                      'meanlog',
                                                      'median',
                                                      'medianlog',
                                                      'melmax',
                                                      'mom3',
                                                      'mom4',
                                                      'mom5',
                                                      'mom6',
                                                      'mom7',
                                                      'mom8',
                                                      'mom9',
                                                      'ncross01',
                                                      'ncross02',
                                                      'ncross05',
                                                      'ncross1',
                                                      'ncross10',
                                                      'ncross2',
                                                      'ncross20',
                                                      'ncross5',
                                                      'q1',
                                                      'q10',
                                                      'q10mel',
                                                      'q1mel',
                                                      'q25',
                                                      'q25log',
                                                      'q25mel',
                                                      'q5',
                                                      'q50',
                                                      'q50mel',
                                                      'q5mel',
                                                      'q75',
                                                      'q75log',
                                                      'q75mel',
                                                      'q90',
                                                      'q90mel',
                                                      'q95',
                                                      'q95mel',
                                                      'q99',
                                                      'q99mel',
                                                      'rawstatav2_m',
                                                      'rawstatav3_m',
                                                      'rawstatav4_m',
                                                      'rawstatav5_m',
                                                      'sfm',
                                                      'spect_shann_ent',
                                                      'spect_shann_ent_norm',
                                                      'statav2_m',
                                                      'statav2_s',
                                                      'statav3_m',
                                                      'statav3_s',
                                                      'statav4_m',
                                                      'statav4_s',
                                                      'statav5_m',
                                                      'statav5_s',
                                                      'std',
                                                      'stdlog',
                                                      'tau',
                                                      'w10_90',
                                                      'w10_90mel',
                                                      'w1_99',
                                                      'w1_99mel',
                                                      'w25_75',
                                                      'w25_75mel',
                                                      'w5_95',
                                                      'w5_95mel',
                                                      'ylogareatopeak']}
    if psdmeth is None:
        out = eng.run_function(1, 'SP_Summaries', x, )
    elif wmeth is None:
        out = eng.run_function(1, 'SP_Summaries', x, psdmeth)
    elif nf is None:
        out = eng.run_function(1, 'SP_Summaries', x, psdmeth, wmeth)
    elif dologabs is None:
        out = eng.run_function(1, 'SP_Summaries', x, psdmeth, wmeth, nf)
    elif dopower is None:
        out = eng.run_function(1, 'SP_Summaries', x, psdmeth, wmeth, nf, dologabs)
    else:
        out = eng.run_function(1, 'SP_Summaries', x, psdmeth, wmeth, nf, dologabs, dopower)
    return outfunc(out)


class SP_Summaries(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a set of measures summarizing an estimate of the Fourier transform of
    % the signal.
    % 
    % The estimation can be done using a periodogram, using the periodogram code in
    % Matlab's Signal Processing Toolbox, or a fast fourier transform, implemented
    % using Matlab's fft code.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % psdmeth, the method of obtaining the spectrum from the signal:
    %               (i) 'periodogram': periodogram
    %               (ii) 'fft': fast fourier transform
    % 
    % wmeth, the window to use:
    %               (i) 'boxcar'
    %               (iii) 'bartlett'
    %               (iv) 'hann'
    %               (v) 'hamming'
    %               (vi) 'none'
    %               
    % nf, the number of frequency components to include, if
    %           empty (default), it's approx length(y)
    %       
    % dologabs, if 1, takes log amplitude of the signal before
    %           transforming to the frequency domain.
    % 
    % dopower, analyzes the power spectrum rather than amplitudes of a Fourier
    %          transform
    % 
    %---OUTPUTS: statistics summarizing various properties of the spectrum,
    % including its maximum, minimum, spread, correlation, centroid, area in certain
    % (normalized) frequency bands, moments of the spectrum, Shannon spectral
    % entropy, a spectral flatness measure, power-law fits, and the number of
    % crossings of the spectrum at various amplitude thresholds.
    % 
    %---HISTORY:
    % Ben Fulcher, August 2009
    ----------------------------------------
    """

    outnames = ('ac1',
                'ac2',
                'ac3',
                'ac4',
                'area_2_1',
                'area_2_2',
                'area_3_1',
                'area_3_2',
                'area_3_3',
                'area_4_1',
                'area_4_2',
                'area_4_3',
                'area_4_4',
                'area_5_1',
                'area_5_2',
                'area_5_3',
                'area_5_4',
                'area_5_5',
                'areatopeak',
                'centroid',
                'fpoly2_adjr2',
                'fpoly2_r2',
                'fpoly2_rmse',
                'fpoly2_sse',
                'fpoly2csS_p1',
                'fpoly2csS_p2',
                'fpoly2csS_p3',
                'fpolysat_a',
                'fpolysat_adjr2',
                'fpolysat_b',
                'fpolysat_r2',
                'fpolysat_rmse',
                'iqr',
                'linfitloglog_all_a1',
                'linfitloglog_all_a2',
                'linfitloglog_all_s',
                'linfitloglog_all_sea1',
                'linfitloglog_all_sea2',
                'linfitloglog_all_sigrat',
                'linfitloglog_hf_a1',
                'linfitloglog_hf_a2',
                'linfitloglog_hf_s',
                'linfitloglog_hf_sea1',
                'linfitloglog_hf_sea2',
                'linfitloglog_hf_sigrat',
                'linfitloglog_lf_a1',
                'linfitloglog_lf_a2',
                'linfitloglog_lf_s',
                'linfitloglog_lf_sea1',
                'linfitloglog_lf_sea2',
                'linfitloglog_lf_sigrat',
                'linfitloglog_mf_a1',
                'linfitloglog_mf_a2',
                'linfitloglog_mf_s',
                'linfitloglog_mf_sea1',
                'linfitloglog_mf_sea2',
                'linfitloglog_mf_sigrat',
                'logac1',
                'logac2',
                'logac3',
                'logac4',
                'logarea_2_1',
                'logarea_2_2',
                'logarea_3_1',
                'logarea_3_2',
                'logarea_3_3',
                'logarea_4_1',
                'logarea_4_2',
                'logarea_4_3',
                'logarea_4_4',
                'logarea_5_1',
                'logarea_5_2',
                'logarea_5_3',
                'logarea_5_4',
                'logarea_5_5',
                'logiqr',
                'logmaxonlogmean1e',
                'logmean',
                'logstd',
                'logtau',
                'maxS',
                'maxSlog',
                'maxw',
                'maxwidth',
                'mean',
                'meanlog',
                'median',
                'medianlog',
                'melmax',
                'mom3',
                'mom4',
                'mom5',
                'mom6',
                'mom7',
                'mom8',
                'mom9',
                'ncross01',
                'ncross02',
                'ncross05',
                'ncross1',
                'ncross10',
                'ncross2',
                'ncross20',
                'ncross5',
                'q1',
                'q10',
                'q10mel',
                'q1mel',
                'q25',
                'q25log',
                'q25mel',
                'q5',
                'q50',
                'q50mel',
                'q5mel',
                'q75',
                'q75log',
                'q75mel',
                'q90',
                'q90mel',
                'q95',
                'q95mel',
                'q99',
                'q99mel',
                'rawstatav2_m',
                'rawstatav3_m',
                'rawstatav4_m',
                'rawstatav5_m',
                'sfm',
                'spect_shann_ent',
                'spect_shann_ent_norm',
                'statav2_m',
                'statav2_s',
                'statav3_m',
                'statav3_s',
                'statav4_m',
                'statav4_s',
                'statav5_m',
                'statav5_s',
                'std',
                'stdlog',
                'tau',
                'w10_90',
                'w10_90mel',
                'w1_99',
                'w1_99mel',
                'w25_75',
                'w25_75mel',
                'w5_95',
                'w5_95mel',
                'ylogareatopeak')

    def __init__(self, psdmeth='periodogram', wmeth='hamming', nf=(), dologabs=1, dopower=0):
        super(SP_Summaries, self).__init__(add_descriptors=False)
        self.psdmeth = psdmeth
        self.wmeth = wmeth
        self.nf = nf
        self.dologabs = dologabs
        self.dopower = dopower

    def eval(self, engine, x):
        return HCTSA_SP_Summaries(engine,
                                  x,
                                  psdmeth=self.psdmeth,
                                  wmeth=self.wmeth,
                                  nf=self.nf,
                                  dologabs=self.dologabs,
                                  dopower=self.dopower)


def HCTSA_ST_FitPolynomial(eng, x, k=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a polynomial of order k to the time series, and returns the mean
    % square error of the fit.
    % 
    % Usually kind of a stupid thing to do with a time series, but it's sometimes
    % somehow informative for time series with large trends.
    % 
    %---INPUTS:
    % y, the input time series.
    % k, the order of the polynomial to fit to y.
    % 
    %---OUTPUT: RMS error of the fit.
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if k is None:
        out = eng.run_function(1, 'ST_FitPolynomial', x, )
    else:
        out = eng.run_function(1, 'ST_FitPolynomial', x, k)
    return outfunc(out)


class ST_FitPolynomial(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Fits a polynomial of order k to the time series, and returns the mean
    % square error of the fit.
    % 
    % Usually kind of a stupid thing to do with a time series, but it's sometimes
    % somehow informative for time series with large trends.
    % 
    %---INPUTS:
    % y, the input time series.
    % k, the order of the polynomial to fit to y.
    % 
    %---OUTPUT: RMS error of the fit.
    %
    ----------------------------------------
    """

    def __init__(self, k=10):
        super(ST_FitPolynomial, self).__init__(add_descriptors=False)
        self.k = k

    def eval(self, engine, x):
        return HCTSA_ST_FitPolynomial(engine,
                                      x,
                                      k=self.k)


def HCTSA_ST_Length(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the length of the input time series vector.
    % 
    %---INPUT:
    % y, the time series vector
    % 
    %---OUTPUT: the length of the time series
    %
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'ST_Length', x, )
    return outfunc(out)


class ST_Length(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures the length of the input time series vector.
    % 
    %---INPUT:
    % y, the time series vector
    % 
    %---OUTPUT: the length of the time series
    %
    ----------------------------------------
    """

    def __init__(self, ):
        super(ST_Length, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_ST_Length(engine, x)


def HCTSA_ST_LocalExtrema(eng, x, lorf='l', n=50):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds maximums and minimums within given segments of the time series and
    % analyses the results. Outputs quantify how local maximums and minimums vary
    % across the time series.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % lorf, whether to use:
    %     (i) 'l', windows of a given length (in which case the third input, n
    %             specifies the length)
    %     (ii) 'n', a specified number of windows to break the time series up into
    %               (in which case the third input, n specifies this number)
    %     (iii) 'tau', sets a window length equal to the correlation length of the
    %                 time series, the first zero-crossing of the autocorrelation
    %                 function.
    %                   
    % n, somehow specifies the window length given the setting of lorf above.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['diffmaxabsmin',
                                                      'maxabsext',
                                                      'maxmaxmed',
                                                      'meanabsext',
                                                      'meanabsmin',
                                                      'meanext',
                                                      'meanmax',
                                                      'meanrat',
                                                      'medianabsext',
                                                      'medianabsmin',
                                                      'medianext',
                                                      'medianmax',
                                                      'medianrat',
                                                      'minabsmin',
                                                      'minmax',
                                                      'minmaxonminabsmin',
                                                      'minminmed',
                                                      'stdext',
                                                      'stdmax',
                                                      'stdmin',
                                                      'uord',
                                                      'zcext']}
    if lorf is None:
        out = eng.run_function(1, 'ST_LocalExtrema', x, )
    elif n is None:
        out = eng.run_function(1, 'ST_LocalExtrema', x, lorf)
    else:
        out = eng.run_function(1, 'ST_LocalExtrema', x, lorf, n)
    return outfunc(out)


class ST_LocalExtrema(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Finds maximums and minimums within given segments of the time series and
    % analyses the results. Outputs quantify how local maximums and minimums vary
    % across the time series.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % lorf, whether to use:
    %     (i) 'l', windows of a given length (in which case the third input, n
    %             specifies the length)
    %     (ii) 'n', a specified number of windows to break the time series up into
    %               (in which case the third input, n specifies this number)
    %     (iii) 'tau', sets a window length equal to the correlation length of the
    %                 time series, the first zero-crossing of the autocorrelation
    %                 function.
    %                   
    % n, somehow specifies the window length given the setting of lorf above.
    % 
    ----------------------------------------
    """

    outnames = ('diffmaxabsmin',
                'maxabsext',
                'maxmaxmed',
                'meanabsext',
                'meanabsmin',
                'meanext',
                'meanmax',
                'meanrat',
                'medianabsext',
                'medianabsmin',
                'medianext',
                'medianmax',
                'medianrat',
                'minabsmin',
                'minmax',
                'minmaxonminabsmin',
                'minminmed',
                'stdext',
                'stdmax',
                'stdmin',
                'uord',
                'zcext')

    def __init__(self, lorf='l', n=50):
        super(ST_LocalExtrema, self).__init__(add_descriptors=False)
        self.lorf = lorf
        self.n = n

    def eval(self, engine, x):
        return HCTSA_ST_LocalExtrema(engine,
                                     x,
                                     lorf=self.lorf,
                                     n=self.n)


def HCTSA_ST_MomentCorr(eng, x, wl=0.02, olap=0.2, mom1='median', mom2='iqr', transf='abs'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates correlations between simple statistics summarizing the distribution
    % of values in local windows of the signal.
    % 
    % Idea to implement this operation was of Nick S. Jones.
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % wl, the sliding window length (can be a fraction to specify a proportion of
    %       the time-series length)
    %       
    % olap, the overlap between consecutive windows as a fraction of the window
    %       length,
    % 
    % mom1, mom2: the statistics to investigate correlations between (in each window):
    %               (i) 'iqr': interquartile range
    %               (ii) 'median': median
    %               (iii) 'std': standard deviation (about the local mean)
    %               (iv) 'mean': mean
    % 
    % transf: the pre-processing transformation to apply to the time series before
    %         analyzing it:
    %               (i) 'abs': takes absolute values of all data points
    %               (ii) 'sqrt': takes the square root of absolute values of all
    %                            data points
    %               (iii) 'sq': takes the square of every data point
    %               (iv) 'none': does no transformation
    %           
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['R',
                                                      'absR',
                                                      'density',
                                                      'mi']}
    if wl is None:
        out = eng.run_function(1, 'ST_MomentCorr', x, )
    elif olap is None:
        out = eng.run_function(1, 'ST_MomentCorr', x, wl)
    elif mom1 is None:
        out = eng.run_function(1, 'ST_MomentCorr', x, wl, olap)
    elif mom2 is None:
        out = eng.run_function(1, 'ST_MomentCorr', x, wl, olap, mom1)
    elif transf is None:
        out = eng.run_function(1, 'ST_MomentCorr', x, wl, olap, mom1, mom2)
    else:
        out = eng.run_function(1, 'ST_MomentCorr', x, wl, olap, mom1, mom2, transf)
    return outfunc(out)


class ST_MomentCorr(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Calculates correlations between simple statistics summarizing the distribution
    % of values in local windows of the signal.
    % 
    % Idea to implement this operation was of Nick S. Jones.
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % wl, the sliding window length (can be a fraction to specify a proportion of
    %       the time-series length)
    %       
    % olap, the overlap between consecutive windows as a fraction of the window
    %       length,
    % 
    % mom1, mom2: the statistics to investigate correlations between (in each window):
    %               (i) 'iqr': interquartile range
    %               (ii) 'median': median
    %               (iii) 'std': standard deviation (about the local mean)
    %               (iv) 'mean': mean
    % 
    % transf: the pre-processing transformation to apply to the time series before
    %         analyzing it:
    %               (i) 'abs': takes absolute values of all data points
    %               (ii) 'sqrt': takes the square root of absolute values of all
    %                            data points
    %               (iii) 'sq': takes the square of every data point
    %               (iv) 'none': does no transformation
    %           
    ----------------------------------------
    """

    outnames = ('R',
                'absR',
                'density',
                'mi')

    def __init__(self, wl=0.02, olap=0.2, mom1='median', mom2='iqr', transf='abs'):
        super(ST_MomentCorr, self).__init__(add_descriptors=False)
        self.wl = wl
        self.olap = olap
        self.mom1 = mom1
        self.mom2 = mom2
        self.transf = transf

    def eval(self, engine, x):
        return HCTSA_ST_MomentCorr(engine,
                                   x,
                                   wl=self.wl,
                                   olap=self.olap,
                                   mom1=self.mom1,
                                   mom2=self.mom2,
                                   transf=self.transf)


def HCTSA_ST_SimpleStats(eng, x, whatstat='pmcross'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a basic statistic about the input time series, depending on the input whatstat
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % whatstat, the statistic to return:
    %          (i) 'zcross': the proportionof zero-crossings of the time series
    %                        (z-scored input thus returns mean-crossings),
    %          (ii) 'maxima': the proportion of the time series that is a local maximum
    %          (iii) 'minima': the proportion of the time series that is a local minimum
    %          (iv) 'pmcross': the ratio of the number of times that the (ideally
    %                          z-scored) time-series crosses +1 (i.e., 1 standard
    %                          deviation above the mean) to the number of times
    %                          that it crosses -1 (i.e., 1 standard deviation below
    %                          the mean).
    %          (v) 'zsczcross': the ratio of zero crossings of raw to detrended
    %                           time series where the raw has zero mean.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if whatstat is None:
        out = eng.run_function(1, 'ST_SimpleStats', x, )
    else:
        out = eng.run_function(1, 'ST_SimpleStats', x, whatstat)
    return outfunc(out)


class ST_SimpleStats(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Returns a basic statistic about the input time series, depending on the input whatstat
    % 
    %---INPUTS:
    % x, the input time series
    % 
    % whatstat, the statistic to return:
    %          (i) 'zcross': the proportionof zero-crossings of the time series
    %                        (z-scored input thus returns mean-crossings),
    %          (ii) 'maxima': the proportion of the time series that is a local maximum
    %          (iii) 'minima': the proportion of the time series that is a local minimum
    %          (iv) 'pmcross': the ratio of the number of times that the (ideally
    %                          z-scored) time-series crosses +1 (i.e., 1 standard
    %                          deviation above the mean) to the number of times
    %                          that it crosses -1 (i.e., 1 standard deviation below
    %                          the mean).
    %          (v) 'zsczcross': the ratio of zero crossings of raw to detrended
    %                           time series where the raw has zero mean.
    % 
    ----------------------------------------
    """

    def __init__(self, whatstat='pmcross'):
        super(ST_SimpleStats, self).__init__(add_descriptors=False)
        self.whatstat = whatstat

    def eval(self, engine, x):
        return HCTSA_ST_SimpleStats(engine,
                                    x,
                                    whatstat=self.whatstat)


def HCTSA_SY_DriftingMean(eng, x, howl='num', l=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This function implements an idea found in the Matlab Central forum:
    % http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    % 
    % >> It seems to me that you are looking for a measure for a drifting mean.
    % >> If so, this is what I would try:
    % >> 
    % >> - Decide on a frame length N
    % >> - Split your signal in a number of frames of length N
    % >> - Compute the means of each frame
    % >> - Compute the variance for each frame
    % >> - Compare the ratio of maximum and minimum mean
    % >>   with the mean variance of the frames.
    % >> 
    % >> Rune
    % 
    % This operation splits the time series into segments, computes the mean and
    % variance in each segment and compares the maximum and minimum mean to the mean
    % variance.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % howl, (i) 'fix': fixed-length segments (of length l)
    %       (ii) 'num': a given number, l, of segments
    %       
    % l, either the length ('fix') or number of segments ('num')
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['max',
                                                      'mean',
                                                      'meanabsmaxmin',
                                                      'meanmaxmin',
                                                      'min']}
    if howl is None:
        out = eng.run_function(1, 'SY_DriftingMean', x, )
    elif l is None:
        out = eng.run_function(1, 'SY_DriftingMean', x, howl)
    else:
        out = eng.run_function(1, 'SY_DriftingMean', x, howl, l)
    return outfunc(out)


class SY_DriftingMean(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This function implements an idea found in the Matlab Central forum:
    % http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    % 
    % >> It seems to me that you are looking for a measure for a drifting mean.
    % >> If so, this is what I would try:
    % >> 
    % >> - Decide on a frame length N
    % >> - Split your signal in a number of frames of length N
    % >> - Compute the means of each frame
    % >> - Compute the variance for each frame
    % >> - Compare the ratio of maximum and minimum mean
    % >>   with the mean variance of the frames.
    % >> 
    % >> Rune
    % 
    % This operation splits the time series into segments, computes the mean and
    % variance in each segment and compares the maximum and minimum mean to the mean
    % variance.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % howl, (i) 'fix': fixed-length segments (of length l)
    %       (ii) 'num': a given number, l, of segments
    %       
    % l, either the length ('fix') or number of segments ('num')
    % 
    ----------------------------------------
    """

    outnames = ('max',
                'mean',
                'meanabsmaxmin',
                'meanmaxmin',
                'min')

    def __init__(self, howl='num', l=10):
        super(SY_DriftingMean, self).__init__(add_descriptors=False)
        self.howl = howl
        self.l = l

    def eval(self, engine, x):
        return HCTSA_SY_DriftingMean(engine,
                                     x,
                                     howl=self.howl,
                                     l=self.l)


def HCTSA_SY_DynWin(eng, x, maxnseg=10):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes how stationarity estimates depend on the number of segments used to
    % segment up the time series.
    % 
    % Specifically, variation in a range of local measures are implemented: mean,
    % standard deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1),
    % AC(2), and the first zero-crossing of the autocorrelation function.
    % 
    % The standard deviation of local estimates of these quantities across the time
    % series are calculated as an estimate of the stationarity in this quantity as a
    % function of the number of splits, n_{seg}, of the time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % maxnseg, the maximum number of segments to consider. Will sweep from 2
    %           segments to maxnseg.
    % 
    % 
    %---OUTPUTS: the standard deviation of this set of 'stationarity' estimates
    % across these window sizes.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['stdac1',
                                                      'stdac2',
                                                      'stdactaug',
                                                      'stdactaul',
                                                      'stdapen1_02',
                                                      'stdkurt',
                                                      'stdmean',
                                                      'stdsampen1_02',
                                                      'stdskew',
                                                      'stdstd',
                                                      'stdtaul']}
    if maxnseg is None:
        out = eng.run_function(1, 'SY_DynWin', x, )
    else:
        out = eng.run_function(1, 'SY_DynWin', x, maxnseg)
    return outfunc(out)


class SY_DynWin(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Analyzes how stationarity estimates depend on the number of segments used to
    % segment up the time series.
    % 
    % Specifically, variation in a range of local measures are implemented: mean,
    % standard deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1),
    % AC(2), and the first zero-crossing of the autocorrelation function.
    % 
    % The standard deviation of local estimates of these quantities across the time
    % series are calculated as an estimate of the stationarity in this quantity as a
    % function of the number of splits, n_{seg}, of the time series.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % maxnseg, the maximum number of segments to consider. Will sweep from 2
    %           segments to maxnseg.
    % 
    % 
    %---OUTPUTS: the standard deviation of this set of 'stationarity' estimates
    % across these window sizes.
    % 
    ----------------------------------------
    """

    outnames = ('stdac1',
                'stdac2',
                'stdactaug',
                'stdactaul',
                'stdapen1_02',
                'stdkurt',
                'stdmean',
                'stdsampen1_02',
                'stdskew',
                'stdstd',
                'stdtaul')

    def __init__(self, maxnseg=10):
        super(SY_DynWin, self).__init__(add_descriptors=False)
        self.maxnseg = maxnseg

    def eval(self, engine, x):
        return HCTSA_SY_DynWin(engine,
                               x,
                               maxnseg=self.maxnseg)


def HCTSA_SY_KPSStest(eng, x, lags=MatlabSequence('0:10')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Performs the KPSS stationarity test, of Kwiatkowski, Phillips, Schmidt, and Shin,
    % "Testing the null hypothesis of stationarity against the alternative of a
    % unit root: How sure are we that economic time series have a unit root?"
    % Kwiatkowski, Denis and Phillips, Peter C. B. and Schmidt, Peter and Shin, Yongcheol
    % J. Econometrics, 54(1-3) 159 (2002)
    % 
    % Uses the function kpsstest from Matlab's Econometrics Toolbox. The null
    % hypothesis is that a univariate time series is trend stationary, the
    % alternative hypothesis is that it is a non-stationary unit-root process.
    % 
    % The code can implemented for a specific time lag, tau. Alternatively, measures
    % of change in p-values and test statistics will be outputted if the input is a
    % vector of time lags.
    % 
    %---INPUTS:
    % y, the input time series
    % lags, can be either a scalar (returns basic test statistic and p-value), or
    %                   vector (returns statistics on changes across these time lags)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['lagmaxstat',
                                                      'lagminstat',
                                                      'maxpValue',
                                                      'maxstat',
                                                      'minpValue',
                                                      'minstat',
                                                      'pValue',
                                                      'stat']}
    if lags is None:
        out = eng.run_function(1, 'SY_KPSStest', x, )
    else:
        out = eng.run_function(1, 'SY_KPSStest', x, lags)
    return outfunc(out)


class SY_KPSStest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Performs the KPSS stationarity test, of Kwiatkowski, Phillips, Schmidt, and Shin,
    % "Testing the null hypothesis of stationarity against the alternative of a
    % unit root: How sure are we that economic time series have a unit root?"
    % Kwiatkowski, Denis and Phillips, Peter C. B. and Schmidt, Peter and Shin, Yongcheol
    % J. Econometrics, 54(1-3) 159 (2002)
    % 
    % Uses the function kpsstest from Matlab's Econometrics Toolbox. The null
    % hypothesis is that a univariate time series is trend stationary, the
    % alternative hypothesis is that it is a non-stationary unit-root process.
    % 
    % The code can implemented for a specific time lag, tau. Alternatively, measures
    % of change in p-values and test statistics will be outputted if the input is a
    % vector of time lags.
    % 
    %---INPUTS:
    % y, the input time series
    % lags, can be either a scalar (returns basic test statistic and p-value), or
    %                   vector (returns statistics on changes across these time lags)
    % 
    ----------------------------------------
    """

    outnames = ('lagmaxstat',
                'lagminstat',
                'maxpValue',
                'maxstat',
                'minpValue',
                'minstat',
                'pValue',
                'stat')

    def __init__(self, lags=MatlabSequence('0:10')):
        super(SY_KPSStest, self).__init__(add_descriptors=False)
        self.lags = lags

    def eval(self, engine, x):
        return HCTSA_SY_KPSStest(engine,
                                 x,
                                 lags=self.lags)


def HCTSA_SY_LinearTrend(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Linearly detrends the time series using the Matlab algorithm detrend,
    % and returns the ratio of standard deviations before and after the linear
    % detrending.
    % 
    % If a strong linear trend is present in the time series, this  operation should
    % output a low value.
    % 
    %---INPUT:
    % x, the input time series
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    out = eng.run_function(1, 'SY_LinearTrend', x, )
    return outfunc(out)


class SY_LinearTrend(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Linearly detrends the time series using the Matlab algorithm detrend,
    % and returns the ratio of standard deviations before and after the linear
    % detrending.
    % 
    % If a strong linear trend is present in the time series, this  operation should
    % output a low value.
    % 
    %---INPUT:
    % x, the input time series
    % 
    ----------------------------------------
    """

    def __init__(self, ):
        super(SY_LinearTrend, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_SY_LinearTrend(engine, x)


def HCTSA_SY_LocalDistributions(eng, x, nseg=4, eachorpar='each', npoints=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares the distribution in consecutive partitions of the signal,
    % returning the sum of differences between each kernel-smoothed distributions
    % (using the Matlab function ksdensity).
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % nseg, the number of segments to break the time series into
    % 
    % eachorpar, (i) 'par': compares each local distribution to the parent (full time
    %                       series) distribution
    %            (ii) 'each': compare each local distribution to all other local
    %                         distributions
    % 
    % npoints, number of points to compute the distribution across (in each local
    %          segments)
    % 
    % The operation behaves in one of two modes: each compares the distribution in
    % each segment to that in every other segment, and par compares each
    % distribution to the so-called 'parent' distribution, that of the full signal.
    % 
    %---OUTPUTS: measures of the sum of absolute deviations between distributions
    % across the different pairwise comparisons.
    % 
    %---HISTORY:
    % Ben Fulcher, August 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['None',
                                                      'meandiv',
                                                      'mediandiv',
                                                      'mindiv',
                                                      'stddiv']}
    if nseg is None:
        out = eng.run_function(1, 'SY_LocalDistributions', x, )
    elif eachorpar is None:
        out = eng.run_function(1, 'SY_LocalDistributions', x, nseg)
    elif npoints is None:
        out = eng.run_function(1, 'SY_LocalDistributions', x, nseg, eachorpar)
    else:
        out = eng.run_function(1, 'SY_LocalDistributions', x, nseg, eachorpar, npoints)
    return outfunc(out)


class SY_LocalDistributions(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares the distribution in consecutive partitions of the signal,
    % returning the sum of differences between each kernel-smoothed distributions
    % (using the Matlab function ksdensity).
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % nseg, the number of segments to break the time series into
    % 
    % eachorpar, (i) 'par': compares each local distribution to the parent (full time
    %                       series) distribution
    %            (ii) 'each': compare each local distribution to all other local
    %                         distributions
    % 
    % npoints, number of points to compute the distribution across (in each local
    %          segments)
    % 
    % The operation behaves in one of two modes: each compares the distribution in
    % each segment to that in every other segment, and par compares each
    % distribution to the so-called 'parent' distribution, that of the full signal.
    % 
    %---OUTPUTS: measures of the sum of absolute deviations between distributions
    % across the different pairwise comparisons.
    % 
    %---HISTORY:
    % Ben Fulcher, August 2009
    % 
    ----------------------------------------
    """

    outnames = ('None',
                'meandiv',
                'mediandiv',
                'mindiv',
                'stddiv')

    def __init__(self, nseg=4, eachorpar='each', npoints=None):
        super(SY_LocalDistributions, self).__init__(add_descriptors=False)
        self.nseg = nseg
        self.eachorpar = eachorpar
        self.npoints = npoints

    def eval(self, engine, x):
        return HCTSA_SY_LocalDistributions(engine,
                                           x,
                                           nseg=self.nseg,
                                           eachorpar=self.eachorpar,
                                           npoints=self.npoints)


def HCTSA_SY_LocalGlobal(eng, x, lorp='l', n=50):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares statistics measured in a local region of the time series to that
    % measured of the full time series.
    % 
    % 
    %---INPUTS:
    % y, the time series to analyze
    % 
    % lorp, the local subset of time series to study:
    %             (i) 'l': the first n points in a time series,
    %             (ii) 'p': an initial proportion of the full time series, n
    %             (iii) 'unicg': n evenly-spaced points throughout the time series
    %             (iv) 'randcg': n randomly-chosen points from the time series (chosen with replacement)
    % 
    % n, the parameter for the method specified above
    % 
    % 
    %---OUTPUTS: the mean, standard deviation, median, interquartile range,
    % skewness, kurtosis, AC(1), and SampEn(1,0.1).
    % 
    % This is not the most reliable or systematic operation because only a single
    % sample is taken from the time series and compared to the full time series.
    % A better approach would be to repeat over many local subsets and compare the
    % statistics of these local regions to the full time series.
    % 
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1',
                                                      'iqr',
                                                      'kurtosis',
                                                      'mean',
                                                      'median',
                                                      'skewness',
                                                      'std']}
    if lorp is None:
        out = eng.run_function(1, 'SY_LocalGlobal', x, )
    elif n is None:
        out = eng.run_function(1, 'SY_LocalGlobal', x, lorp)
    else:
        out = eng.run_function(1, 'SY_LocalGlobal', x, lorp, n)
    return outfunc(out)


class SY_LocalGlobal(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares statistics measured in a local region of the time series to that
    % measured of the full time series.
    % 
    % 
    %---INPUTS:
    % y, the time series to analyze
    % 
    % lorp, the local subset of time series to study:
    %             (i) 'l': the first n points in a time series,
    %             (ii) 'p': an initial proportion of the full time series, n
    %             (iii) 'unicg': n evenly-spaced points throughout the time series
    %             (iv) 'randcg': n randomly-chosen points from the time series (chosen with replacement)
    % 
    % n, the parameter for the method specified above
    % 
    % 
    %---OUTPUTS: the mean, standard deviation, median, interquartile range,
    % skewness, kurtosis, AC(1), and SampEn(1,0.1).
    % 
    % This is not the most reliable or systematic operation because only a single
    % sample is taken from the time series and compared to the full time series.
    % A better approach would be to repeat over many local subsets and compare the
    % statistics of these local regions to the full time series.
    % 
    % 
    %---HISTORY:
    % Ben Fulcher, September 2009
    % 
    ----------------------------------------
    """

    outnames = ('ac1',
                'iqr',
                'kurtosis',
                'mean',
                'median',
                'skewness',
                'std')

    def __init__(self, lorp='l', n=50):
        super(SY_LocalGlobal, self).__init__(add_descriptors=False)
        self.lorp = lorp
        self.n = n

    def eval(self, engine, x):
        return HCTSA_SY_LocalGlobal(engine,
                                    x,
                                    lorp=self.lorp,
                                    n=self.n)


def HCTSA_SY_PPtest(eng, x, lags=MatlabSequence('0:5'), model='ts', teststat='t1'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Performs the Phillips-Peron unit root test for a time series via the code
    % pptest from Matlab's Econometrics Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % lags, a vector of lags
    % 
    % model, a specified model:
    %               'ar': autoregressive
    %               'ard': autoregressive with drift, or
    %               'ts': trend stationary,
    %               (see Matlab documentation for information)
    %               
    % teststat, the test statistic:
    %               't1': the standard t-statistic, or
    %               't2' a lag-adjusted, 'unStudentized' t statistic.
    %               (see Matlab documentation for information)
    %               
    %---OUTPUTS: statistics on the p-values and lags obtained from the set of tests, as
    % well as measures of the regression statistics.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['lagmaxp',
                                                      'lagminp',
                                                      'maxpValue',
                                                      'maxrmse',
                                                      'maxstat',
                                                      'meanloglikelihood',
                                                      'meanpValue',
                                                      'meanstat',
                                                      'minAIC',
                                                      'minBIC',
                                                      'minHQC',
                                                      'minpValue',
                                                      'minrmse',
                                                      'minstat',
                                                      'stdpValue']}
    if lags is None:
        out = eng.run_function(1, 'SY_PPtest', x, )
    elif model is None:
        out = eng.run_function(1, 'SY_PPtest', x, lags)
    elif teststat is None:
        out = eng.run_function(1, 'SY_PPtest', x, lags, model)
    else:
        out = eng.run_function(1, 'SY_PPtest', x, lags, model, teststat)
    return outfunc(out)


class SY_PPtest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Performs the Phillips-Peron unit root test for a time series via the code
    % pptest from Matlab's Econometrics Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % lags, a vector of lags
    % 
    % model, a specified model:
    %               'ar': autoregressive
    %               'ard': autoregressive with drift, or
    %               'ts': trend stationary,
    %               (see Matlab documentation for information)
    %               
    % teststat, the test statistic:
    %               't1': the standard t-statistic, or
    %               't2' a lag-adjusted, 'unStudentized' t statistic.
    %               (see Matlab documentation for information)
    %               
    %---OUTPUTS: statistics on the p-values and lags obtained from the set of tests, as
    % well as measures of the regression statistics.
    % 
    ----------------------------------------
    """

    outnames = ('lagmaxp',
                'lagminp',
                'maxpValue',
                'maxrmse',
                'maxstat',
                'meanloglikelihood',
                'meanpValue',
                'meanstat',
                'minAIC',
                'minBIC',
                'minHQC',
                'minpValue',
                'minrmse',
                'minstat',
                'stdpValue')

    def __init__(self, lags=MatlabSequence('0:5'), model='ts', teststat='t1'):
        super(SY_PPtest, self).__init__(add_descriptors=False)
        self.lags = lags
        self.model = model
        self.teststat = teststat

    def eval(self, engine, x):
        return HCTSA_SY_PPtest(engine,
                               x,
                               lags=self.lags,
                               model=self.model,
                               teststat=self.teststat)


def HCTSA_SY_RangeEvolve(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures of the range of the time series as a function of time,
    % i.e., range(x_{1:i}) for i = 1, 2, ..., N, where N is the length of the time
    % series.
    % 
    %---INPUT:
    % y, the time series
    % 
    %---OUTPUTS: based on the dynamics of how new extreme events occur with time.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['l10',
                                                      'l100',
                                                      'l1000',
                                                      'l50',
                                                      'nuql10',
                                                      'nuql100',
                                                      'nuql1000',
                                                      'nuql50',
                                                      'nuqp1',
                                                      'nuqp10',
                                                      'nuqp20',
                                                      'nuqp50',
                                                      'p1',
                                                      'p10',
                                                      'p20',
                                                      'p50',
                                                      'totnuq']}
    out = eng.run_function(1, 'SY_RangeEvolve', x, )
    return outfunc(out)


class SY_RangeEvolve(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Measures of the range of the time series as a function of time,
    % i.e., range(x_{1:i}) for i = 1, 2, ..., N, where N is the length of the time
    % series.
    % 
    %---INPUT:
    % y, the time series
    % 
    %---OUTPUTS: based on the dynamics of how new extreme events occur with time.
    % 
    ----------------------------------------
    """

    outnames = ('l10',
                'l100',
                'l1000',
                'l50',
                'nuql10',
                'nuql100',
                'nuql1000',
                'nuql50',
                'nuqp1',
                'nuqp10',
                'nuqp20',
                'nuqp50',
                'p1',
                'p10',
                'p20',
                'p50',
                'totnuq')

    def __init__(self, ):
        super(SY_RangeEvolve, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_SY_RangeEvolve(engine, x)


def HCTSA_SY_SlidingWindow(eng, x, windowstat='ent', acrosswindowstat='ent', nseg=5, nmov=2):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This function is based on sliding a window along the time series, measuring
    % some quantity in each window, and outputting some summary of this set of local
    % estimates of that quantity.
    % 
    % Another way of saying it: calculate 'windowstat' in each window, and computes
    % 'acrosswindowstat' for the set of statistics calculated in each window.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % windowstat, the measure to calculate in each window:
    %               (i) 'mean', mean
    %               (ii) 'std', standard deviation
    %               (iii) 'ent', distribution entropy
    %               (iv) 'mom3', skewness
    %               (v) 'mom4', kurtosis
    %               (vi) 'mom5', the fifth moment of the distribution
    %               (vii) 'lillie', the p-value for a Lilliefors Gaussianity test
    %               (viii) 'AC1', the lag-1 autocorrelation
    %               (ix) 'apen', Approximate Entropy
    % 
    % acrosswindowstat, controls how the obtained sequence of local estimates is
    %                   compared (as a ratio to the full time series):
    %                       (i) 'std': standard deviation
    %                       (ii) 'ent' histogram entropy
    %                       (iii) 'apen': Approximate Entropy, ApEn(1,0.2)
    %                               cf. "Approximate entropy as a measure of system
    %                               complexity", S. M. Pincus, P. Natl. Acad. Sci.
    %                               USA 88(6) 2297 (1991)
    % 
    % nseg, the number of segments to divide the time series up into, thus
    %       controlling the window length
    % 
    % nmov, the increment to move the window at each iteration, as 1/fraction of the
    %       window length (e.g., nmov = 2, means the window moves half the length of the
    %       window at each increment)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if windowstat is None:
        out = eng.run_function(1, 'SY_SlidingWindow', x, )
    elif acrosswindowstat is None:
        out = eng.run_function(1, 'SY_SlidingWindow', x, windowstat)
    elif nseg is None:
        out = eng.run_function(1, 'SY_SlidingWindow', x, windowstat, acrosswindowstat)
    elif nmov is None:
        out = eng.run_function(1, 'SY_SlidingWindow', x, windowstat, acrosswindowstat, nseg)
    else:
        out = eng.run_function(1, 'SY_SlidingWindow', x, windowstat, acrosswindowstat, nseg, nmov)
    return outfunc(out)


class SY_SlidingWindow(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This function is based on sliding a window along the time series, measuring
    % some quantity in each window, and outputting some summary of this set of local
    % estimates of that quantity.
    % 
    % Another way of saying it: calculate 'windowstat' in each window, and computes
    % 'acrosswindowstat' for the set of statistics calculated in each window.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % windowstat, the measure to calculate in each window:
    %               (i) 'mean', mean
    %               (ii) 'std', standard deviation
    %               (iii) 'ent', distribution entropy
    %               (iv) 'mom3', skewness
    %               (v) 'mom4', kurtosis
    %               (vi) 'mom5', the fifth moment of the distribution
    %               (vii) 'lillie', the p-value for a Lilliefors Gaussianity test
    %               (viii) 'AC1', the lag-1 autocorrelation
    %               (ix) 'apen', Approximate Entropy
    % 
    % acrosswindowstat, controls how the obtained sequence of local estimates is
    %                   compared (as a ratio to the full time series):
    %                       (i) 'std': standard deviation
    %                       (ii) 'ent' histogram entropy
    %                       (iii) 'apen': Approximate Entropy, ApEn(1,0.2)
    %                               cf. "Approximate entropy as a measure of system
    %                               complexity", S. M. Pincus, P. Natl. Acad. Sci.
    %                               USA 88(6) 2297 (1991)
    % 
    % nseg, the number of segments to divide the time series up into, thus
    %       controlling the window length
    % 
    % nmov, the increment to move the window at each iteration, as 1/fraction of the
    %       window length (e.g., nmov = 2, means the window moves half the length of the
    %       window at each increment)
    % 
    ----------------------------------------
    """

    def __init__(self, windowstat='ent', acrosswindowstat='ent', nseg=5, nmov=2):
        super(SY_SlidingWindow, self).__init__(add_descriptors=False)
        self.windowstat = windowstat
        self.acrosswindowstat = acrosswindowstat
        self.nseg = nseg
        self.nmov = nmov

    def eval(self, engine, x):
        return HCTSA_SY_SlidingWindow(engine,
                                      x,
                                      windowstat=self.windowstat,
                                      acrosswindowstat=self.acrosswindowstat,
                                      nseg=self.nseg,
                                      nmov=self.nmov)


def HCTSA_SY_SpreadRandomLocal(eng, x, l='ac5', nseg=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements a bootstrap-based stationarity measure: nseg time-series segments
    % of length l are selected at random from the time series and in each
    % segment a local quantity is calculated: mean, standard deviation, skewness,
    % kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1), AC(2), and the first
    % zero-crossing of the autocorrelation function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % l, the length of local time-series segments to analyze as a positive integer.
    %    Can also be a specified character string:
    %       (i) 'ac2': twice the first zero-crossing of the autocorrelation function
    %       (ii) 'ac5': five times the first zero-crossing of the autocorrelation function
    % 
    % nseg, the number of randomly-selected local segments to analyze
    % 
    %---OUTPUTS: the mean and also the standard deviation of this set of 100 local
    % estimates.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['meanac1',
                                                      'meanac2',
                                                      'meanapen1_02',
                                                      'meankurt',
                                                      'meanmean',
                                                      'meansampen1_02',
                                                      'meanskew',
                                                      'meanstd',
                                                      'meantaul',
                                                      'stdac1',
                                                      'stdac2',
                                                      'stdapen1_02',
                                                      'stdkurt',
                                                      'stdmean',
                                                      'stdsampen1_02',
                                                      'stdskew',
                                                      'stdstd',
                                                      'stdtaul']}
    if l is None:
        out = eng.run_function(1, 'SY_SpreadRandomLocal', x, )
    elif nseg is None:
        out = eng.run_function(1, 'SY_SpreadRandomLocal', x, l)
    else:
        out = eng.run_function(1, 'SY_SpreadRandomLocal', x, l, nseg)
    return outfunc(out)


class SY_SpreadRandomLocal(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Implements a bootstrap-based stationarity measure: nseg time-series segments
    % of length l are selected at random from the time series and in each
    % segment a local quantity is calculated: mean, standard deviation, skewness,
    % kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1), AC(2), and the first
    % zero-crossing of the autocorrelation function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % l, the length of local time-series segments to analyze as a positive integer.
    %    Can also be a specified character string:
    %       (i) 'ac2': twice the first zero-crossing of the autocorrelation function
    %       (ii) 'ac5': five times the first zero-crossing of the autocorrelation function
    % 
    % nseg, the number of randomly-selected local segments to analyze
    % 
    %---OUTPUTS: the mean and also the standard deviation of this set of 100 local
    % estimates.
    % 
    ----------------------------------------
    """

    outnames = ('meanac1',
                'meanac2',
                'meanapen1_02',
                'meankurt',
                'meanmean',
                'meansampen1_02',
                'meanskew',
                'meanstd',
                'meantaul',
                'stdac1',
                'stdac2',
                'stdapen1_02',
                'stdkurt',
                'stdmean',
                'stdsampen1_02',
                'stdskew',
                'stdstd',
                'stdtaul')

    def __init__(self, l='ac5', nseg=None):
        super(SY_SpreadRandomLocal, self).__init__(add_descriptors=False)
        self.l = l
        self.nseg = nseg

    def eval(self, engine, x):
        return HCTSA_SY_SpreadRandomLocal(engine,
                                          x,
                                          l=self.l,
                                          nseg=self.nseg)


def HCTSA_SY_StatAv(eng, x, whattype='len', n=100):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % The StatAv measure is a simple mean-stationarity metric that divides
    % the time series into non-overlapping subsegments, calculates the mean in each
    % of these segments and returns the standard deviation of this set of means.
    % 
    % "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
    % Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % whattype, the type of StatAv to perform:
    %           (i) 'seg': divide the time series into n segments
    %           (ii) 'len': divide the time series into segments of length n
    % 
    % n, either the number of subsegments ('seg') or their length ('len')
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if whattype is None:
        out = eng.run_function(1, 'SY_StatAv', x, )
    elif n is None:
        out = eng.run_function(1, 'SY_StatAv', x, whattype)
    else:
        out = eng.run_function(1, 'SY_StatAv', x, whattype, n)
    return outfunc(out)


class SY_StatAv(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % The StatAv measure is a simple mean-stationarity metric that divides
    % the time series into non-overlapping subsegments, calculates the mean in each
    % of these segments and returns the standard deviation of this set of means.
    % 
    % "Heart rate control in normal and aborted-SIDS infants", S. M. Pincus et al.
    % Am J. Physiol. Regul. Integr. Comp. Physiol. 264(3) R638 (1993)
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % whattype, the type of StatAv to perform:
    %           (i) 'seg': divide the time series into n segments
    %           (ii) 'len': divide the time series into segments of length n
    % 
    % n, either the number of subsegments ('seg') or their length ('len')
    % 
    ----------------------------------------
    """

    def __init__(self, whattype='len', n=100):
        super(SY_StatAv, self).__init__(add_descriptors=False)
        self.whattype = whattype
        self.n = n

    def eval(self, engine, x):
        return HCTSA_SY_StatAv(engine,
                               x,
                               whattype=self.whattype,
                               n=self.n)


def HCTSA_SY_StdNthDer(eng, x, n=6):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the standard deviation of the nth derivative of the time series.
    % 
    % Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    % Consultant in a Matlab forum, who stated that You can measure the standard
    % deviation of the nth derivative, if you like".
    % 
    % cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    % 
    % The derivative is estimated very simply by simply taking successive increments
    % of the time series; the process is repeated to obtain higher order
    % derivatives.
    % 
    %---INPUTS:
    % 
    % y, time series to analyze
    % 
    % n, the order of derivative to analyze
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return out
    if n is None:
        out = eng.run_function(1, 'SY_StdNthDer', x, )
    else:
        out = eng.run_function(1, 'SY_StdNthDer', x, n)
    return outfunc(out)


class SY_StdNthDer(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates the standard deviation of the nth derivative of the time series.
    % 
    % Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    % Consultant in a Matlab forum, who stated that You can measure the standard
    % deviation of the nth derivative, if you like".
    % 
    % cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    % 
    % The derivative is estimated very simply by simply taking successive increments
    % of the time series; the process is repeated to obtain higher order
    % derivatives.
    % 
    %---INPUTS:
    % 
    % y, time series to analyze
    % 
    % n, the order of derivative to analyze
    % 
    ----------------------------------------
    """

    def __init__(self, n=6):
        super(SY_StdNthDer, self).__init__(add_descriptors=False)
        self.n = n

    def eval(self, engine, x):
        return HCTSA_SY_StdNthDer(engine,
                                  x,
                                  n=self.n)


def HCTSA_SY_StdNthDerChange(eng, x, maxd=None):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This operation returns statistics on how the output of SY_StdNthDer changes
    % with the order of the derivative of the signal.
    % 
    % Operation inspired by a comment on the Matlab Central forum: "You can
    % measure the standard deviation of the n-th derivative, if you like." --
    % Vladimir Vassilevsky, DSP and Mixed Signal Design Consultant from
    % http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % maxd, the maximum derivative to take.
    % 
    % An exponential function, f(x) = Aexp(bx), is fitted to the variation across
    % successive derivatives; outputs are the parameters and quality of this fit.
    % 
    % Typically an excellent fit to exponential: regular signals decrease, irregular
    % signals increase...?
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['fexp_a',
                                                      'fexp_adjr2',
                                                      'fexp_b',
                                                      'fexp_r2',
                                                      'fexp_rmse']}
    if maxd is None:
        out = eng.run_function(1, 'SY_StdNthDerChange', x, )
    else:
        out = eng.run_function(1, 'SY_StdNthDerChange', x, maxd)
    return outfunc(out)


class SY_StdNthDerChange(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This operation returns statistics on how the output of SY_StdNthDer changes
    % with the order of the derivative of the signal.
    % 
    % Operation inspired by a comment on the Matlab Central forum: "You can
    % measure the standard deviation of the n-th derivative, if you like." --
    % Vladimir Vassilevsky, DSP and Mixed Signal Design Consultant from
    % http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % maxd, the maximum derivative to take.
    % 
    % An exponential function, f(x) = Aexp(bx), is fitted to the variation across
    % successive derivatives; outputs are the parameters and quality of this fit.
    % 
    % Typically an excellent fit to exponential: regular signals decrease, irregular
    % signals increase...?
    % 
    ----------------------------------------
    """

    outnames = ('fexp_a',
                'fexp_adjr2',
                'fexp_b',
                'fexp_r2',
                'fexp_rmse')

    def __init__(self, maxd=None):
        super(SY_StdNthDerChange, self).__init__(add_descriptors=False)
        self.maxd = maxd

    def eval(self, engine, x):
        return HCTSA_SY_StdNthDerChange(engine,
                                        x,
                                        maxd=self.maxd)


def HCTSA_SY_TISEAN_nstat_z(eng, x, nseg=5, embedparams=(1, 3)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the nstat_z routine from the TISEAN package for nonlinear time-series
    % analysis to calculate cross-forecast errors of zeroth-order models for the
    % time-delay embedded time series.
    % 
    % The program looks for nonstationarity in a time series by dividing it
    % into a number of segments and calculating the cross-forecast errors
    % between the different segments. The model used for the forecast is
    % zeroth order model as proposed by Schreiber.
    % 
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    % 
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    % 
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % nseg, the number of equally-spaced segments to divide the time series into,
    %       and used to predict the other time series segments
    % 
    % embedparams, in the form {tau,m}, as usual for BF_embed. That is, for an
    %               embedding dimension, tau, and embedding dimension, m. E.g.,
    %               {1,3} has a time-delay of 1 and embedding dimension of 3.
    % 
    % 
    %---OUTPUTS: include the trace of the cross-prediction error matrix, the mean,
    % minimum, and maximum cross-prediction error, the minimum off-diagonal
    % cross-prediction error, and eigenvalues of the cross-prediction error matrix.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['iqr',
                                                      'iqroffdiag',
                                                      'max',
                                                      'maxeig',
                                                      'maximageig',
                                                      'mean',
                                                      'median',
                                                      'min',
                                                      'mineig',
                                                      'minimageig',
                                                      'minlower',
                                                      'minoffdiag',
                                                      'minupper',
                                                      'range',
                                                      'rangeeig',
                                                      'rangemean',
                                                      'rangemedian',
                                                      'rangeoffdiag',
                                                      'rangerange',
                                                      'rangestd',
                                                      'std',
                                                      'stdeig',
                                                      'stdmean',
                                                      'stdmedian',
                                                      'stdoffdiag',
                                                      'stdrange',
                                                      'stdstd',
                                                      'trace']}
    if nseg is None:
        out = eng.run_function(1, 'SY_TISEAN_nstat_z', x, )
    elif embedparams is None:
        out = eng.run_function(1, 'SY_TISEAN_nstat_z', x, nseg)
    else:
        out = eng.run_function(1, 'SY_TISEAN_nstat_z', x, nseg, embedparams)
    return outfunc(out)


class SY_TISEAN_nstat_z(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the nstat_z routine from the TISEAN package for nonlinear time-series
    % analysis to calculate cross-forecast errors of zeroth-order models for the
    % time-delay embedded time series.
    % 
    % The program looks for nonstationarity in a time series by dividing it
    % into a number of segments and calculating the cross-forecast errors
    % between the different segments. The model used for the forecast is
    % zeroth order model as proposed by Schreiber.
    % 
    % cf. "Practical implementation of nonlinear time series methods: The TISEAN
    % package", R. Hegger, H. Kantz, and T. Schreiber, Chaos 9(2) 413 (1999)
    % 
    % Available here:
    % http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    % 
    % The TISEAN routines are performed in the command line using 'system' commands
    % in Matlab, and require that TISEAN is installed and compiled, and able to be
    % executed in the command line.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % nseg, the number of equally-spaced segments to divide the time series into,
    %       and used to predict the other time series segments
    % 
    % embedparams, in the form {tau,m}, as usual for BF_embed. That is, for an
    %               embedding dimension, tau, and embedding dimension, m. E.g.,
    %               {1,3} has a time-delay of 1 and embedding dimension of 3.
    % 
    % 
    %---OUTPUTS: include the trace of the cross-prediction error matrix, the mean,
    % minimum, and maximum cross-prediction error, the minimum off-diagonal
    % cross-prediction error, and eigenvalues of the cross-prediction error matrix.
    % 
    ----------------------------------------
    """

    outnames = ('iqr',
                'iqroffdiag',
                'max',
                'maxeig',
                'maximageig',
                'mean',
                'median',
                'min',
                'mineig',
                'minimageig',
                'minlower',
                'minoffdiag',
                'minupper',
                'range',
                'rangeeig',
                'rangemean',
                'rangemedian',
                'rangeoffdiag',
                'rangerange',
                'rangestd',
                'std',
                'stdeig',
                'stdmean',
                'stdmedian',
                'stdoffdiag',
                'stdrange',
                'stdstd',
                'trace')

    def __init__(self, nseg=5, embedparams=(1, 3)):
        super(SY_TISEAN_nstat_z, self).__init__(add_descriptors=False)
        self.nseg = nseg
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_SY_TISEAN_nstat_z(engine,
                                       x,
                                       nseg=self.nseg,
                                       embedparams=self.embedparams)


def HCTSA_SY_VarRatioTest(eng, x, periods=4, IIDs=0):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This code performs a variance ratio test on the time series, implemented using
    % the vratiotest function from Matlab's Econometrics Toolbox.
    % 
    % The test assesses the null hypothesis of a random walk in the time series,
    % which is rejected for some critical p-value.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % periods, a vector (or scalar) of period(s)
    % 
    % IIDs, a vector (or scalar) representing boolean values indicating whether to
    %       assume independent and identically distributed (IID) innovations for
    %       each period.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['IIDperiodmaxpValue',
                                                      'IIDperiodminpValue',
                                                      'maxpValue',
                                                      'maxstat',
                                                      'meanpValue',
                                                      'meanstat',
                                                      'minpValue',
                                                      'minstat',
                                                      'pValue',
                                                      'periodmaxpValue',
                                                      'periodminpValue',
                                                      'ratio',
                                                      'stat']}
    if periods is None:
        out = eng.run_function(1, 'SY_VarRatioTest', x, )
    elif IIDs is None:
        out = eng.run_function(1, 'SY_VarRatioTest', x, periods)
    else:
        out = eng.run_function(1, 'SY_VarRatioTest', x, periods, IIDs)
    return outfunc(out)


class SY_VarRatioTest(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % This code performs a variance ratio test on the time series, implemented using
    % the vratiotest function from Matlab's Econometrics Toolbox.
    % 
    % The test assesses the null hypothesis of a random walk in the time series,
    % which is rejected for some critical p-value.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % periods, a vector (or scalar) of period(s)
    % 
    % IIDs, a vector (or scalar) representing boolean values indicating whether to
    %       assume independent and identically distributed (IID) innovations for
    %       each period.
    % 
    ----------------------------------------
    """

    outnames = ('IIDperiodmaxpValue',
                'IIDperiodminpValue',
                'maxpValue',
                'maxstat',
                'meanpValue',
                'meanstat',
                'minpValue',
                'minstat',
                'pValue',
                'periodmaxpValue',
                'periodminpValue',
                'ratio',
                'stat')

    def __init__(self, periods=4, IIDs=0):
        super(SY_VarRatioTest, self).__init__(add_descriptors=False)
        self.periods = periods
        self.IIDs = IIDs

    def eval(self, engine, x):
        return HCTSA_SY_VarRatioTest(engine,
                                     x,
                                     periods=self.periods,
                                     IIDs=self.IIDs)


def HCTSA_TSTL_delaytime(eng, x, maxdelay=0.1, past=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the TSTOOL code delaytime, that computes an optimal delay time using the
    % method of Parlitz and Wichard (this method is specified in the TSTOOL
    % documentation but without reference).
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time series data
    % 
    % maxdelay, maximum value of the delay to consider (can also specify a
    %           proportion of time series length)
    %           
    % past, the TSTOOL documentation describes this parameter as "?", which is
    %       relatively uninformative.
    % 
    % 
    % It's a stochastic algorithm, so it must rely on some random sampling of the
    % input time series... A bit of a strange one, but I'll return some statistics
    % and see what they do.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['difftau12',
                                                      'difftau13',
                                                      'maxtau',
                                                      'meantau',
                                                      'mintau',
                                                      'stdtau',
                                                      'tau1',
                                                      'tau2',
                                                      'tau3']}
    if maxdelay is None:
        out = eng.run_function(1, 'TSTL_delaytime', x, )
    elif past is None:
        out = eng.run_function(1, 'TSTL_delaytime', x, maxdelay)
    else:
        out = eng.run_function(1, 'TSTL_delaytime', x, maxdelay, past)
    return outfunc(out)


class TSTL_delaytime(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the TSTOOL code delaytime, that computes an optimal delay time using the
    % method of Parlitz and Wichard (this method is specified in the TSTOOL
    % documentation but without reference).
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % y, column vector of time series data
    % 
    % maxdelay, maximum value of the delay to consider (can also specify a
    %           proportion of time series length)
    %           
    % past, the TSTOOL documentation describes this parameter as "?", which is
    %       relatively uninformative.
    % 
    % 
    % It's a stochastic algorithm, so it must rely on some random sampling of the
    % input time series... A bit of a strange one, but I'll return some statistics
    % and see what they do.
    % 
    ----------------------------------------
    """

    outnames = ('difftau12',
                'difftau13',
                'maxtau',
                'meantau',
                'mintau',
                'stdtau',
                'tau1',
                'tau2',
                'tau3')

    def __init__(self, maxdelay=0.1, past=1):
        super(TSTL_delaytime, self).__init__(add_descriptors=False)
        self.maxdelay = maxdelay
        self.past = past

    def eval(self, engine, x):
        return HCTSA_TSTL_delaytime(engine,
                                    x,
                                    maxdelay=self.maxdelay,
                                    past=self.past)


def HCTSA_TSTL_localdensity(eng, x, NNR=5, past=40, embedparams=('ac', 'cao')):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses TSTOOL code localdensity, which is very poorly documented in the TSTOOL
    % documentation, but we can assume it returns local density estimates in the
    % time-delay embedding space.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, the time series as a column vector
    % 
    % NNR, number of nearest neighbours to compute
    % 
    % past, number of time-correlated points to discard (samples)
    % 
    % embedparams, the embedding parameters, inputs to BF_embed as {tau,m}, where
    %               tau and m can be characters specifying a given automatic method
    %               of determining tau and/or m (see BF_embed).
    % 
    %---OUTPUTS: various statistics on the local density estimates at each point in
    % the time-delay embedding, including the minimum and maximum values, the range,
    % the standard deviation, mean, median, and autocorrelation.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['ac1den',
                                                      'ac2den',
                                                      'ac3den',
                                                      'ac4den',
                                                      'ac5den',
                                                      'iqrden',
                                                      'maxden',
                                                      'meanden',
                                                      'medianden',
                                                      'minden',
                                                      'rangeden',
                                                      'stdden',
                                                      'tauacden',
                                                      'taumiden']}
    if NNR is None:
        out = eng.run_function(1, 'TSTL_localdensity', x, )
    elif past is None:
        out = eng.run_function(1, 'TSTL_localdensity', x, NNR)
    elif embedparams is None:
        out = eng.run_function(1, 'TSTL_localdensity', x, NNR, past)
    else:
        out = eng.run_function(1, 'TSTL_localdensity', x, NNR, past, embedparams)
    return outfunc(out)


class TSTL_localdensity(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses TSTOOL code localdensity, which is very poorly documented in the TSTOOL
    % documentation, but we can assume it returns local density estimates in the
    % time-delay embedding space.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, the time series as a column vector
    % 
    % NNR, number of nearest neighbours to compute
    % 
    % past, number of time-correlated points to discard (samples)
    % 
    % embedparams, the embedding parameters, inputs to BF_embed as {tau,m}, where
    %               tau and m can be characters specifying a given automatic method
    %               of determining tau and/or m (see BF_embed).
    % 
    %---OUTPUTS: various statistics on the local density estimates at each point in
    % the time-delay embedding, including the minimum and maximum values, the range,
    % the standard deviation, mean, median, and autocorrelation.
    % 
    ----------------------------------------
    """

    outnames = ('ac1den',
                'ac2den',
                'ac3den',
                'ac4den',
                'ac5den',
                'iqrden',
                'maxden',
                'meanden',
                'medianden',
                'minden',
                'rangeden',
                'stdden',
                'tauacden',
                'taumiden')

    def __init__(self, NNR=5, past=40, embedparams=('ac', 'cao')):
        super(TSTL_localdensity, self).__init__(add_descriptors=False)
        self.NNR = NNR
        self.past = past
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_TSTL_localdensity(engine,
                                       x,
                                       NNR=self.NNR,
                                       past=self.past,
                                       embedparams=self.embedparams)


def HCTSA_TSTL_predict(eng, x, plen=1, NNR=1, stepsize=2, pmode=1, embedparams=(1, 10)):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % References TSTOOL code 'predict', which does local constant iterative
    % prediction for scalar data using fast nearest neighbour searching. There are
    % four modes available for the prediction output.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, scalar column vector time series
    % 
    % plen, prediction length in samples or as proportion of time series length NNR,
    % 
    % NNR, number of nearest neighbours
    % 
    % stepsize, number of samples to step for each prediction
    % 
    % pmode, prediction mode, four options:
    %           (i) 0: output vectors are means of images of nearest neighbours
    %           (ii) 1: output vectors are distance-weighted means of images
    %                     nearest neighbours
    %           (iii) 2: output vectors are calculated using local flow and the
    %                    mean of the images of the neighbours
    %           (iv) 3: output vectors are calculated using local flow and the
    %                    weighted mean of the images of the neighbours
    % embedparams, as usual to feed into BF_embed, except that now you can set
    %              to zero to not embed.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['Nlagxcorr',
                                                      'ac1bestres',
                                                      'acs1_10_sumabsdiffpred1',
                                                      'bestpred1rmsres',
                                                      'fracres005',
                                                      'fracres01',
                                                      'fracres02',
                                                      'fracres03',
                                                      'fracres05',
                                                      'maxabsxcf',
                                                      'maxabsxcflag',
                                                      'maxxcf',
                                                      'maxxcflag',
                                                      'meanxcf',
                                                      'minxcf',
                                                      'pred1_ac1',
                                                      'pred1_ac2',
                                                      'pred1ac1diff',
                                                      'pred1ac2diff',
                                                      'pred1maxc',
                                                      'pred1mean',
                                                      'pred1minc',
                                                      'pred1rangec',
                                                      'pred1rmsres',
                                                      'pred1std',
                                                      'pred_tau_comp',
                                                      'predminc',
                                                      'stdxcf']}
    if plen is None:
        out = eng.run_function(1, 'TSTL_predict', x, )
    elif NNR is None:
        out = eng.run_function(1, 'TSTL_predict', x, plen)
    elif stepsize is None:
        out = eng.run_function(1, 'TSTL_predict', x, plen, NNR)
    elif pmode is None:
        out = eng.run_function(1, 'TSTL_predict', x, plen, NNR, stepsize)
    elif embedparams is None:
        out = eng.run_function(1, 'TSTL_predict', x, plen, NNR, stepsize, pmode)
    else:
        out = eng.run_function(1, 'TSTL_predict', x, plen, NNR, stepsize, pmode, embedparams)
    return outfunc(out)


class TSTL_predict(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % References TSTOOL code 'predict', which does local constant iterative
    % prediction for scalar data using fast nearest neighbour searching. There are
    % four modes available for the prediction output.
    % 
    % TSTOOL: http://www.physik3.gwdg.de/tstool/
    % 
    %---INPUTS:
    % 
    % y, scalar column vector time series
    % 
    % plen, prediction length in samples or as proportion of time series length NNR,
    % 
    % NNR, number of nearest neighbours
    % 
    % stepsize, number of samples to step for each prediction
    % 
    % pmode, prediction mode, four options:
    %           (i) 0: output vectors are means of images of nearest neighbours
    %           (ii) 1: output vectors are distance-weighted means of images
    %                     nearest neighbours
    %           (iii) 2: output vectors are calculated using local flow and the
    %                    mean of the images of the neighbours
    %           (iv) 3: output vectors are calculated using local flow and the
    %                    weighted mean of the images of the neighbours
    % embedparams, as usual to feed into BF_embed, except that now you can set
    %              to zero to not embed.
    % 
    ----------------------------------------
    """

    outnames = ('Nlagxcorr',
                'ac1bestres',
                'acs1_10_sumabsdiffpred1',
                'bestpred1rmsres',
                'fracres005',
                'fracres01',
                'fracres02',
                'fracres03',
                'fracres05',
                'maxabsxcf',
                'maxabsxcflag',
                'maxxcf',
                'maxxcflag',
                'meanxcf',
                'minxcf',
                'pred1_ac1',
                'pred1_ac2',
                'pred1ac1diff',
                'pred1ac2diff',
                'pred1maxc',
                'pred1mean',
                'pred1minc',
                'pred1rangec',
                'pred1rmsres',
                'pred1std',
                'pred_tau_comp',
                'predminc',
                'stdxcf')

    def __init__(self, plen=1, NNR=1, stepsize=2, pmode=1, embedparams=(1, 10)):
        super(TSTL_predict, self).__init__(add_descriptors=False)
        self.plen = plen
        self.NNR = NNR
        self.stepsize = stepsize
        self.pmode = pmode
        self.embedparams = embedparams

    def eval(self, engine, x):
        return HCTSA_TSTL_predict(engine,
                                  x,
                                  plen=self.plen,
                                  NNR=self.NNR,
                                  stepsize=self.stepsize,
                                  pmode=self.pmode,
                                  embedparams=self.embedparams)


def HCTSA_WL_DetailCoeffs(eng, x, wname='db3', maxlevel='max'):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares the detail coefficients obtained at each level of the wavelet
    % decomposition from 1 to the maximum possible level for the wavelet given the
    % length of the input time series (computed using wmaxlev from
    % Matlab's Wavelet Toolbox).
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    %           
    % maxlevel, the maximum wavelet decomposition level (can also set to 'max' to be
    %               that determined by wmaxlev)
    % 
    %---OUTPUTS: A set of statistics on the detail coefficients.
    %       
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['corrcoef_max_medians',
                                                      'max1on2_max',
                                                      'max1on2_mean',
                                                      'max1on2_median',
                                                      'max_max',
                                                      'max_mean',
                                                      'max_median',
                                                      'std_max',
                                                      'std_mean',
                                                      'std_median',
                                                      'wheremax_max',
                                                      'wheremax_mean',
                                                      'wheremax_median',
                                                      'wslesr_max',
                                                      'wslesr_mean',
                                                      'wslesr_median']}
    if wname is None:
        out = eng.run_function(1, 'WL_DetailCoeffs', x, )
    elif maxlevel is None:
        out = eng.run_function(1, 'WL_DetailCoeffs', x, wname)
    else:
        out = eng.run_function(1, 'WL_DetailCoeffs', x, wname, maxlevel)
    return outfunc(out)


class WL_DetailCoeffs(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Compares the detail coefficients obtained at each level of the wavelet
    % decomposition from 1 to the maximum possible level for the wavelet given the
    % length of the input time series (computed using wmaxlev from
    % Matlab's Wavelet Toolbox).
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    %           
    % maxlevel, the maximum wavelet decomposition level (can also set to 'max' to be
    %               that determined by wmaxlev)
    % 
    %---OUTPUTS: A set of statistics on the detail coefficients.
    %       
    ----------------------------------------
    """

    outnames = ('corrcoef_max_medians',
                'max1on2_max',
                'max1on2_mean',
                'max1on2_median',
                'max_max',
                'max_mean',
                'max_median',
                'std_max',
                'std_mean',
                'std_median',
                'wheremax_max',
                'wheremax_mean',
                'wheremax_median',
                'wslesr_max',
                'wslesr_mean',
                'wslesr_median')

    def __init__(self, wname='db3', maxlevel='max'):
        super(WL_DetailCoeffs, self).__init__(add_descriptors=False)
        self.wname = wname
        self.maxlevel = maxlevel

    def eval(self, engine, x):
        return HCTSA_WL_DetailCoeffs(engine,
                                     x,
                                     wname=self.wname,
                                     maxlevel=self.maxlevel)


def HCTSA_WL_coeffs(eng, x, wname='db3', level=5):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Performs a wavelet decomposition of the time series using a given wavelet at a
    % given level and returns a set of statistics on the coefficients obtained.
    % 
    % Uses Matlab's Wavelet Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the wavelet name, e.g., 'db3' (see Wavelet Toolbox Documentation for
    %                                       all options)
    % 
    % level, the level of wavelet decomposition
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['max_coeff',
                                                      'mean_coeff',
                                                      'med_coeff',
                                                      'wb10m',
                                                      'wb1m',
                                                      'wb25m',
                                                      'wb50m',
                                                      'wb75m',
                                                      'wb90m',
                                                      'wb99m']}
    if wname is None:
        out = eng.run_function(1, 'WL_coeffs', x, )
    elif level is None:
        out = eng.run_function(1, 'WL_coeffs', x, wname)
    else:
        out = eng.run_function(1, 'WL_coeffs', x, wname, level)
    return outfunc(out)


class WL_coeffs(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Performs a wavelet decomposition of the time series using a given wavelet at a
    % given level and returns a set of statistics on the coefficients obtained.
    % 
    % Uses Matlab's Wavelet Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the wavelet name, e.g., 'db3' (see Wavelet Toolbox Documentation for
    %                                       all options)
    % 
    % level, the level of wavelet decomposition
    % 
    ----------------------------------------
    """

    outnames = ('max_coeff',
                'mean_coeff',
                'med_coeff',
                'wb10m',
                'wb1m',
                'wb25m',
                'wb50m',
                'wb75m',
                'wb90m',
                'wb99m')

    def __init__(self, wname='db3', level=5):
        super(WL_coeffs, self).__init__(add_descriptors=False)
        self.wname = wname
        self.level = level

    def eval(self, engine, x):
        return HCTSA_WL_coeffs(engine,
                               x,
                               wname=self.wname,
                               level=self.level)


def HCTSA_WL_cwt(eng, x, wname='sym2', maxscale=32):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Applies a continuous wavelet transform to the time series using the function
    % cwt from Matlab's Wavelet Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet),
    %                           etc. (see Wavelet Toolbox Documentation for all
    %                           options)
    % 
    % maxscale, the maximum scale of wavelet analysis.
    % 
    % 
    %---OUTPUTS: statistics on the coefficients, entropy, and results of
    % coefficients summed across scales.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['SC_h',
                                                      'dd_SC_h',
                                                      'exp_muhat',
                                                      'gam1',
                                                      'gam2',
                                                      'maxSC',
                                                      'max_ssc',
                                                      'maxabsC',
                                                      'maxonmedC',
                                                      'maxonmedSC',
                                                      'maxonmed_ssc',
                                                      'meanC',
                                                      'meanSC',
                                                      'meanabsC',
                                                      'medianSC',
                                                      'medianabsC',
                                                      'min_ssc',
                                                      'pcross_maxssc50',
                                                      'pover80',
                                                      'pover90',
                                                      'pover95',
                                                      'pover98',
                                                      'pover99',
                                                      'stat_2_m_s',
                                                      'stat_2_s_m',
                                                      'stat_2_s_s',
                                                      'stat_5_m_s',
                                                      'stat_5_s_m',
                                                      'stat_5_s_s',
                                                      'std_ssc']}
    if wname is None:
        out = eng.run_function(1, 'WL_cwt', x, )
    elif maxscale is None:
        out = eng.run_function(1, 'WL_cwt', x, wname)
    else:
        out = eng.run_function(1, 'WL_cwt', x, wname, maxscale)
    return outfunc(out)


class WL_cwt(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Applies a continuous wavelet transform to the time series using the function
    % cwt from Matlab's Wavelet Toolbox.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the wavelet name, e.g., 'db3' (Daubechies wavelet), 'sym2' (Symlet),
    %                           etc. (see Wavelet Toolbox Documentation for all
    %                           options)
    % 
    % maxscale, the maximum scale of wavelet analysis.
    % 
    % 
    %---OUTPUTS: statistics on the coefficients, entropy, and results of
    % coefficients summed across scales.
    % 
    ----------------------------------------
    """

    outnames = ('SC_h',
                'dd_SC_h',
                'exp_muhat',
                'gam1',
                'gam2',
                'maxSC',
                'max_ssc',
                'maxabsC',
                'maxonmedC',
                'maxonmedSC',
                'maxonmed_ssc',
                'meanC',
                'meanSC',
                'meanabsC',
                'medianSC',
                'medianabsC',
                'min_ssc',
                'pcross_maxssc50',
                'pover80',
                'pover90',
                'pover95',
                'pover98',
                'pover99',
                'stat_2_m_s',
                'stat_2_s_m',
                'stat_2_s_s',
                'stat_5_m_s',
                'stat_5_s_m',
                'stat_5_s_s',
                'std_ssc')

    def __init__(self, wname='sym2', maxscale=32):
        super(WL_cwt, self).__init__(add_descriptors=False)
        self.wname = wname
        self.maxscale = maxscale

    def eval(self, engine, x):
        return HCTSA_WL_cwt(engine,
                            x,
                            wname=self.wname,
                            maxscale=self.maxscale)


def HCTSA_WL_dwtcoeff(eng, x, wname='sym2', level=5):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Decomposes the time series using a given wavelet and outputs statistics on the
    % coefficients obtained up to a maximum level, level.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % wname, the mother wavelet, e.g., 'db3', 'sym2' (see Wavelet Toolbox
    %           Documentation)
    %           
    % level, the level of wavelet decomposition (can be set to 'max' for the maximum
    %               level determined by wmaxlev)
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['maxd_l1',
                                                      'maxd_l2',
                                                      'maxd_l3',
                                                      'maxd_l4',
                                                      'maxd_l5',
                                                      'mind_l1',
                                                      'mind_l2',
                                                      'mind_l3',
                                                      'mind_l4',
                                                      'mind_l5',
                                                      'stdd_l1',
                                                      'stdd_l2',
                                                      'stdd_l3',
                                                      'stdd_l4',
                                                      'stdd_l5',
                                                      'stddd_l1',
                                                      'stddd_l2',
                                                      'stddd_l3',
                                                      'stddd_l4',
                                                      'stddd_l5']}
    if wname is None:
        out = eng.run_function(1, 'WL_dwtcoeff', x, )
    elif level is None:
        out = eng.run_function(1, 'WL_dwtcoeff', x, wname)
    else:
        out = eng.run_function(1, 'WL_dwtcoeff', x, wname, level)
    return outfunc(out)


class WL_dwtcoeff(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Decomposes the time series using a given wavelet and outputs statistics on the
    % coefficients obtained up to a maximum level, level.
    % 
    %---INPUTS:
    % 
    % y, the input time series
    % 
    % wname, the mother wavelet, e.g., 'db3', 'sym2' (see Wavelet Toolbox
    %           Documentation)
    %           
    % level, the level of wavelet decomposition (can be set to 'max' for the maximum
    %               level determined by wmaxlev)
    % 
    ----------------------------------------
    """

    outnames = ('maxd_l1',
                'maxd_l2',
                'maxd_l3',
                'maxd_l4',
                'maxd_l5',
                'mind_l1',
                'mind_l2',
                'mind_l3',
                'mind_l4',
                'mind_l5',
                'stdd_l1',
                'stdd_l2',
                'stdd_l3',
                'stdd_l4',
                'stdd_l5',
                'stddd_l1',
                'stddd_l2',
                'stddd_l3',
                'stddd_l4',
                'stddd_l5')

    def __init__(self, wname='sym2', level=5):
        super(WL_dwtcoeff, self).__init__(add_descriptors=False)
        self.wname = wname
        self.level = level

    def eval(self, engine, x):
        return HCTSA_WL_dwtcoeff(engine,
                                 x,
                                 wname=self.wname,
                                 level=self.level)


def HCTSA_WL_fBM(eng, x):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the wfbmesti function from Matlab's Wavelet Toolbox to estimate the
    % parameters of fractional Gaussian Noise, or fractional Brownian motion in a
    % time series.
    % 
    %---INPUT:
    % y, the time series to analyze.
    % 
    %---OUTPUTS: All three outputs of wfbmesti are returned from this function.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['p1',
                                                      'p2',
                                                      'p3']}
    out = eng.run_function(1, 'WL_fBM', x, )
    return outfunc(out)


class WL_fBM(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Uses the wfbmesti function from Matlab's Wavelet Toolbox to estimate the
    % parameters of fractional Gaussian Noise, or fractional Brownian motion in a
    % time series.
    % 
    %---INPUT:
    % y, the time series to analyze.
    % 
    %---OUTPUTS: All three outputs of wfbmesti are returned from this function.
    % 
    ----------------------------------------
    """

    outnames = ('p1',
                'p2',
                'p3')

    def __init__(self, ):
        super(WL_fBM, self).__init__(add_descriptors=False)

    @staticmethod
    def eval(engine, x):
        return HCTSA_WL_fBM(engine, x)


def HCTSA_WL_scal2frq(eng, x, wname='db3', amax='max', delta=1):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates frequency components in a periodic time series using functions from
    % Matlab's Wavelet Toolbox, including the scal2frq function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    % 
    % amax, the maximum scale / level (can be 'max' to set according to wmaxlev)
    % 
    % delta, the sampling period
    % 
    %---OUTPUTS: the level with the highest energy coefficients, the dominant
    % period, and the dominant pseudo-frequency.
    % 
    % Adapted from example in Matlab Wavelet Toolbox documentation. It's kind of a
    % weird idea to apply the method to generic time series.
    % 
    ----------------------------------------
    """
    def outfunc(out):
        return {outname: out[outname] for outname in ['lmax',
                                                      'period',
                                                      'pf']}
    if wname is None:
        out = eng.run_function(1, 'WL_scal2frq', x, )
    elif amax is None:
        out = eng.run_function(1, 'WL_scal2frq', x, wname)
    elif delta is None:
        out = eng.run_function(1, 'WL_scal2frq', x, wname, amax)
    else:
        out = eng.run_function(1, 'WL_scal2frq', x, wname, amax, delta)
    return outfunc(out)


class WL_scal2frq(Configurable):
    """
    Matlab doc:
    ----------------------------------------
    % 
    % Estimates frequency components in a periodic time series using functions from
    % Matlab's Wavelet Toolbox, including the scal2frq function.
    % 
    %---INPUTS:
    % y, the input time series
    % 
    % wname, the name of the mother wavelet to analyze the data with: e.g., 'db3',
    %           'sym2', cf. Wavelet Toolbox Documentation for details
    % 
    % amax, the maximum scale / level (can be 'max' to set according to wmaxlev)
    % 
    % delta, the sampling period
    % 
    %---OUTPUTS: the level with the highest energy coefficients, the dominant
    % period, and the dominant pseudo-frequency.
    % 
    % Adapted from example in Matlab Wavelet Toolbox documentation. It's kind of a
    % weird idea to apply the method to generic time series.
    % 
    ----------------------------------------
    """

    outnames = ('lmax',
                'period',
                'pf')

    def __init__(self, wname='db3', amax='max', delta=1):
        super(WL_scal2frq, self).__init__(add_descriptors=False)
        self.wname = wname
        self.amax = amax
        self.delta = delta

    def eval(self, engine, x):
        return HCTSA_WL_scal2frq(engine,
                                 x,
                                 wname=self.wname,
                                 amax=self.amax,
                                 delta=self.delta)


ALL_HCTSA_FUNCS = (
    HCTSA_CO_AddNoise,
    HCTSA_CO_AutoCorr,
    HCTSA_CO_AutoCorrShape,
    HCTSA_CO_CompareMinAMI,
    HCTSA_CO_Embed2,
    HCTSA_CO_Embed2_AngleTau,
    HCTSA_CO_Embed2_Basic,
    HCTSA_CO_Embed2_Dist,
    HCTSA_CO_Embed2_Shapes,
    HCTSA_CO_FirstMin,
    HCTSA_CO_FirstZero,
    HCTSA_CO_HistogramAMI,
    HCTSA_CO_NonlinearAutocorr,
    HCTSA_CO_RM_AMInformation,
    HCTSA_CO_StickAngles,
    HCTSA_CO_TSTL_AutoCorrMethod,
    HCTSA_CO_TSTL_amutual,
    HCTSA_CO_TSTL_amutual2,
    HCTSA_CO_TranslateShape,
    HCTSA_CO_f1ecac,
    HCTSA_CO_fzcglscf,
    HCTSA_CO_glscf,
    HCTSA_CO_tc3,
    HCTSA_CO_trev,
    HCTSA_CP_ML_StepDetect,
    HCTSA_CP_l1pwc_sweep_lambda,
    HCTSA_CP_wavelet_varchg,
    HCTSA_DN_Burstiness,
    HCTSA_DN_CompareKSFit,
    HCTSA_DN_Compare_zscore,
    HCTSA_DN_Cumulants,
    HCTSA_DN_CustomSkewness,
    HCTSA_DN_FitKernelSmooth,
    HCTSA_DN_Fit_mle,
    HCTSA_DN_HighLowMu,
    HCTSA_DN_HistogramMode,
    HCTSA_DN_Mean,
    HCTSA_DN_MinMax,
    HCTSA_DN_Moments,
    HCTSA_DN_OutlierInclude,
    HCTSA_DN_OutlierTest,
    HCTSA_DN_ProportionValues,
    HCTSA_DN_Quantile,
    HCTSA_DN_RemovePoints,
    HCTSA_DN_SimpleFit,
    HCTSA_DN_Spread,
    HCTSA_DN_TrimmedMean,
    HCTSA_DN_Withinp,
    HCTSA_DN_cv,
    HCTSA_DN_nlogL_norm,
    HCTSA_DN_pleft,
    HCTSA_DT_IsSeasonal,
    HCTSA_EN_ApEn,
    HCTSA_EN_DistributionEntropy,
    HCTSA_EN_MS_shannon,
    HCTSA_EN_PermEn,
    HCTSA_EN_RM_entropy,
    HCTSA_EN_Randomize,
    HCTSA_EN_SampEn,
    HCTSA_EN_Shannonpdf,
    HCTSA_EN_TSentropy,
    HCTSA_EN_wentropy,
    HCTSA_EX_MovingThreshold,
    HCTSA_FC_LocalSimple,
    HCTSA_FC_LoopLocalSimple,
    HCTSA_FC_Surprise,
    HCTSA_HT_DistributionTest,
    HCTSA_HT_HypothesisTest,
    HCTSA_MD_hrv_classic,
    HCTSA_MD_pNN,
    HCTSA_MD_polvar,
    HCTSA_MD_rawHRVmeas,
    HCTSA_MF_ARMA_orders,
    HCTSA_MF_AR_arcov,
    HCTSA_MF_CompareAR,
    HCTSA_MF_CompareTestSets,
    HCTSA_MF_ExpSmoothing,
    HCTSA_MF_FitSubsegments,
    HCTSA_MF_GARCHcompare,
    HCTSA_MF_GARCHfit,
    HCTSA_MF_GP_FitAcross,
    HCTSA_MF_GP_LearnHyperp,
    HCTSA_MF_GP_LocalPrediction,
    HCTSA_MF_GP_hyperparameters,
    HCTSA_MF_ResidualAnalysis,
    HCTSA_MF_StateSpaceCompOrder,
    HCTSA_MF_StateSpace_n4sid,
    HCTSA_MF_arfit,
    HCTSA_MF_armax,
    HCTSA_MF_hmm_CompareNStates,
    HCTSA_MF_hmm_fit,
    HCTSA_MF_steps_ahead,
    HCTSA_NL_BoxCorrDim,
    HCTSA_NL_CaosMethod,
    HCTSA_NL_MS_LZcomplexity,
    HCTSA_NL_MS_fnn,
    HCTSA_NL_MS_nlpe,
    HCTSA_NL_TISEAN_c1,
    HCTSA_NL_TISEAN_d2,
    HCTSA_NL_TSTL_FractalDimensions,
    HCTSA_NL_TSTL_GPCorrSum,
    HCTSA_NL_TSTL_LargestLyap,
    HCTSA_NL_TSTL_PoincareSection,
    HCTSA_NL_TSTL_ReturnTime,
    HCTSA_NL_TSTL_TakensEstimator,
    HCTSA_NL_TSTL_acp,
    HCTSA_NL_TSTL_dimensions,
    HCTSA_NL_crptool_fnn,
    HCTSA_NL_embed_PCA,
    HCTSA_NW_VisibilityGraph,
    HCTSA_PD_PeriodicityWang,
    HCTSA_PH_ForcePotential,
    HCTSA_PH_Walker,
    HCTSA_PP_Compare,
    HCTSA_PP_Iterate,
    HCTSA_PP_ModelFit,
    HCTSA_PP_PreProcess,
    HCTSA_RN_Gaussian,
    HCTSA_SB_BinaryStats,
    HCTSA_SB_BinaryStretch,
    HCTSA_SB_CoarseGrain,
    HCTSA_SB_MotifThree,
    HCTSA_SB_MotifTwo,
    HCTSA_SB_TransitionMatrix,
    HCTSA_SB_TransitionpAlphabet,
    HCTSA_SC_FluctAnal,
    HCTSA_SC_HurstExponent,
    HCTSA_SC_fastdfa,
    HCTSA_SD_MakeSurrogates,
    HCTSA_SD_SurrogateTest,
    HCTSA_SD_TSTL_surrogates,
    HCTSA_SP_Summaries,
    HCTSA_ST_FitPolynomial,
    HCTSA_ST_Length,
    HCTSA_ST_LocalExtrema,
    HCTSA_ST_MomentCorr,
    HCTSA_ST_SimpleStats,
    HCTSA_SY_DriftingMean,
    HCTSA_SY_DynWin,
    HCTSA_SY_KPSStest,
    HCTSA_SY_LinearTrend,
    HCTSA_SY_LocalDistributions,
    HCTSA_SY_LocalGlobal,
    HCTSA_SY_PPtest,
    HCTSA_SY_RangeEvolve,
    HCTSA_SY_SlidingWindow,
    HCTSA_SY_SpreadRandomLocal,
    HCTSA_SY_StatAv,
    HCTSA_SY_StdNthDer,
    HCTSA_SY_StdNthDerChange,
    HCTSA_SY_TISEAN_nstat_z,
    HCTSA_SY_VarRatioTest,
    HCTSA_TSTL_delaytime,
    HCTSA_TSTL_localdensity,
    HCTSA_TSTL_predict,
    HCTSA_WL_DetailCoeffs,
    HCTSA_WL_coeffs,
    HCTSA_WL_cwt,
    HCTSA_WL_dwtcoeff,
    HCTSA_WL_fBM,
    HCTSA_WL_scal2frq,
)

ALL_HCTSA_CLASSES = (
    CO_AddNoise,
    CO_AutoCorr,
    CO_AutoCorrShape,
    CO_CompareMinAMI,
    CO_Embed2,
    CO_Embed2_AngleTau,
    CO_Embed2_Basic,
    CO_Embed2_Dist,
    CO_Embed2_Shapes,
    CO_FirstMin,
    CO_FirstZero,
    CO_HistogramAMI,
    CO_NonlinearAutocorr,
    CO_RM_AMInformation,
    CO_StickAngles,
    CO_TSTL_AutoCorrMethod,
    CO_TSTL_amutual,
    CO_TSTL_amutual2,
    CO_TranslateShape,
    CO_f1ecac,
    CO_fzcglscf,
    CO_glscf,
    CO_tc3,
    CO_trev,
    CP_ML_StepDetect,
    CP_l1pwc_sweep_lambda,
    CP_wavelet_varchg,
    DN_Burstiness,
    DN_CompareKSFit,
    DN_Compare_zscore,
    DN_Cumulants,
    DN_CustomSkewness,
    DN_FitKernelSmooth,
    DN_Fit_mle,
    DN_HighLowMu,
    DN_HistogramMode,
    DN_Mean,
    DN_MinMax,
    DN_Moments,
    DN_OutlierInclude,
    DN_OutlierTest,
    DN_ProportionValues,
    DN_Quantile,
    DN_RemovePoints,
    DN_SimpleFit,
    DN_Spread,
    DN_TrimmedMean,
    DN_Withinp,
    DN_cv,
    DN_nlogL_norm,
    DN_pleft,
    DT_IsSeasonal,
    EN_ApEn,
    EN_DistributionEntropy,
    EN_MS_shannon,
    EN_PermEn,
    EN_RM_entropy,
    EN_Randomize,
    EN_SampEn,
    EN_Shannonpdf,
    EN_TSentropy,
    EN_wentropy,
    EX_MovingThreshold,
    FC_LocalSimple,
    FC_LoopLocalSimple,
    FC_Surprise,
    HT_DistributionTest,
    HT_HypothesisTest,
    MD_hrv_classic,
    MD_pNN,
    MD_polvar,
    MD_rawHRVmeas,
    MF_ARMA_orders,
    MF_AR_arcov,
    MF_CompareAR,
    MF_CompareTestSets,
    MF_ExpSmoothing,
    MF_FitSubsegments,
    MF_GARCHcompare,
    MF_GARCHfit,
    MF_GP_FitAcross,
    MF_GP_LearnHyperp,
    MF_GP_LocalPrediction,
    MF_GP_hyperparameters,
    MF_ResidualAnalysis,
    MF_StateSpaceCompOrder,
    MF_StateSpace_n4sid,
    MF_arfit,
    MF_armax,
    MF_hmm_CompareNStates,
    MF_hmm_fit,
    MF_steps_ahead,
    NL_BoxCorrDim,
    NL_CaosMethod,
    NL_MS_LZcomplexity,
    NL_MS_fnn,
    NL_MS_nlpe,
    NL_TISEAN_c1,
    NL_TISEAN_d2,
    NL_TSTL_FractalDimensions,
    NL_TSTL_GPCorrSum,
    NL_TSTL_LargestLyap,
    NL_TSTL_PoincareSection,
    NL_TSTL_ReturnTime,
    NL_TSTL_TakensEstimator,
    NL_TSTL_acp,
    NL_TSTL_dimensions,
    NL_crptool_fnn,
    NL_embed_PCA,
    NW_VisibilityGraph,
    PD_PeriodicityWang,
    PH_ForcePotential,
    PH_Walker,
    PP_Compare,
    PP_Iterate,
    PP_ModelFit,
    PP_PreProcess,
    RN_Gaussian,
    SB_BinaryStats,
    SB_BinaryStretch,
    SB_CoarseGrain,
    SB_MotifThree,
    SB_MotifTwo,
    SB_TransitionMatrix,
    SB_TransitionpAlphabet,
    SC_FluctAnal,
    SC_HurstExponent,
    SC_fastdfa,
    SD_MakeSurrogates,
    SD_SurrogateTest,
    SD_TSTL_surrogates,
    SP_Summaries,
    ST_FitPolynomial,
    ST_Length,
    ST_LocalExtrema,
    ST_MomentCorr,
    ST_SimpleStats,
    SY_DriftingMean,
    SY_DynWin,
    SY_KPSStest,
    SY_LinearTrend,
    SY_LocalDistributions,
    SY_LocalGlobal,
    SY_PPtest,
    SY_RangeEvolve,
    SY_SlidingWindow,
    SY_SpreadRandomLocal,
    SY_StatAv,
    SY_StdNthDer,
    SY_StdNthDerChange,
    SY_TISEAN_nstat_z,
    SY_VarRatioTest,
    TSTL_delaytime,
    TSTL_localdensity,
    TSTL_predict,
    WL_DetailCoeffs,
    WL_coeffs,
    WL_cwt,
    WL_dwtcoeff,
    WL_fBM,
    WL_scal2frq,
)