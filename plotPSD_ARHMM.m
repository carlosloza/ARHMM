function plotPSD_ARHMM(ARHMModel, Fs)
% Plot Power Spectral Density (PSD) of estimated autoregressive models from
% an Autoregressive Hidden Markov Models (ARHMM)
% Author: Carlos Loza
%
% Parameters
% ----------
% ARHMModel :           structure array. Fields:
% a :                   matrix, size (p, K) 
%                       Autoregressive coefficients
% sig :                 vector, size (1, K)
%                       Observation noise standard deviations
% pi :                  vector, size (K, 1)    
%                       Initial state probabilities
% A :                   matrix, size (K, K)
%                       State transition matrix
% Fs :                  Sampling frequency in Hz
%
% No Returns, only plots

figure
K = size(ARHMModel.a, 2);
for k = 1:K
    [H, F] = freqz(1, [1; ARHMModel.a(:,k)], 1024, Fs);
    subplot(K, 1, k)
    plot(F, 20*log10(abs(H)))
    xlabel('Hz')
    ylabel(['PSD, State ' num2str(k)])
end
end