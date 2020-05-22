%% Estimation of AR parameters of a non-stationary time series using the 
% Kalman filter
% Then, cluster AR coefficients without low evidence
% Then, EM algorithm (1 iteration)

clearvars
close all
clc

p = 6;
alph = 0.1;
evTh = 0.5;            % Evidence threshold
ncomp = 3;

rng(34)

% Generate synthetic data
Fs = 100;
y = zeros(1, 3*Fs);
t = 1/Fs:1/Fs:3;
for i = 1:3
    t_aux = t(Fs*(i - 1) + 1:Fs*i);
    y(Fs*(i - 1) + 1:Fs*i) = sin(2*pi*10*i*t_aux);
end
% Add noise (0.1 standard deviation)
y = y + 0.1*randn(size(y));
%y = y + 0.01*randn(size(y));

%% Kalman Filter
disp('Kalman Filter')
sig2n = 0.2;
[theta, ev_y] = KalmanAR(y, p, alph, sig2n);

%% HMM AR
% Initial transitions
A = [0.98 0.01 0.01;
    0.01 0.98 0.01;
    0.01 0.01 0.98];
A_ini = A;
sigk = sqrt(sig2n)*ones(1, ncomp);
tic
HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk);
toc
plotHMMAR(HMMAR, Fs)
figure, plot(HMMAR.loglike)
title(HMMAR.loglike(end))

[zopt, loglike] = HMMARViterbi(y, HMMAR);
