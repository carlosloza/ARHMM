%% Estimation of AR parameters of a non-stationary time series using the 
% EM
% Same as test3 but there is random noise in between modes

clearvars
close all
clc

p = 20;
alph = 0.1;
evTh = 0.5;            % Evidence threshold
ncomp = 3;

rng(34)

% Generate synthetic data
Fs = 100;
y = zeros(1, ncomp*Fs);
t = 1/Fs:1/Fs:3;
for i = 1:ncomp
    t_aux = t(Fs*(i - 1) + 1:Fs*i);
    y(Fs*(i - 1) + 1:Fs*i) = sin(2*pi*10*i*t_aux);
end
% Add noise in between
yext = [];
for i = 1:ncomp
    yext = [yext y(Fs*(i - 1) + 1:Fs*i) 0.3*randn(1, Fs)];
end
y = yext;
% Add noise (0.1 standard deviation)
y = y + 0.1*randn(size(y));
ncomp = ncomp + 1;

%% Kalman Filter
disp('Kalman Filter')
sig2n = 0.2;
[theta, ev_y] = KalmanAR(y, p, alph, sig2n);

%% HMM AR
% Initial transitions
A = [0.97 0.01 0.01 0.01;
    0.01 0.97 0.01 0.01;
    0.01 0.01 0.97 0.01;
    0.01 0.01 0.01 0.97];
A_ini = A;
sigk = sqrt(sig2n)*ones(1, ncomp);

HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk);

plotHMMAR(HMMAR, Fs)
