%% 
clearvars
close all
clc

p = 8;
alph = 0.1;
evTh = 0.5;            % Evidence threshold
ncomp = 2;

% Load data
load('DREAMS/Subject1.mat')
%y = X(1:round(length(X)/20));

%SigmaIndexTh = 1;
%idxSpindles = SigmaIndex(X, Fs, SigmaIndexTh);

y = X;
%y = zscore(y);

%load(['DREAMS/Filters/Spindles_' num2str(Fs) 'Hz'])
%y = filtfilt(h_b, h_a, y);


y = zscore(y);

figure, plot(y)

%% Kalman Filter
disp('Kalman Filter')
sig2n = 0.2;
[theta, ev_y] = KalmanAR(y, p, alph, sig2n);

figure
subplot(2,1,1)
plot(ev_y)
subplot(2,1,2)
ksdensity(ev_y, linspace(0,1,100))

%% HMM AR
% Initial transitions
A = [0.98 0.02;
    0.02 0.98];
A_ini = A;
sigk = sqrt(sig2n)*ones(1, ncomp);
tic
HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk);
toc
%%
plotHMMAR(HMMAR, Fs)
%%
figure, plot(HMMAR.loglike)