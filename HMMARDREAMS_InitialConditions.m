%%
% Script that gets the initial conditions for EEGMDL from the DREAMS
% database
clearvars
close all
clc

subj = 1;

p_opt = [8, 40, 5, 0, 0, 30];
% Load data
load(['DREAMS/Subject' num2str(subj) '.mat'])
p = p_opt(subj);
y = X;

ndown = 2;
y = downsample(y, ndown);
Fs = Fs/ndown;

y = zscore(y);

alph = 0.1;
evTh = 0.5;            % Evidence threshold
ncomp = 2;
f = 0:0.5:Fs/2;
bandHz = [10.5 16];     % Band of interest

% Initial transitions
stP = 0.5;              % Order of stationarity (in seconds) for EEG
stPaux = 1 - 1/(stP*Fs);
A = [stPaux, 1 - stPaux;
    1- stPaux, stPaux];
sig2n = 0.01;            % Could be latter estimated via static AR model

% "Optimal" order
[theta, ev_y] = KalmanAR(y, p, alph, sig2n);
sigk = sqrt(sig2n)*ones(1, ncomp);
HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk);
pwBand = zeros(1, ncomp);
for k = 1:ncomp
    [H, f] = freqz(1, [1; HMMAR.ak(:,k)], f, Fs);
    pwBand(k) = sum(abs(H(f > bandHz(1) & f < bandHz(2))));
end
[~, maxBand] = max(pwBand);
z = HMMARViterbi(y, HMMAR);
ylabel = false(size(y));
ylabel(z == maxBand) = true(1);

%% Extract all potential spindles epochs
