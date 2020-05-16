%% 
% Check whether power in different bands (rhythms) is statistically 
% significant between modes (2 components/modes) for different AR model orders
clearvars
close all
clc

p_v = [2 5 10 15 20 25 30];
p_v = 2;
alph = 0.1;
evTh = 0.5;            % Evidence threshold
sig2n = 0.2;
ncomp = 2;

A = [0.98 0.02;
    0.02 0.98];

% Load data
load('DREAMS/Subject6.mat')
y = X;
y = zscore(y);
figure, plot(y)

%%
for i = 1:length(p_v)
    fprintf('Order p = %d \n', p_v(i))
    disp('Kalman Filter')
    [theta, ev_y] = KalmanAR(y, p_v(i), alph, sig2n);
    disp('HMMAR')
    sigk = sqrt(sig2n)*ones(1, ncomp);
    HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk);
    gamp = HMMAR.gam(:, p_v(i)+1:end);
    yp = y(p_v(i)+1:end);
    sd = 1;
end