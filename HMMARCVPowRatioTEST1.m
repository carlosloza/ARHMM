clearvars
close all
clc

kCV = 5;
warning('off')
% Load data
load('DREAMS/Subject6.mat')

p_v = round((Fs/100)*[3:20]);

[logp, loglikeTest] = HMMARCVPowRatio(X, Fs, p_v, kCV);