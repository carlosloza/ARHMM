clearvars
close all
clc

kCV = 5;
warning('off')
% Load data
load('DREAMS/Subject6.mat')

p_v = round((Fs/100)*[3:2:20]);
y = zscore(X);
[PowRatio, LogLike] = HMMARPowRatio(y, Fs, p_v);