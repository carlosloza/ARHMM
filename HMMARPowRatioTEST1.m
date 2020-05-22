clearvars
close all
clc

warning('off')
% Load data
load('DREAMS/Subject4.mat')

p_v = round((Fs/100)*[2:2:30]);
y = zscore(X);
[PowRatio, LogLike, akcell] = HMMARPowRatio(y, Fs, p_v);