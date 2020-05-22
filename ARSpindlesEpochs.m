%% 
% Extract Spindles epochs (marked by experts), inspect regressors (i.e. past 
% p samples) and estimate AR coefficients by OLS
clearvars 
close all
clc

p = 15;
subj = 1;
load(['DREAMS/Subject' num2str(subj) '.mat']);
y = X;


% ndown = 1;
% y = downsample(y, ndown);
% Fs = Fs/ndown;

load(['DREAMS/Filters/Spindles_' num2str(Fs) 'Hz'])
y = filtfilt(h_b, h_a, y);

y = zscore(y);
[zopt, loglike] = HMMARViterbiGroundTruth(y, p, v_sc1, Fs)

%% Expert 1 - Spindles
X = [];
Y = [];
for i = 1:size(v_sc1, 1)
    xaux = y(round(Fs*v_sc1(i, 1)):round(Fs*v_sc1(i, 1)) + round(Fs*v_sc1(i, 2)));
    %xaux = zscore(xaux);
    for j = 1:length(xaux)-p
        X = [X; -fliplr(xaux(j:j+p-1))];
        Y = [Y; xaux(j+p)];
    end
    as = 1;
end
a = robustfit(X, Y, 'ols', [], 'off');
[H, F] = freqz(1, [1; a], 1024, Fs);
figure
subplot(2,2,1)
plot(F, 20*log10(abs(H)))

%% Expert 1 - Not Spindles
X = zeros(numel(y), p);
Y = zeros(numel(y), 1);
yback = zeros(size(y));
for i = 1:size(v_sc1, 1)
    yback(round(Fs*v_sc1(i, 1)):round(Fs*v_sc1(i, 1)) + round(Fs*v_sc1(i, 2))) = 1;
end
idx = [find(diff(yback) ~= 0) numel(y)];
xaux = y(1:idx(1));
%xaux = zscore(xaux);
ct = 1;
for j = 1:length(xaux)-p
    %X = [X; -fliplr(xaux(j:j+p-1))];
    X(ct, :) = -fliplr(xaux(j:j+p-1));
    Y(ct) = xaux(j+p);
    ct = ct + 1;
end
for i = 2:2:numel(idx) - 1
    xaux = y(idx(i) + 1:idx(i+1));
    %xaux = zscore(xaux);
    for j = 1:length(xaux)-p
        %X = [X; -fliplr(xaux(j:j+p-1))];
        X(ct, :) = -fliplr(xaux(j:j+p-1));
        Y(ct) = xaux(j+p);
        ct = ct + 1;
    end
end
X = X(1:ct - 1, :);
Y = Y(1:ct -1);
a = robustfit(X, Y, 'ols', [], 'off');
[H, F] = freqz(1, [1; a], 1024, Fs);
subplot(2,2,2)
plot(F, 20*log10(abs(H)))

%% Expert 2
X = [];
Y = [];
for i = 1:size(v_sc2, 1)
    xaux = y(round(Fs*v_sc2(i, 1)):round(Fs*v_sc2(i, 1)) + round(Fs*v_sc2(i, 2)));
    %xaux = zscore(xaux);
    for j = 1:length(xaux)-p
        X = [X; -fliplr(xaux(j:j+p-1))];
        Y = [Y; xaux(j+p)];
    end
    as = 1;
end
a = robustfit(X, Y, 'ols', [], 'off');
[H, F] = freqz(1, [1; a], 1024, Fs);
subplot(2,2,3)
plot(F, 20*log10(abs(H)))

%% Expert 2 - Not Spindles
X = zeros(numel(y), p);
Y = zeros(numel(y), 1);
yback = zeros(size(y));
for i = 1:size(v_sc2, 1)
    yback(round(Fs*v_sc2(i, 1)):round(Fs*v_sc2(i, 1)) + round(Fs*v_sc2(i, 2))) = 1;
end
idx = [find(diff(yback) ~= 0) numel(y)];
xaux = y(1:idx(1));
%xaux = zscore(xaux);
ct = 1;
for j = 1:length(xaux)-p
    %X = [X; -fliplr(xaux(j:j+p-1))];
    X(ct, :) = -fliplr(xaux(j:j+p-1));
    Y(ct) = xaux(j+p);
    ct = ct + 1;
end
for i = 2:2:numel(idx) - 1
    xaux = y(idx(i) + 1:idx(i+1));
    %xaux = zscore(xaux);
    for j = 1:length(xaux)-p
        %X = [X; -fliplr(xaux(j:j+p-1))];
        X(ct, :) = -fliplr(xaux(j:j+p-1));
        Y(ct) = xaux(j+p);
        ct = ct + 1;
    end
end
X = X(1:ct - 1, :);
Y = Y(1:ct -1);
a = robustfit(X, Y, 'ols', [], 'off');
[H, F] = freqz(1, [1; a], 1024, Fs);
subplot(2,2,4)
plot(F, 20*log10(abs(H)))
