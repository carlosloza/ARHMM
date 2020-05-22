function [zopt, loglike] = HMMARViterbiGroundTruth(y, p, v_sc, Fs)

ak = zeros(p, 2);           % 2 modes
sigk = zeros(1, 2);

%% Emission parameters
% Not spindles
X = zeros(numel(y), p);
Y = zeros(numel(y), 1);
yback = zeros(size(y));
for i = 1:size(v_sc, 1)
    yback(round(Fs*v_sc(i, 1)):round(Fs*v_sc(i, 1)) + round(Fs*v_sc(i, 2))) = 1;
end
idx = [find(diff(yback) ~= 0) numel(y)];
xaux = y(1:idx(1));
ct = 1;
for j = 1:length(xaux)-p
    X(ct, :) = -fliplr(xaux(j:j+p-1));
    Y(ct) = xaux(j+p);
    ct = ct + 1;
end
for i = 2:2:numel(idx) - 1
    xaux = y(idx(i) + 1:idx(i+1));
    for j = 1:length(xaux)-p
        X(ct, :) = -fliplr(xaux(j:j+p-1));
        Y(ct) = xaux(j+p);
        ct = ct + 1;
    end
end
X = X(1:ct - 1, :);
Y = Y(1:ct -1);
ak(:, 1) = robustfit(X, Y, 'ols', [], 'off');
sigk(1) = std(Y - X*ak(:, 1));

% Spindles
X = zeros(numel(y), p);
Y = zeros(numel(y), 1);
ct = 1;
for i = 1:size(v_sc, 1)
    xaux = y(round(Fs*v_sc(i, 1)):round(Fs*v_sc(i, 1)) + round(Fs*v_sc(i, 2)));
    for j = 1:length(xaux)-p
        X(ct, :) = -fliplr(xaux(j:j+p-1));
        Y(ct) = xaux(j+p);
        ct = ct + 1;
    end
end
X = X(1:ct - 1, :);
Y = Y(1:ct -1);
ak(:, 2) = robustfit(X, Y, 'ols', [], 'off');
sigk(2) = std(Y - X*ak(:, 2));

%% Hidden state probabilities
pik = [1; 0];           % Always starts with not spindle
A = zeros(2, 2);
for i = 1:numel(y) - 1
    if yback(i) == 0 && yback(i + 1) == 0
        A(1, 1) = A(1, 1) + 1;
    elseif yback(i) == 0 && yback(i + 1) == 1
        A(1, 2) = A(1, 2) + 1;
    elseif yback(i) == 1 && yback(i + 1) == 0
        A(2, 1) = A(2, 1) + 1;
    elseif yback(i) == 1 && yback(i + 1) == 1
        A(2, 2) = A(2, 2) + 1;
    end
end

for i = 1:2
    A(i, :) = A(i, :)./sum( A(i, :));
end

HMMAR.pi = pik;
HMMAR.A = A;
HMMAR.ak = ak;
HMMAR.sig = sigk;

%% Viterbi
[zopt, loglike] = HMMARViterbi(y, HMMAR);
end