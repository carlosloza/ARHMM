function [zopt, loglike] = HMMARViterbi(y, HMMAR)

dmin = 2e-300;          % To avoid underflow
ak = HMMAR.ak;
sigk = HMMAR.sig;           % Standard deviation
[p, ncomp] = size(ak);
zopt = zeros(1, length(y));

% Preprocessing
pi = HMMAR.pi;
pi(pi < dmin) = dmin;
logpi = log(pi);
A = HMMAR.A;
A(A < dmin) = dmin;         % Unless I strictly want some transitions to be impossible
logA = log(A);

bet = zeros(ncomp, length(y));
delt = zeros(ncomp, length(y));
psin = zeros(ncomp, length(y));
% For n = 1
i = p + 1;
Fn = -fliplr(y(i - p: i - 1));
for k = 1:ncomp
    bet(k, i) = log(max([normpdf(y(i), Fn*ak(:,k), sigk(k)) , dmin]));
end
delt(:, i) = logpi + bet(:, i);
psin(:, i) = zeros(ncomp, 1);           % Redundant

for i = p + 2:length(y)
    Fn = -fliplr(y(i - p: i - 1));
    for k = 1:ncomp
        bet(k, i) = log(max([normpdf(y(i), Fn*ak(:,k), sigk(k)) , dmin]));
        [aux1, aux2] = max(delt(:, i-1) + logA(:, k));
        delt(k, i) = bet(k, i) + aux1;
        psin(k, i) = aux2;
    end
end
% Maximum probability and argument at last time stamp
[loglike, zopt(end)] = max(delt(:, end));

% Backtracking
for i = length(y)-1:-1:p+1
    zopt(i) = psin(zopt(i+1), i+1);
end