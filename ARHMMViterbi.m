function [z, LogLike] = ARHMMViterbi(x, ARHMModel)
% Implementation of the Viterbi algorithm for estimation of the optimal
% state sequence of Autoregressive Hidden Markov Models (ARHMM)
% Author: Carlos Loza
%
% Parameters
% ----------
% x :                   vector, size (N, 1) or (1, N)
%                       Observations from ARHMM
% ARHMModel :           structure array. Fields:
% a :                   matrix, size (p, K) 
%                       Autoregressive coefficients
% sig :                 vector, size (1, K)
%                       Observation noise standard deviations
% pi :                  vector, size (K, 1)    
%                       Initial state probabilities
% A :                   matrix, size (K, K)
%                       State transition matrix
%
% Returns
% -------
% z :                   vector, size (1, N)
%                       Optimal hidden state sequence
% LogLike :             Loglikelihood of observations given model
%                       parameters

%% Pre-processing
% Observations must be row vectors
if iscolumn(x)
    x = x';
end
x = zscore(x);                          % Normalize observations
N = numel(x);
dmin = 2e-300;                          % To avoid underflow
a = ARHMModel.a;
sig = ARHMModel.sig;
[p, K] = size(a);
z = zeros(1, N);
pi = ARHMModel.pi;
pi(pi < dmin) = dmin;
logpi = log(pi);
A = ARHMModel.A;
A(A < dmin) = dmin;         % Unless I strictly want some transitions to be impossible
logA = log(A);

%% Forward direction
bet = zeros(K, N);
delt = zeros(K, N);
psin = zeros(K, N);
% First iteration
i = p + 1;
Fn = -fliplr(x(i - p: i - 1));
for k = 1:K
    bet(k, i) = log(max([normpdf(x(i), Fn*a(:,k), sig(k)) , dmin]));
end
delt(:, i) = logpi + bet(:, i);
% Rest of iterations
for i = p + 2:N
    Fn = -fliplr(x(i - p: i - 1));
    for k = 1:K
        bet(k, i) = log(max([normpdf(x(i), Fn*a(:,k), sig(k)) , dmin]));
        [aux1, aux2] = max(delt(:, i-1) + logA(:, k));
        delt(k, i) = bet(k, i) + aux1;
        psin(k, i) = aux2;
    end
end
% Maximum probability and corresponding statfor last obervation
[LogLike, z(end)] = max(delt(:, end));

%% Backtracking
for i = N-1:-1:p+1
    z(i) = psin(z(i+1), i+1);
end