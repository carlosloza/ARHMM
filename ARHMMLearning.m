function ARHMModel = ARHMMLearning(x, K, varargin)
% Implementation of Expectation-Maximization (EM) algorithm for maximum 
% likelihood estimation of Autoregressive Hidden Markov Models (ARHMM)
% given univariate observations
% Forward-backwards implementation
% Author: Carlos Loza
% 
% Parameters
% -----------
% x :                   vector, size (N, 1) or (1, N)
%                       Observations from ARHMM
% K :                   int
%                       Number of components, states, modes of hidden
%                       variable, z
% p :                   int
%                       Order of autoregressive model
%                       Optional only if initial conditions of ARcoef 
%                       is provided
% ARcoef :              matrix, size (p, K)
%                       Initial conditions for autoregressive coefficients
% sig :                 vector, size (1, K)
%                       Initial conditions for observation noise standard 
%                       deviations
% pi :                  vector, size (K, 1)
%                       Initial conditions for initial state probabilities
% A :                   matrix, size (K, K)
%                       Initial conditions for state transition matrix
%
% Returns
% -------
% ARHMModel :           structure array. Fields:
% a :                   matrix, size (p, K) 
%                       Autoregressive coefficients
% sig :                 vector, size (1, K)
%                       Observation noise standard deviations
% pi :                  vector, size (K, 1)    
%                       Initial state probabilities
% A :                   matrix, size (K, K)
%                       State transition matrix
% LogLikelihood :       vector, size (1, nIt)
%                       Loglikelihoods given observations over iterations
%                       nIt is the number of iterations
%                       until convergence
%
% Notes
% -----
% Forward-backwards implementation with scaled alphas (so they are a proper 
% normalized distribution). The scaling weights require to have two
% non-nested for loops (one for the forward direction, one for the backwards
% direction). An alternative would be to have only one for loop where both
% directions are run per iteration and the log-sum-exp trick is used.
% However, underflow would still be possible for very long sequences.
%
% If no initial conditions are provided, the following steps takes place: 
% 1. A static AR model is fitted over an initial portion of the observation 
% vector to estimate an initial value of the observation noise standard 
% deviation. 
% 2. A Kalman filter AR provides initial estimates of AR coefficients and 
% initial state probabilities (after prunning low evidence sets of coefficients)
% 3. If no initial state transition matrix, A, is provided, the transitions
% are set favoring self-transitions (probability = 0.8). All remaining
% probabilities are assigned in a random fashion with the constraint of
% always "going" to a state, i.e. rows sum up to 1.
%
% Examples : 
% ARHMModel = HMMARLearning(x, K, 'p', 10);
% ARHMModel = HMMARLearning(x, K, 'p', 10, 'A', [0.9, 0.1; 0.2, 0.8]);
% ARHMModel = HMMARLearning(x, K, 'p', 10, 'A', [0.9, 0.1; 0.2, 0.8], 'pi', [0.25; 0.75]);

%% Check inputs and initial conditions
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'p')
        p = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'ARcoef')
        a = varargin{i + 1};
        if exist('p', 'var')
            if p ~= size(a, 1)
                fprintf('Error: Order provided must match order in initial conditions \n')
            else
                p = size(a, 1);
            end
        end
    elseif strcmpi(varargin{i}, 'sig')
        sig = varargin{i + 1};
        if numel(sig) == 1
            sig = sig*ones(1, K);
        else
            if numel(sig) ~= K
                fprintf('Error: Number of states in variable sig does not match the value in K \n')
            end            
        end
    elseif strcmpi(varargin{i}, 'pi')
        pi_ini = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'A')
        A = varargin{i + 1};   
    end
        
end
if ~exist('p', 'var')
    fprintf('AR model order is required \n')
end

% Observations must be row vectors
if iscolumn(x)
    x = x';
end
x = zscore(x);                          % Normalize observations
N = numel(x);
dmin = 2e-300;                          % To avoid underflow
maxIt = 20;                             % Maximum number of EM iterations
convTh = 0.01;                          % Convergence threshold

% Build auxiliary matrix of AR predictors
Xp = zeros(N, p);
for i = p + 1:N
    Xp(i, :) = -fliplr(x(i - p:i - 1));
end

% If no observation noise parameter is provided, fit an AR model of order p 
% to an initial portion of the input
if ~exist('sig', 'var')
    Xpini = Xp(p+1:5*p, :);
    Yini = x(p+1:5*p)';
    aini = (Xpini'*Xpini)\(Xpini'*Yini);
    sig = std(Xpini*aini - Yini);
    sig = sig*ones(1, K);
end

% Kalman filter AR initialization
if ~exist('a', 'var')
    if exist('sig', 'var')
        sig2 = sig(1)^2;
    end
    % Tracking parameter alph, a small value tracks the modes in the observations
    % in a better way, a larger value smooths the states stimates, but
    % compromises tracking
    alph = 0.1;             
    evTh = 0.5;             % Threshold for low evidence observations
    KalmanARModel = KalmanAR(x, p, 'alph', alph, 'sig2', sig2);
    a = KalmanARModel.a;
    evX = KalmanARModel.evX;
    % Cluster AR coefficients without low evidence
    idxTh = evX > evTh;
    a = a(:, idxTh);
    % k-means may be utilized too, but GMM is more general
    GMModel = fitgmdist(a', K, 'Replicates', 10, 'RegularizationValue', 0.1, 'CovarianceType', 'diagonal');
    a = GMModel.mu';
    if ~exist('pi_ini', 'var')
        pi_ini = GMModel.ComponentProportion';
    end
end

% Initial state probabilities and state transition matrix initializations
if ~exist('pi_ini', 'var')
    aux = gamrnd(1, 1, K, 1);           % Sample from dirichlet distribution
    pi_ini = aux./sum(aux);
end
if ~exist('A', 'var')
    A = 0.8*eye(K);                     % Probability of self-transitions
    for i = 1:K
        aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
        aux = (1 - A(i,i))*aux./sum(aux);
        A(i, 1:K ~= i) = aux;
    end
end

%% Expectation-Maximization (EM)
fl = 1;                                 % Convergence flag
ct = 0;
LogLike = [];
ufaux = dmin*ones(1, K);
iforw = p + 1:N;
iback = N:-1:p + 1;
while fl
    % E STEP
    % Forward and backward algorithm
    c = ones(1, N);
    alphaforw = ones(K, N);    
    pemaux1 = (1./(sqrt(2*pi).*sig));
    pemaux2 = -(1./(2.*sig.^2));
    % ALPHAS
    % First iteration
    i = 1;
    Fnforw = Xp(iforw(i), :);
    pemk = pemaux1.*exp(pemaux2.*(x(iforw(i)) - Fnforw*a).^2);
    pemkuf = [pemk' dmin*ones(K, 1)];
    aux = pi_ini.*max(pemkuf,[], 2);
    % Scaled versions - to avoid underflow
    c(iforw(i)) = 1/sum(aux);
    alphaforw(:, iforw(i)) = c(iforw(i))*aux;
    % Rest of iterations
    for i = 2:numel(iforw)        
        % Alpha
        Fnforw = Xp(iforw(i), :);
        pemk = pemaux1.*exp(pemaux2.*(x(iforw(i)) - Fnforw*a).^2);
        pemkuf = [pemk; ufaux];
        aux = (max(pemkuf,[], 1)').*(A' * alphaforw(:, iforw(i) - 1));
        % Scaled versions
        c(iforw(i)) = 1/sum(aux);
        alphaforw(:, iforw(i)) = c(iforw(i))*aux;
    end    
    % Likelihood
    LogLike = [LogLike -sum(log(c))];
    
    % BETAS, GAMMAS and XIS
    betaback = ones(K, N);
    xi = zeros(K, K, N);
    % First iteration
    i = 1;
    betaback(:, iback(i)) = c(iback(i))*1;
    % Rest of iterations
    for i = 2:numel(iback)
        % Betas
        Fnback = Xp(iback(i) + 1, :);        
        pemk = pemaux1.*exp(pemaux2.*(x(iback(i) + 1) - Fnback*a).^2);
        pemkuf = [pemk; ufaux];
        aux = bsxfun(@times, max(pemkuf, [], 1), A) * betaback(:, iback(i) + 1);
        betaback(:, iback(i)) = c(iback(i))*aux;          
        % Xis
        Fn = Xp(iback(i), :);
        pemk = pemaux1.*exp(pemaux2.*(x(iback(i)) - Fn*a).^2);
        pemkuf = [pemk; ufaux];        
        aux = ((betaback(:, iback(i))*alphaforw(:, iback(i) - 1)')' .* bsxfun(@times, max(pemkuf, [], 1), A));
        xi(:, :, iback(i)) = aux;       
    end
    % Gammas
    gamaux = alphaforw.*betaback;
    gam = bsxfun(@rdivide, gamaux, sum(gamaux, 1));
    gam(:, 1:p) = 0;
 
    % M STEP
    % Initial state probabilities
    pi_ini = gam(:,p + 1)./sum(gam(:,p + 1));
    % State transition probabilities, AR coefficients and observation noise 
    % variances per state/mode
    A = zeros(K, K);
    for k = 1:K
        % Transition probabilities
        aux = squeeze(xi(k, :, p + 2:end));
        A(k, :) = sum(aux,2)'/sum(aux(:));        
        % AR coefficients
        Ckv = sqrt(gam(k, p + 1: end));
        Xtil = bsxfun(@times, Ckv', Xp(p+1:end, :));
        Ytil = (Ckv.*x(p + 1:end))';        
        a(:, k) = (Xtil'*Xtil)\(Xtil'*Ytil);
        
        %ak(:, k) = robustfit(Xtil, Ytil, 'huber', [], 'off');        
        %[B, FitInfo] = lasso(Xtil, Ytil, 'CV', 10, 'RelTol', 1e-2);
        %ak(:, k) = B(:, FitInfo.IndexMinMSE);
        
        % Noise variance
        err = (Xp(p + 1: end, :)*a(:, k))' - x(p + 1: end);
        aux = (err.^2)*gam(k, p + 1 : end)';
        sig(k) = sqrt(aux/sum(gam(k,:)));
    end   
    ct = ct + 1;  
    % Check for convergence
    if ct > 1
        if abs(LogLike(ct) - LogLike(ct-1))/abs(LogLike(ct - 1)) <= convTh
            ARHMModel.pi = pi_ini;
            ARHMModel.A = A;
            ARHMModel.a = a;
            ARHMModel.sig = sig;
            ARHMModel.gam = gam;
            ARHMModel.loglike = LogLike;
            break       
        end
    end
    
    if ct == maxIt
        ARHMModel.pi = pi_ini;
        ARHMModel.A = A;
        ARHMModel.a = a;
        ARHMModel.sig = sig;
        ARHMModel.gam = gam;
        ARHMModel.loglike = LogLike;
        fprintf('Solution did not converge after %u iterations. \n', maxIt)
        break       
    end  
end
end