%% Estimation of AR parameters of a non-stationary time series using the 
% Kalman filter
% Then, cluster AR coefficients without low evidence
% Then, EM algorithm (1 iteration)

clearvars
close all
clc

p = 8;
alph = 0.1;
evTh = 0.5;            % Evidence threshold
ncomp = 3;

rng(34)

% Generate synthetic data
Fs = 100;
y = zeros(1, 3*Fs);
t = 1/Fs:1/Fs:3;
for i = 1:3
    t_aux = t(Fs*(i - 1) + 1:Fs*i);
    y(Fs*(i - 1) + 1:Fs*i) = sin(2*pi*10*i*t_aux);
end
% Add noise (0.1 standard deviation)
y = y + 0.1*randn(size(y));
%% Initial SVD solution
[U,S,V] = svd(fliplr(y(1:p)));
s = diag(S);
d = U'*y(p+1);
b = V(:,1)*(d./S(1));

%% Kalman Filter
disp('Kalman Filter')
sig2_t = 0.2;
theta = zeros(p, numel(y));
q = zeros(1, numel(y));
theta(:, p) = b;
Ft = -fliplr(y(1:p));
SigCovtprev = sig2_t*(Ft*Ft');
e = zeros(1, numel(y));
sig2_q0 = zeros(1, numel(y));
sig2_yt = zeros(1, numel(y));
K = zeros(p, numel(y));
Rt = zeros(p, p, numel(y));
LRate = zeros(1, numel(y));
pred_y = zeros(1, numel(y));
ev_y = zeros(1, numel(y));
for i = p + 1:numel(y)
    %
    Ft = -fliplr(y(i-p:i-1));
    e(i) = y(i) - Ft*theta(:, i-1);
    sig2_q0(i) = sig2_t + Ft*SigCovtprev*Ft';
    h_arg = (e(i)^2 - sig2_q0(i))/(Ft*Ft');
    if h_arg >= 0
        h = h_arg;
    else
        h = 0;
    end
    q(i) = alph*q(i-1) + (1-alph)*h;
    
    if numel(SigCovtprev) == 1
        Rt(:,:,i) = SigCovtprev*eye(p) + q(i)*eye(p);
    else
        Rt(:,:,i) = SigCovtprev + q(i)*eye(p);
    end
    
    sig2_theta = Ft*Rt(:,:,i)*Ft';
    sig2_yt(i) = sig2_t + sig2_theta;
    
    pred_y(i) = Ft*theta(:, i-1);
    ev_y(i) = (1/sqrt(2*pi*sig2_yt(i)))*(exp(-e(i)^2/(2*sig2_yt(i))));
    
    LRate(i) = (1/p)*(trace(Rt(:,:,i))/sig2_yt(i));
    
    K(:, i) = (Rt(:,:,i)*Ft')/sig2_yt(i);
    
    theta(:, i) = theta(:, i - 1) + K(:, i)*e(i);
    SigCovtprev = Rt(:,:,i) - K(:, i)*Ft*Rt(:,:,i) ;
end

%% Cluster AR coefficients without low evidence
idxTh = ev_y > evTh;
X = theta(:, idxTh);
GMModel = fitgmdist(X', 3);
idx = cluster(GMModel, theta');

%% E STEP. Forward part
disp('E step')
disp('Forward')
pik = GMModel.ComponentProportion';
pik_ini = pik;
ak = GMModel.mu';
ak_ini = ak;
alph = zeros(ncomp, length(y));
logalpha = zeros(ncomp, length(y));

% Initial transitions
A = [0.98 0.01 0.01;
    0.01 0.98 0.01;
    0.01 0.01 0.98];
A_ini = A;

% First iteration - initialization
i = p+1;
Ft = -fliplr(y(i-p:i-1));
for k = 1:ncomp
    logalpha(k, i) = log(pik(k)) + log(normpdf(y(i), Ft*ak(:,k), sig2_t));
end
ct = 0;
for i = p+2:numel(y)
    ct = ct + 1;
    Ft = -fliplr(y(i-p:i-1));
    for j = 1:ncomp
        a = zeros(1, ncomp);
        lpem = log(normpdf(y(i), Ft*ak(:,j), sig2_t));
        for k = 1:ncomp
            a(k) = lpem + logalpha(k, i-1) + log(A(k, j));
        end
        b = max(a);
        logalpha(j, i) = b + log(sum(exp(a - b)));
    end   
end
ct
%% Backward part
disp('Backward')
logbeta = zeros(ncomp, length(y));

% First iteration - initialization DONE - beta = 1
ct = 0;
for i = numel(y)-1:-1:p+1
    ct = ct + 1;
    Ftp1 = -fliplr(y(i-p+1:i));
    for j = 1:ncomp
        a = zeros(1, ncomp);
        for k = 1:ncomp
            a(k) = logbeta(k, i+1) + log(normpdf(y(i+1), Ftp1*ak(:,k), sig2_t)) + log(A(j, k));
        end
        b = max(a);
        logbeta(j, i) =  b + log(sum(exp(a - b)));
    end
end
ct
%% Conditional posterior of latent variable
disp('gamma')
pX = (sum(exp(logalpha(:,end))));       % Likelihood
gam = zeros(ncomp, length(y));
for i = p+1:numel(y)
    gam(:, i) = (exp(logalpha(:,i)).*exp(logbeta(:,i)))/pX;
end

%% Joint conditional posterior of two latent variables
disp('eta')
eta = zeros(ncomp, ncomp, length(y));
for i = p+2:length(y)
    Ft = -fliplr(y(i-p:i-1));
    for j = 1:ncomp
        for k = 1:ncomp
            eta(j,k, i) = exp(logalpha(j, i - 1))*normpdf(y(i), Ft*ak(:,k), sig2_t)*A(j,k)*exp(logbeta(k, i));
        end
    end
    asd = 1;
end
eta = eta/pX;
disp('E step done')

%% M STEP Initial latent variable probabilities
pik = gam(:,p+1)./sum(gam(:,p+1));

%% State transition probabilities
A = zeros(ncomp, ncomp);
for j = 1:ncomp
    denA = 0;
    for kk = 1:ncomp
        denA = denA + sum(eta(j,kk,p+2:end));
    end
    for k = 1:ncomp
        numA = sum(eta(j,k,p+2:end));
        A(j,k) = numA/denA;
    end
end

%% AR coefficients per mode
X = [];
for i = p + 1:length(y)
    Ft = -fliplr(y(i-p:i-1));
    X = [X; Ft];
end
for k = 1:ncomp
    Ck = diag(sqrt(gam(k, p+1:end)));
    Xtil = Ck*X;
    Ytil = Ck*y(p+1:end)';
    ak(:, k) = (Xtil'*Xtil)\(Xtil'*Ytil);
end

%% Noise variance for each mode
sigk = zeros(1, ncomp);
for k = 1:ncomp
    aux = zeros(1, length(y));
    for i = p+1:length(y)
        Ft = -fliplr(y(i-p:i-1));
        aux(i) = gam(k,i)*(y(i) - Ft*ak(:,k))^2;
    end
    sigk(k) = sum(aux)/sum(gam(k,:));
end
