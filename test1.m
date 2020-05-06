%% Estimation of AR parameters of a non-stationary time series using the 
% Kalman filter
clearvars
close all
clc

p = 8;
alph = 0.1;
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

figure
plot(y)
%% Initial SVD solution
[U,S,V] = svd(fliplr(y(1:p)));
s = diag(S);
d = U'*y(p+1);
b = V(:,1)*(d./S(1));

%%
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
    i
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

%%
figure
plot(log10(ev_y))
xlabel('t')
ylabel('Log-evidence')

%%
figure
subplot(2,2,1)
plot(t, e.^2)
xlabel('t')
ylabel('Squared prediction error (e^2_t)')
subplot(2,2,2)
plot(t, sig2_q0)
xlabel('t')
ylabel('\sigma^2_{q0}')
subplot(2,2,3)
plot(t, sig2_yt)
xlabel('t')
ylabel('\sigma^2_{yt}')
subplot(2,2,4)
plot(t, q)
xlabel('t')
ylabel('q_t')

%%
priorVarx = zeros(1, numel(y));
for i = 1:numel(y)
    priorVarx(i) = trace(Rt(:,:,i))/p;
end
figure
plot(t, priorVarx);
xlabel('t')
ylabel('Average prior variance of states')

%%
figure
plot(t, LRate);
xlabel('t')
ylabel('Average learning rate')