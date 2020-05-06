function [theta, ev_y] = KalmanAR(y, p, alph, sig2_n)

% Initial SVD solution
[U,S,V] = svd(fliplr(y(1:p)));
d = U'*y(p+1);
b = V(:,1)*(d./S(1));

% Kalman Filter
theta = zeros(p, numel(y));
q = zeros(1, numel(y));
theta(:, p) = b;
Fn = -fliplr(y(1:p));
SigCovtprev = sig2_n*(Fn*Fn');
e = zeros(1, numel(y));
sig2_q0 = zeros(1, numel(y));
sig2_yt = zeros(1, numel(y));
K = zeros(p, numel(y));
Rn = zeros(p, p, numel(y));
LRate = zeros(1, numel(y));
pred_y = zeros(1, numel(y));
ev_y = zeros(1, numel(y));
for i = p + 1:numel(y)
    Fn = -fliplr(y(i-p:i-1));
    e(i) = y(i) - Fn*theta(:, i-1);
    sig2_q0(i) = sig2_n + Fn*SigCovtprev*Fn';
    h_arg = (e(i)^2 - sig2_q0(i))/(Fn*Fn');
    if h_arg >= 0
        h = h_arg;
    else
        h = 0;
    end
    q(i) = alph*q(i-1) + (1-alph)*h;    
    if numel(SigCovtprev) == 1
        Rn(:,:,i) = SigCovtprev*eye(p) + q(i)*eye(p);
    else
        Rn(:,:,i) = SigCovtprev + q(i)*eye(p);
    end    
    sig2_theta = Fn*Rn(:,:,i)*Fn';
    sig2_yt(i) = sig2_n + sig2_theta;    
    pred_y(i) = Fn*theta(:, i-1);
    ev_y(i) = (1/sqrt(2*pi*sig2_yt(i)))*(exp(-e(i)^2/(2*sig2_yt(i))));   
    LRate(i) = (1/p)*(trace(Rn(:,:,i))/sig2_yt(i));   
    K(:, i) = (Rn(:,:,i)*Fn')/sig2_yt(i);    
    theta(:, i) = theta(:, i - 1) + K(:, i)*e(i);
    SigCovtprev = Rn(:,:,i) - K(:, i)*Fn*Rn(:,:,i) ;
end

end