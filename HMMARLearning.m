function HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk)
% Implementation with scaled alphas and betas
dmin = 2e-300;          % To avoid underflow
n_it = 20;
th = 0.01;
p = size(theta, 1);
X = zeros(length(y), p);
for i = p + 1:length(y)
    X(i, :) = -fliplr(y(i - p:i - 1));
end
% Cluster AR coefficients without low evidence
idxTh = ev_y > evTh;
thetaev = theta(:, idxTh);
GMModel = fitgmdist(thetaev', ncomp, 'Replicates', 10, 'RegularizationValue', 0.1, 'CovarianceType', 'diagonal');
pik = GMModel.ComponentProportion';
pik_ini = pik;
ak = GMModel.mu';
ak_ini = ak;
fl = 1;
ct = 0;
loglike = [];
ufaux = dmin*ones(1, ncomp);
iforw = p + 1:numel(y);
iback = numel(y):-1:p+1;
while fl
    % E STEP
    % Forward and backward algorithm
    c = ones(1, numel(y));
    alphaforw = ones(ncomp, length(y));    
    pemaux1 = (1./(sqrt(2*pi).*sigk));
    pemaux2 = -(1./(2.*sigk.^2));
    % ALPHA
    % First iteration
    i = 1;
    Fnforw = X(iforw(i), :);
    pemk = pemaux1.*exp(pemaux2.*(y(iforw(i)) - Fnforw*ak).^2);
    pemkuf = [pemk' dmin*ones(ncomp, 1)];
    aux = pik.*max(pemkuf,[], 2);
    % Scaled versions
    c(iforw(i)) = 1/sum(aux);
    alphaforw(:, iforw(i)) = c(iforw(i))*aux;
    % Rest of iterations
    for i = 2:numel(iforw)        
        % Alpha
        Fnforw = X(iforw(i), :);
        pemk = pemaux1.*exp(pemaux2.*(y(iforw(i)) - Fnforw*ak).^2);
        pemkuf = [pemk; ufaux];
        aux = (max(pemkuf,[], 1)').*(A' * alphaforw(:, iforw(i) - 1));
        % Scaled versions
        c(iforw(i)) = 1/sum(aux);
        alphaforw(:, iforw(i)) = c(iforw(i))*aux;
    end    
    % Likelihood
    loglike = [loglike -sum(log(c))];
    % Beta, Gamma and Xi
    betaback = ones(ncomp, length(y));
    xi = zeros(ncomp, ncomp, length(y));
    
    % BETA
    % First iteration
    i = 1;
    betaback(:, iback(i)) = c(iback(i))*1;
    % Rest of iterations
    for i = 2:numel(iback)
        Fnback = X(iback(i) + 1, :);        
        pemk = pemaux1.*exp(pemaux2.*(y(iback(i) + 1) - Fnback*ak).^2);
        pemkuf = [pemk; ufaux];
        aux = bsxfun(@times, max(pemkuf, [], 1), A) * betaback(:, iback(i) + 1);
        betaback(:, iback(i)) = c(iback(i))*aux;  
        
        % Xi
        Fn = X(iback(i), :);
        pemk = pemaux1.*exp(pemaux2.*(y(iback(i)) - Fn*ak).^2);
        pemkuf = [pemk; ufaux];        
        aux = ((betaback(:, iback(i))*alphaforw(:, iback(i) - 1)')' .* bsxfun(@times, max(pemkuf, [], 1), A));
        xi(:, :, iback(i)) = aux;       

%         for j = 1:ncomp
%             xi(j, :, iback(i)) = alphaforw(j, iback(i) - 1).*(max(pemkuf, [], 1).*A(j, :))'.*betaback(:, iback(i));
%         end
    end
    % Gamma
    gamaux = alphaforw.*betaback;
    gam = bsxfun(@rdivide, gamaux, sum(gamaux, 1));
    gam(:, 1:p) = 0;
 
    % M STEP
    % Initial latent variable probabilities
    pik = gam(:,p + 1)./sum(gam(:,p + 1));
    % State transition probabilities, AR coefficients and noise variance
    % per mode
    A = zeros(ncomp, ncomp);
    for k = 1:ncomp
        % Transition probabilities
        aux = squeeze(xi(k, :, p+2:end));
        A(k, :) = sum(aux,2)'/sum(aux(:));
        
        % AR coefficients
        Ckv = sqrt(gam(k, p + 1:end));
        Xtil = bsxfun(@times, Ckv', X(p+1:end, :));
        Ytil = (Ckv.*y(p + 1:end))';
        
        ak(:, k) = (Xtil'*Xtil)\(Xtil'*Ytil);
        
        %ak(:, k) = robustfit(Xtil, Ytil, 'huber', [], 'off');
        
        %[B, FitInfo] = lasso(Xtil, Ytil, 'CV', 10, 'RelTol', 1e-2);
        %ak(:, k) = B(:, FitInfo.IndexMinMSE);
        
        % Noise variance
        err = (X(p+1:end,:)*ak(:,k))' - y(p+1:end);
        aux = (err.^2)*gam(k, p+1:end)';
        sigk(k) = sqrt(aux/sum(gam(k,:)));
    end
    
    ct = ct + 1;    
    if ct > 1
        if abs(loglike(ct) - loglike(ct-1))/abs(loglike(ct - 1)) <= th
            HMMAR.pi = pik;
            HMMAR.A = A;
            HMMAR.ak = ak;
            HMMAR.sig = sigk;
            HMMAR.gam = gam;
            HMMAR.loglike = loglike;
            break       
        end
    end
    
    if ct == n_it
        HMMAR.pi = pik;
        HMMAR.A = A;
        HMMAR.ak = ak;
        HMMAR.sig = sigk;
        HMMAR.gam = gam;
        HMMAR.loglike = loglike;
        break       
    end
    
    
end



end