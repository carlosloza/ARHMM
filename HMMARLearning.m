function HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk)
% Implementation with scaled alphas and betas
dmin = 2e-300;          % To avoid underflow
n_it = 20;
th = 0.01;
p = size(theta, 1);
X = zeros(length(y) - p, p);
ct = 1;
for i = p + 1:length(y)
    Fn = -fliplr(y(i - p:i - 1));
    X(ct, :) = Fn;
    ct = ct + 1;
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
while fl
    % E STEP
    % Forward and backward algorithm
    c = ones(1, numel(y));
    % Alpha
    alphaforw = ones(ncomp, length(y));
    iforw = p+1:numel(y);
    for i = 1:numel(iforw)
        if i == 1
            % Alpha
            Fnforw = -fliplr(y(iforw(i) - p:iforw(i) - 1));
            aux = zeros(ncomp, 1);
            for k = 1:ncomp
                %aux(k) = pik(k) * normpdf(y(iforw(i)), Fnforw*ak(:,k), sigk(k));
                aux(k) = pik(k) * max([normpdf(y(iforw(i)), Fnforw*ak(:,k), sigk(k)), dmin]);
            end
            % Scaled versions
            c(iforw(i)) = 1/sum(aux);
            alphaforw(:, iforw(i)) = c(iforw(i))*aux;
        else
            % Alpha
            Fnforw = -fliplr(y(iforw(i) - p:iforw(i) - 1));
            aux = zeros(ncomp, 1);
            for j = 1:ncomp
                %pem = normpdf(y(iforw(i)), Fnforw*ak(:,j), sigk(j));
                pem = max([normpdf(y(iforw(i)), Fnforw*ak(:,j), sigk(j)), dmin]);
                aux(j) = pem*(alphaforw(:, iforw(i) - 1)'*A(:,j));
            end
            % Scaled versions
            c(iforw(i)) = 1/sum(aux);
            alphaforw(:, iforw(i)) = c(iforw(i))*aux;
        end       
    end    
    % Likelihood
    loglike = [loglike -sum(log(c))];
    % Beta
    iback = numel(y):-1:p+1;
    betaback = ones(ncomp, length(y));
    for i = 1:numel(iback)
        if i == 1
            betaback(:, iback(i)) = c(iback(i))*1;
        else
            Fnback = -fliplr(y(iback(i) - p + 1: iback(i))); 
            aux1 = zeros(ncomp, 1);
            for j = 1:ncomp
                aux2 = zeros(1, ncomp);
                for k = 1:ncomp
                    %aux2(k) = betaback(k, iback(i) + 1) * normpdf(y(iback(i) + 1), Fnback*ak(:,k), sigk(k)) * A(j, k);
                    aux2(k) = betaback(k, iback(i) + 1) * max([normpdf(y(iback(i) + 1), Fnback*ak(:,k), sigk(k)), dmin]) * A(j, k);
                end
                aux1(j) = sum(aux2);
            end
            betaback(:, iback(i)) = c(iback(i))*aux1;
        end
    end
    
    % Gamma and Xi    
    gam = zeros(ncomp, length(y));
    eta = zeros(ncomp, ncomp, length(y));
    % Special case for first gamma (to avoid an if inside the for loop)
    i = p + 1;
    gam(:, i) = alphaforw(:,i).*betaback(:,i)/(sum(alphaforw(:,i).*betaback(:,i)));   
    for i = p+2:numel(y)
        % Gamma
        gam(:, i) = alphaforw(:,i).*betaback(:,i)/(sum(alphaforw(:,i).*betaback(:,i)));
        % Xi
        Fn = -fliplr(y(i - p: i - 1));
        for j = 1:ncomp
            for k = 1:ncomp
                %eta(j, k, i) = alphaforw(j, i - 1) * normpdf(y(i), Fn*ak(:,k), sigk(k)) * A(j,k) * betaback(k, i);
                eta(j, k, i) = alphaforw(j, i - 1) * max([normpdf(y(i), Fn*ak(:,k), sigk(k)), dmin]) * A(j,k) * betaback(k, i);
            end
        end
    end
 
    % M STEP
    % Initial latent variable probabilities
    pik = gam(:,p + 1)./sum(gam(:,p + 1));
    % State transition probabilities
    A = zeros(ncomp, ncomp);
    for j = 1:ncomp
        denA = 0;
        for kk = 1:ncomp
            denA = denA + sum(eta(j, kk , p+2:end));
        end
        for k = 1:ncomp
            numA = sum(eta(j, k, p+2:end));
            A(j,k) = numA/denA;
        end
    end
    % AR coefficients and nise variance per mode
    for k = 1:ncomp
        % AR coefficients
        Ckv = sqrt(gam(k, p + 1:end));
        %Ck = diag(sqrt(gam(k, p + 1:end)));
        %Xtil = Ck*X;
        Xtil = bsxfun(@times, Ckv', X);
        %Ytil = Ck*y(p + 1:end)';
        Ytil = (Ckv.*y(p + 1:end))';
        
        ak(:, k) = (Xtil'*Xtil)\(Xtil'*Ytil);
        %ak(:, k) = robustfit(Xtil, Ytil, 'huber', [], 'off');
        
        % Noise variance
        aux = zeros(1, length(y));
        for i = p + 1:length(y)
            Fn = -fliplr(y(i - p:i - 1));
            aux(i) = gam(k, i)*(y(i) - Fn*ak(:,k))^2;
        end
        sigk(k) = sqrt(sum(aux)/sum(gam(k,:)));
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