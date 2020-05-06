function HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk)

n_it = 20;
th = 0.01;
p = size(theta, 1);
X = [];
for i = p + 1:length(y)
    Fn = -fliplr(y(i - p:i - 1));
    X = [X; Fn];
end
% Cluster AR coefficients without low evidence
idxTh = ev_y > evTh;
thetaev = theta(:, idxTh);
GMModel = fitgmdist(thetaev', ncomp);
pik = GMModel.ComponentProportion';
pik_ini = pik;
ak = GMModel.mu';
ak_ini = ak;
fl = 1;
ct = 0;
loglike = [];
while fl
    logalpha = zeros(ncomp, length(y));
    logbeta = zeros(ncomp, length(y));
    % E STEP
    % Forward and backward algorithm
    iforw = p+1:numel(y);
    iback = numel(y):-1:p+1;
    for i = 1:numel(iforw)
        if i == 1
            % Alpha
            Fnforw = -fliplr(y(iforw(i) - p:iforw(i) - 1));
            for k = 1:ncomp
                logalpha(k, iforw(i)) = log(pik(k)) + log(normpdf(y(iforw(i)), Fnforw*ak(:,k), sigk(k)));
            end
            % Beta - Redundant
            logbeta(:, iback(i)) = 0;
        else
            % Alpha
            Fnforw = -fliplr(y(iforw(i) - p:iforw(i) - 1));
            for j = 1:ncomp
                a = zeros(1, ncomp);
                lpem = log(normpdf(y(iforw(i)), Fnforw*ak(:,j), sigk(j)));
                for k = 1:ncomp
                    a(k) = lpem + logalpha(k, iforw(i) - 1) + log(A(k, j));
                end
                b = max(a);
                logalpha(j, iforw(i)) = b + log(sum(exp(a - b)));
            end
            % Beta
            Fnback = -fliplr(y(iback(i) - p + 1: iback(i)));
            for j = 1:ncomp
                a = zeros(1, ncomp);
                for k = 1:ncomp
                    a(k) = logbeta(k, iback(i) + 1) + log(normpdf(y(iback(i) + 1), Fnback*ak(:,k), sigk(k))) + log(A(j, k));
                end
                b = max(a);
                logbeta(j, iback(i)) =  b + log(sum(exp(a - b)));
            end
        end       
    end    
    pX = (sum(exp(logalpha(:,end))));       % Likelihood
    loglike = [loglike log(pX)];
    % Gamma and Xi    
    gam = zeros(ncomp, length(y));
    eta = zeros(ncomp, ncomp, length(y));
    % Special case for first gamma (to avoid an if inside the for loop)
    i = p + 1;
    gam(:, i) = (exp(logalpha(:,i)).*exp(logbeta(:,i)))/pX;   
    for i = p+2:numel(y)
        % Gamma
        gam(:, i) = (exp(logalpha(:,i)).*exp(logbeta(:,i)))/pX;
        % Xi
        Fn = -fliplr(y(i - p: i - 1));
        for j = 1:ncomp
            for k = 1:ncomp
                eta(j, k, i) = exp(logalpha(j, i - 1))*normpdf(y(i), Fn*ak(:,k), sigk(k))*A(j,k)*exp(logbeta(k, i));
            end
        end
    end
    eta = eta/pX;
 
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
        Ck = diag(sqrt(gam(k, p + 1:end)));
        Xtil = Ck*X;
        Ytil = Ck*y(p + 1:end)';
        ak(:, k) = (Xtil'*Xtil)\(Xtil'*Ytil);
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
            HMMAR.loglike = loglike;
            break       
        end
    end
    
    if ct == n_it
        HMMAR.pi = pik;
        HMMAR.A = A;
        HMMAR.ak = ak;
        HMMAR.sig = sigk;
        HMMAR.loglike = loglike;
        break       
    end
    
    
end



end