function [logp, loglikeTest] = HMMARCVPowRatio(y, Fs, p_v, kCV)

alph = 0.1;
evTh = 0.5;            % Evidence threshold
ncomp = 2;

f = 0:0.5:Fs/2;
bandHz = [10.5 16];     % Band of interest

% Initial transitions
stP = 0.5;              % Order of stationarity (in seconds) for EEG
stPaux = 1 - 1/(stP*Fs);
A = [stPaux, 1 - stPaux;
    1- stPaux, stPaux];

sig2n = 0.2;            % Could be latter estimated via static AR model

% Training/Test partitions
LCV = floor(length(y)/kCV);
tEp(:, 1) = 1:LCV:length(y);
tEp(:, 2) = tEp(:, 1) + LCV - 1;
tEp(end, 2) = length(y);
loglikeTest = zeros(length(p_v), kCV);
logp = NaN(length(p_v), kCV);
for i = 1:length(p_v)
    p_v(i)  
    for k = 1:kCV
        if k == 5
            sdf = 1;
        end
        % Only two modes so far
        pxxBand = struct();
        pxxBand(1).dyn = [];
        pxxBand(2).dyn = [];
        ytest = y(tEp(k,1):tEp(k,2));
        ytrain = y(setdiff(1:length(y),tEp(k,1):tEp(k,2)));
        [theta, ev_y] = KalmanAR(ytrain, p_v(i), alph, sig2n);
        sigk = sqrt(sig2n)*ones(1, ncomp);
        HMMAR = HMMARLearning(ytrain, ev_y, theta, ncomp, evTh, A, sigk);
        [zopt, loglikeTest(i, k)] = HMMARViterbi(ytest, HMMAR);
        idx = find(diff(zopt) ~= 0);
        for j = 1:length(idx)
            if j == length(idx)
                yaux = ytest(idx(j)+1:end);
            else
                yaux = ytest(idx(j)+1:idx(j+1));
            end
            dynmod = zopt(idx(j) + 1);
            if length(yaux) > 0.25*Fs && length(yaux) > 2*p_v(i)                   % Threshold on duration and AR order
            %if length(yaux) > 0.25*Fs
                [pxx, ~] = pcov(yaux, p_v(i), f, Fs);
                pxxBand(dynmod).dyn = [pxxBand(dynmod).dyn; sum(10*log10(pxx(f > bandHz(1) & f < bandHz(2))))];
            else
                sdf = 1;
            end
        end
        if isempty(pxxBand(1).dyn) || isempty(pxxBand(2).dyn)
            esdf = 1;
        else
            [~, p] = ttest2(pxxBand(1).dyn, pxxBand(2).dyn);
            logp(i, k) = log10(p);
        end      
    end
    nanmean(logp(i, :))
end
end