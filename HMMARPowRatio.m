function [PowRatio, LogLike, akcell] = HMMARPowRatio(y, Fs, p_v)

% NO CROSS-VALIDATION

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

PowRatio = zeros(1, length(p_v));
LogLike = zeros(1, length(p_v));
%logp = NaN(length(p_v));
for i = 1:length(p_v)
    fprintf('p = %d \n', p_v(i))
    pxxBand = struct();
    pxxBand(1).dyn = [];
    pxxBand(2).dyn = [];
    [theta, ev_y] = KalmanAR(y, p_v(i), alph, sig2n);
    sigk = sqrt(sig2n)*ones(1, ncomp);
    HMMAR = HMMARLearning(y, ev_y, theta, ncomp, evTh, A, sigk);
    akcell{i} = HMMAR.ak;
    pwBand = zeros(1, ncomp);
    for k = 1:ncomp
        [H, f] = freqz(1, [1; HMMAR.ak(:,k)], f, Fs);
        pwBand(k) = sum(abs(H(f > bandHz(1) & f < bandHz(2))));
    end
    PowRatio(i) = max(pwBand)/min(pwBand)      % valid only for two components
    LogLike(i) = HMMAR.loglike(end)
    
%     [zopt, loglike(i)] = HMMARViterbi(y, HMMAR);
%     idx = find(diff(zopt) ~= 0);
%     for j = 1:length(idx)
%         if j == length(idx)
%             yaux = y(idx(j)+1:end);
%         else
%             yaux = y(idx(j)+1:idx(j+1));
%         end
%         dynmod = zopt(idx(j) + 1);
%         if length(yaux) > 0.25*Fs && length(yaux) > 2*p_v(i)                   % Threshold on duration and AR order
%             [pxx, ~] = pcov(yaux, p_v(i), f, Fs);
%             pxxBand(dynmod).dyn = [pxxBand(dynmod).dyn; sum(10*log10(pxx(f > bandHz(1) & f < bandHz(2))))];
%         else
%             sdf = 1;
%         end
%     end
%     if isempty(pxxBand(1).dyn) || isempty(pxxBand(2).dyn)
%         esdf = 1;
%     else
%         [~, p] = ttest2(pxxBand(1).dyn, pxxBand(2).dyn, 'Vartype','unequal');
%         logp(i) = log10(p);
%     end
%     p
%     logp(i)
end
end