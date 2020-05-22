function plotHMMAR(HMMAR, Fs)

figure
ncomp = size(HMMAR.ak, 2);
for k = 1:ncomp
    [H, F] = freqz(1, [1; HMMAR.ak(:,k)], 1024*4, Fs);
    subplot(ncomp, 1, k)
    %plot(F, 20*log10(abs(H)))  
    plot(F, abs(H))  
end
end