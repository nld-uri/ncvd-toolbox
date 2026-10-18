%EX04_ENSEMBLE  16 trials of Table 5 (different fs, length, SNR): least-squares fusion.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = ncvd.setStyle("matlab");
[M, C, K] = ncvd.paperChain10(); lt = ncvd.trueModes(M, C, K);
Af = [9 10 5 10 8 5 6 8 10 10 5 10 10 7 9 5];
SNR = [64 67 39 67 57 38 45 54 69 69 40 69 69 52 63 40];
Te = [254 277 81 279 208 74 119 187 290 292 89 293 290 171 250 85];
FS = [111 100 200 100 125 200 166 125 100 100 200 100 100 142 111 200];
rng(3); recs = cell(1,16);
for j = 1:16
    t = (0:1/FS(j):Te(j)-1/FS(j))';
    [d, v, a] = ncvd.simulateForcedLinear(M, C, K, Af(j)*randn(numel(t), 10), t);
    recs{j} = {ncvd.addNoise(d, SNR(j), DB=true), ncvd.addNoise(v, SNR(j), DB=true), ncvd.addNoise(a, SNR(j), DB=true)};
end
Ns = [1 2 4 8 16]; e = zeros(numel(Ns), 2);
for i = 1:numel(Ns)
    lam = ncvd.ensemble(recs(1:Ns(i))).lam; le = lam(ncvd.physicalModes(lam));
    le = arrayfun(@(l) le(find(abs(le - l) == min(abs(le - l)), 1)), lt);
    zt = -real(lt)./abs(lt);
    e(i,:) = [mean(abs(abs(le) - abs(lt))./abs(lt)), mean(abs(-real(le)./abs(le) - zt)./zt)];
end
figure; loglog(Ns, e(:,1), '-o', Ns, e(:,2), '-s'); grid on
xlabel('number of trials'); legend('$e_{\omega_n}$', '$e_\zeta$'); disp(e)
