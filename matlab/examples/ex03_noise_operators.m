%EX03_NOISE_OPERATORS  Sensor noise: LS, bias-compensated LS and TLS (OMA setting).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = ncvd.setStyle("matlab");
[M, C, K] = ncvd.paperChain10(); lt = ncvd.trueModes(M, C, K);
fs = 200; T = 150; t = (0:1/fs:T-1/fs)'; snrs = [Inf 40 30 25 20 15]; nReal = 5;
ops = {{}, {'Operator', "bcls", 'NoiseVar', "auto", 'Fs', fs}, {'Operator', "tls"}};
names = ["LS (canonical)" "BCLS, estimated noise" "TLS"];
E = zeros(numel(snrs), nReal, 3, 2); rng(0);
for i = 1:numel(snrs)
    for r = 1:nReal
        [d, v, a] = ncvd.simulateForcedLinear(M, C, K, 10*randn(numel(t), 10), t);
        d = ncvd.addNoise(d, snrs(i), DB=true, PerChannel=true);
        v = ncvd.addNoise(v, snrs(i), DB=true, PerChannel=true);
        a = ncvd.addNoise(a, snrs(i), DB=true, PerChannel=true);
        for k = 1:3
            lam = ncvd.decompose(d, v, a, ops{k}{:}).lam; le = lam(ncvd.physicalModes(lam));
            le = arrayfun(@(l) le(find(abs(le - l) == min(abs(le - l)), 1)), lt);
            zt = -real(lt)./abs(lt); ze = -real(le)./abs(le);
            E(i,r,k,:) = [median(abs(abs(le) - abs(lt))./abs(lt)), median(abs(ze - zt)./zt)];
        end
    end
end
xs = snrs; xs(isinf(xs)) = 50;
figure('Position', [100 100 720 280]);
lab = {'median $|\hat\omega_n-\omega_n|/\omega_n$', 'median $|\hat\zeta-\zeta|/\zeta$'};
for q = 1:2
    subplot(1,2,q); hold on; box on; set(gca, 'YScale', 'log', 'XDir', 'reverse');
    for k = 1:3
        errorbar(xs, mean(E(:,:,k,q), 2), std(E(:,:,k,q), 0, 2), '-o', 'Color', C0(k,:), 'DisplayName', names(k));
    end
    xlabel('SNR (dB)'); ylabel(lab{q});
end
legend
