%EX02_FD_CORRECTION  Displacement-only data: FIR rates and exact pole correction.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = ncvd.setStyle("matlab");
[M, C, K] = ncvd.paperChain3(); lt = ncvd.trueModes(M, C, K);
methods = ["forward" "central" "central4" "savgol"];
fsv = logspace(log10(60), log10(3000), 18);
raw = zeros(4, numel(fsv)); cor = raw;
for j = 1:numel(fsv)
    fs = fsv(j); t = (0:1/fs:3-1/fs)';
    q = ncvd.simulateLinear(M, C, K, [1 1 1 0 0 0]', t);
    for i = 1:4
        [d, v, a, f] = ncvd.stateFromDisplacement(q, fs, methods(i));
        lam = ncvd.decompose(d, v, a).lam; lam = lam(ncvd.physicalModes(lam));
        lc = ncvd.correctPoles(lam, f, fs);
        raw(i,j) = max(arrayfun(@(l) min(abs(lam - l))/abs(l), lt));
        cor(i,j) = max(arrayfun(@(l) min(abs(lc - l))/abs(l), lt));
    end
end
figure; hold on; box on; set(gca, 'XScale', 'log', 'YScale', 'log');
for i = 1:4
    plot(fsv, raw(i,:), '-', 'Color', C0(i,:), 'DisplayName', methods(i));
    plot(fsv, max(cor(i,:), 1e-16), ':o', 'Color', C0(i,:), 'MarkerSize', 3, 'HandleVisibility', 'off');
    xline(max(abs(imag(lt)))/ncvd.filterBand(ncvd.derivFilter(methods(i))), '--', 'Color', C0(i,:), 'HandleVisibility', 'off');
end
xlabel('sampling rate $f_s$ (Hz)'); ylabel('$\max_i|\hat\lambda_i-\lambda_i|/|\lambda_i|$'); legend
