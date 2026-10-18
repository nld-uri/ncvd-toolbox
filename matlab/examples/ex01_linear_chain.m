%EX01_LINEAR_CHAIN  Exact NCVD of the generally damped 10-DOF chain (App. C.1).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = ncvd.setStyle("matlab");
[M, C, K] = ncvd.paperChain10();
t = (0:0.05:60-0.05)';
rng(0); [d, v, a] = ncvd.simulateLinear(M, C, K, randn(20,1), t);
rec = ncvd.decompose(d, v, a);
[lt, Et] = ncvd.trueModes(M, C, K);
idx = ncvd.physicalModes(rec.lam);
fprintf('identity 1: %.1e   identity 2: %.1e   pole error: %.1e\n', ...
    rec.exact1, rec.res, max(abs(rec.lam(idx) - lt)));
mp = ncvd.modalParameters(rec.lam(idx));
for i = 1:10
    fprintf('mode %2d  wn %.5f  zeta %.5f  MAC %.12f\n', i, mp.wn(i), mp.zeta(i), ...
        ncvd.mac(rec.Psi(1:10, idx(i)), Et(:,i)));
end
figure('Position', [100 100 700 300]);
subplot(1,2,1); hold on; box on
plot(real([lt; conj(lt)]), imag([lt; conj(lt)]), 'ko', 'MarkerSize', 8);
plot(real(rec.lam), imag(rec.lam), 'o', 'Color', C0(1,:), 'MarkerSize', 4);
xline(0, '--', 'Color', [0.5 0.5 0.5]);
xlabel('$\mathrm{Re}\,\lambda$'); ylabel('$\mathrm{Im}\,\lambda$'); legend('truth', 'NCVD');
subplot(1,2,2); hold on; box on
for j = 1:3
    k = [1 2 4]; e = ncvd.normalizeShape(Et(:,k(j))); s = ncvd.normalizeShape(rec.Psi(1:10, idx(k(j))));
    plot(real([0; e]), imag([0; e]), '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'HandleVisibility', 'off');
    plot(real([0; s]), imag([0; s]), 'o', 'Color', C0(j,:), 'MarkerSize', 4, 'DisplayName', sprintf('mode %d', k(j)));
end
xlabel('$\mathrm{Re}\,\psi$'); ylabel('$\mathrm{Im}\,\psi$'); legend('Location', 'southwest'); axis equal
