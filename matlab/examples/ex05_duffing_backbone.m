%EX05_DUFFING_BACKBONE  Frequency-energy dependence from one free decay.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = ncvd.setStyle("matlab");
[M, ~, K, fnl] = ncvd.duffingChain(3, 1, 1, 0, [1 1]); C = 0.003*M;
[l0, E0] = ncvd.trueModes(M, C, K);
sh = zeros(3); for i = 1:3, s = ncvd.realify(E0(:,i)); sh(:,i) = s/max(abs(s)); end
x0 = [1.5*sh(:,1) + 0.6*sh(:,2) + 0.5*sh(:,3); zeros(3,1)];
fs = 20; t = (0:1/fs:1500-1/fs)';
[d, v, a] = ncvd.simulateNonlinear(M, C, K, fnl, x0, t);
tr = ncvd.track(d, v, a, Fs=fs, Length=800, Step=400, NTrack=3, M=M);
fmeas = arrayfun(@(w) ncvd.bandFrequency(d(tr.edges(w,1):tr.edges(w,2), 1), 1/fs, 0.6, 1.1), (1:size(tr.edges,1))');
fprintf('mode-1 pole vs measured frequency: max rel. difference %.2e\n', max(abs(tr.wd(:,1) - fmeas)./fmeas));
figure('Position', [100 100 760 280]);
subplot(1,3,1); hold on; box on; set(gca, 'XScale', 'log');
for j = 1:3, plot(tr.energy(:,j), tr.wd(:,j)/abs(imag(l0(j))), '-o', 'Color', C0(j,:), 'MarkerSize', 3); end
plot(tr.energy(:,1), fmeas/abs(imag(l0(1))), 'kx'); yline(1, '--');
xlabel('modal energy'); ylabel('$\omega_d/\omega_{d,\mathrm{lin}}$'); legend('mode 1', 'mode 2', 'mode 3', 'measured')
subplot(1,3,2); semilogy(tr.t, tr.res, '-o', 'MarkerSize', 3); xlabel('$t$ (s)'); ylabel('$\|\mathbf{R}\|/\|\mathbf{X}_2\|$')
subplot(1,3,3); semilogy(tr.t, 1 - tr.macPrev); xlabel('$t$ (s)'); ylabel('$1-\mathrm{MAC}$ to previous window')
