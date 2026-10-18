% NCVD  Nonlinear Characteristic Value Decomposition toolbox, version 1.0.0
%
%   X1 = [d v] = U*Phi',   X2 = [v a] = U*diag(lam)*Phi' + R
%
% Core
%   decompose            - canonical NCVD of a state record
%   decomposePair        - canonical NCVD of a generic pair (Y1, Y2)
%   balancedQuotient     - balanced quotient Q
%   regressionOperator   - LS, TLS or bias-compensated LS operator
%   autoTimeScale        - time unit placing all poles outside |lam| = 1
%   noiseVariance        - sensor-noise variance from the Nyquist plateau
% Modal quantities
%   modalParameters, physicalModes, realify, mac, macMatrix,
%   macPhaseMatched, relativeError, normalizeShape, matchModes
% Kinematics
%   derivFilter, filterSymbol, filterBand, applyFilter,
%   stateFromDisplacement, correctPoles
% Nonlinear tracking
%   track, windowEdges, bandFrequency
% Ensemble
%   ensemble
% Reduced-order models
%   robustnessBasis, deflate, escape, closure, project
% Benchmarks
%   gcvd, dmd, itd
% Test systems
%   chain, paperChain10, paperChain3, stateMatrix, trueModes,
%   simulateLinear, simulateForcedLinear, duffingChain,
%   simulateNonlinear, addNoise
% Figures
%   setStyle
