% NCVD  Nonlinear Characteristic Value Decomposition toolbox, version 1.1.2
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
%   numericalRank        - declared numerical-rank rule of Rank="auto" (1.1.2)
% Modal quantities
%   modalParameters, physicalModes, realify, mac, macMatrix,
%   macPhaseMatched, relativeError, normalizeShape, matchModes
% Kinematics
%   derivFilter, filterSymbol, filterBand, applyFilter,
%   stateFromDisplacement, correctPoles
% Nonlinear tracking
%   track, windowed, windowEdges, bandFrequency
%   (track's defaults are the 1.1.1 procedure; windowed is the 1.1.2 protocol)
% Ensemble
%   ensemble
% Reduced-order models
%   robustnessBasis, deflate, escape, closure, familyClosure, project
% Benchmarks
%   gcvd, dmd, itd
% Test systems
%   chain, paperChain10, paperChain3, stateMatrix, trueModes,
%   simulateLinear, simulateForcedLinear, duffingChain,
%   simulateNonlinear, addNoise
% Figures
%   setStyle
% Citation
%   cite                 - BibTeX for the software (doi:10.5281/zenodo.23549068) and the method paper
