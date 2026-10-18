function [M, C, K] = paperChain3()
%NCVD.PAPERCHAIN3  3-DOF chain with widely spaced modes, App. C.2.
M = eye(3);
C = [7.07 -0.01 0; -0.01 0.42 -0.0055; 0 -0.0055 15.81];
K = [5000 -9 0; -9 18 -9; 0 -9 25000];
end
