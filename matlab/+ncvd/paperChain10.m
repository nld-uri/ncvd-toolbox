function [M, C, K] = paperChain10()
%NCVD.PAPERCHAIN10  10-DOF chain of Li, Stein & Chelidze (2025), App. C.1.
n = 10; M = eye(n);
K = 8*eye(n) - 4*(diag(ones(n-1,1),1) + diag(ones(n-1,1),-1));
C = diag([0.9 0.9 0.6*ones(1,8)]) - 0.3*(diag(ones(n-1,1),1) + diag(ones(n-1,1),-1));
end
