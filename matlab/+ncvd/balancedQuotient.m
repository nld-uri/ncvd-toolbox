function Q = balancedQuotient(Y1, Y2)
%NCVD.BALANCEDQUOTIENT  Q = (K21/K11 + K22/K12 + K11/K21 + K12/K22)/4, Kij = Yi'*Yj.
K11 = Y1'*Y1; K12 = Y1'*Y2; K21 = Y2'*Y1; K22 = Y2'*Y2;
Q = 0.25*(K21/K11 + K22/K12 + K11/K21 + K12/K22);
end
