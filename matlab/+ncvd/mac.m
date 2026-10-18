function m = mac(a, b)
%NCVD.MAC  Modal assurance criterion |a'*b|^2/((a'*a)(b'*b)) for complex vectors.
a = a(:); b = b(:);
m = abs(a'*b)^2/max(real(a'*a)*real(b'*b), 1e-300);
end
