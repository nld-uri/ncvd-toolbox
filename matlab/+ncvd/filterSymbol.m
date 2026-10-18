function d = filterSymbol(f, theta, fs)
%NCVD.FILTERSYMBOL  d(theta) = fs*sum c_j theta^j of a derivative stencil.
d = fs*sum(f.c(:).*theta(:).'.^f.off(:), 1);
d = reshape(d, size(theta));
end
