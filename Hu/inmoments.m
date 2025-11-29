function  phinorm = inmoments(F)

if (ndims(F)~=2)||issparse(F)||~isreal(F)||~(isnumeric(F))||islogical(F)
    error("F must be a 2-D ")
end

F = double(F);

phi = compute_phi(compute_eta(compute_m(F)));

phinorm = -sign(phi).*(log10(abs(phi)));

end
 