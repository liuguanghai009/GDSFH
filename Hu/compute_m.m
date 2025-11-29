function m = compute_m(F)

[M, N] = size(F);
[x,y] = meshgrid(1:N,1:M);

x = x(:);
y = y(:);
F = F(:);

m.m00 = sum(F);

if(m.m00==0)
    m.m00 =eps;
end
m.m10 = sum(x.*F);
m.m01 = sum(y.*F);
m.m11 = sum(x.*y.*F);
m.m20 = sum(x.^2.*F);
m.m02 = sum(y.^2.*F);
m.m30 = sum(x.^3.*F);
m.m03 = sum(y.^3.*F);
m.m12 = sum(x.*y.^2.*F);
m.m21 = sum(x.^2.*y.*F);

end