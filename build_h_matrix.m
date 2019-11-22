function [h] = build_h_matrix(order_of_basis, x)

h = [];
for n=0:order_of_basis-1
    Ll = legendre(n,2*x-1);
    h = [h; Ll(1,:)];
end

h = h';

