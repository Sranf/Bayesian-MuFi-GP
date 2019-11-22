function [Kf] = kernelfun(x1,x2,b, nugget_prior)

[X1, X2] = ndgrid(x1,x2);
Kf = exp(-b.*(X1-X2).^2) +nugget_prior*eye(size(X1));