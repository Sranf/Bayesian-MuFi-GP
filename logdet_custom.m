function [logdet] = logdet_custom(matrix)
try
    logdet = 2*sum(log(diag(chol(matrix))));
catch
    logdet = sum(log(eig(matrix)));
end
