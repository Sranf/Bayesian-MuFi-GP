function [x1,x2, z1, z2, y2_true_mean, beta1_gen, beta2_gen] =...
    mockdata(x1, sparsification_vector, samplesize, sigma1_true, sigma2_true,...
    b1_true, b2_true, rho_true, nugget_prior_1, nugget_prior_2, order_of_basis_1, order_of_basis_2, test_set);

x2 = x1(sparsification_vector);

beta1_gen = [
    0.32422;...
    -0.39938;...
    0.09610;...
    0.35325;...
    -0.51293;...
    0.33443;...
    -0.03421;...
    -0.14240;...
    0.17497;...
    -0.10342];
beta1_gen = beta1_gen(1:order_of_basis_1);
beta2_gen = [
    0.333333333333333;...
    0.150000000000000;...
    0.016666666666667;...
    0];
beta2_gen = beta2_gen(1:order_of_basis_2);

h1 = build_h_matrix(order_of_basis_1, x1);
yl = h1*beta1_gen;
h2 = build_h_matrix(order_of_basis_2, x2);
yh = h2*beta2_gen + rho_true * build_h_matrix(order_of_basis_1, x2)*beta1_gen;
y2_true_mean = build_h_matrix(order_of_basis_2, test_set)*beta2_gen + rho_true*build_h_matrix(order_of_basis_1, test_set)*beta1_gen ;

K1f_train = kernelfun(x1,x1, b1_true, nugget_prior_1);
K2f_train = kernelfun(x2,x2, b2_true, nugget_prior_2);
 
[nx1, ~] = size(h1);
[nx2, ~] = size(h2);

yL_sample = [];
yH_sample = [];
 
for k=1:samplesize
    rand1 = randn(nx1,1);
    rand2 = randn(nx2,1);
    [A1, S1,~] = svd(K1f_train);
    [A2, S2,~] = svd(K2f_train);
    newsampleL = yl + sigma1_true*A1*sqrt(S1)*rand1;
    newsampleH = yh + sigma2_true*A2*sqrt(S2)*rand2;
    yL_sample = [yL_sample, newsampleL'];
    yH_sample = [yH_sample, newsampleH'];
 
end

x_rep1 = repmat(x1, [1 samplesize]);
x_rep2 = repmat(x2, [1 samplesize]);
x1 = x_rep1';
x2 = x_rep2';
z1 = yL_sample';
z2 = yH_sample';
 
