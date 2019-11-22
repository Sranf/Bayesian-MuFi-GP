close all
clear all
fprintf('****************** LETs GO! *******************')


seed = 8; rng(seed);
x1 = [0:0.020:1];
sparsification_vector = [1:5:length(x1)];
NS = 1;
 
test_set = linspace(0,1,100);

flag_alphascan = 1;
flag_prior = 1;


grid_size1 = 100;
grid_size2 = 100;
grid_size_comp = grid_size1*grid_size2;
GPsamples = 1;
sigma1_true = 1*10^-1;
sigma2_true= 1*10^-2;
order_of_basis_1 = 10;
order_of_basis_2 = 4;
order_of_basis_comp = 10;
nugget_prior_1 = 10^-8;
nugget_prior_2 = 10^-8;
nugget_prior_comp = 10^-3;
alpha1_true = 10.1;
alpha2_true = 20.1;
rho_true = 3;

if flag_alphascan == 1
    alpha1_grid = linspace(0.05,2,grid_size1)*alpha1_true;
    alpha2_grid = linspace(0.05,2,grid_size2)*alpha2_true;
    alpha2_grid_comp  = linspace(0.1,5,grid_size_comp)*alpha2_true;
else
    alpha1_grid = alpha1_true;
    alpha2_grid = alpha2_true;
end

[x1,x2, z1, z2, y2_true_mean, beta1_truth, beta2_truth] =...
    mockdata(x1, sparsification_vector, GPsamples, sigma1_true, sigma2_true,...
    alpha1_true, alpha2_true, rho_true, nugget_prior_1, nugget_prior_2, order_of_basis_1, order_of_basis_2,test_set);

z21 = z1(sparsification_vector);


tic
[logevidence, probability, predictive_mean, predictive_variance,...
    avg_beta1, avg_beta1var, avg_beta2, avg_beta2var, avg_rho1, avg_rho1SQ,...
    avg_sigma1,avg_sigma1SQ, avg_sigma1SQSQ, avg_sigma2, avg_sigma2SQ, avg_sigma2SQSQ]...
    = MUFI(test_set, x1,x2,z1,z2,z21,alpha1_grid,alpha2_grid, nugget_prior_1, nugget_prior_2, order_of_basis_1, order_of_basis_2,flag_prior);
comptime = toc;

errorbar = sqrt(predictive_variance-predictive_mean.^2);
marginal_alpha1 = sum(probability,2);
marginal_alpha2 = sum(probability,1);




figure;
subplot(1,2,1); plot(alpha1_grid, marginal_alpha1, 'LineWidth', 3); xlabel('\alpha_1'); hold on
ylabel('Posterior probability');
plot([alpha1_true, alpha1_true], [0, max(marginal_alpha1)], 'k--', 'LineWidth', 3);
set(gca, 'FontSize', 15)
subplot(1,2,2); plot(alpha2_grid, marginal_alpha2, 'LineWidth', 3); xlabel('\alpha_2');hold on
plot([alpha2_true, alpha2_true], [0, max(marginal_alpha2)], 'k--', 'LineWidth', 3);
set(gca, 'FontSize', 15)



figure;
L2 = fill([test_set, fliplr(test_set)], [predictive_mean+NS*errorbar; flipud(predictive_mean-NS*errorbar)], [1 0.7 0.7], 'facealpha', 0.5); hold on
L3 =plot(test_set, predictive_mean, 'r-', 'LineWidth', 3);
L4 =plot(test_set, y2_true_mean, 'b--', 'LineWidth', 2);
L5 =plot(x1,z1,'kd', 'MarkerSize', 8, 'MarkerFaceColor', 'white', 'MarkerEdgeColor', 'black');
L6 =plot(x2,z2,'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'white', 'MarkerEdgeColor', 'black');
legend([L3, L2, L4, L6, L5],{'Prediction',['Uncertainty (', num2str(NS),'\sigma)'], 'True Mean', 'z_2 Data', 'z_1 Data'})
xlabel('x'); ylabel('z'); %title('Comparison of MuFi to Truth');
ylim([min(z1),4.1])
set(gca, 'FontSize', 15)


std_beta1 = sqrt(avg_beta1var);
std_beta2 = sqrt(avg_beta2var);
std_sigma1 = sqrt(abs(avg_sigma1SQ-avg_sigma1.^2));
std_sigma2 = sqrt(abs(avg_sigma2SQ-avg_sigma2.^2));
std_sigma1SQ = sqrt(avg_sigma1SQSQ-avg_sigma1SQ.^2);
std_sigma2SQ = sqrt(avg_sigma2SQSQ-avg_sigma2SQ.^2);
std_rho1 = sqrt(avg_rho1SQ-avg_rho1.^2);

table(avg_beta1,std_beta1, beta1_truth)
table(avg_beta2,std_beta2, beta2_truth)
fprintf('<rho> = %8.4f +/- %8.4f (%8.4f)\n<sigma_1>    = %8.4f +/- %8.4f (%8.4f)\n<sigma_2>    = %8.4f +/- %8.4f (%8.4f)',...
            avg_rho1, std_rho1, rho_true, avg_sigma1, std_sigma1, sigma1_true, avg_sigma2, std_sigma2, sigma2_true)                
fprintf('\nLogEvidence MuFi = %8.0f\nLogEvidence Simple = %8.0f\n', logevidence)
        
comptime