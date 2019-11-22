% This function evaluates Equations 4

function [logevidence, probability, predictive_mean, predictive_variance,...
    avg_beta1, avg_beta1_var, avg_beta2, avg_beta2_var, avg_rho1, avg_rho1SQ,...
    avg_sigma1, avg_sigma1SQ, avg_sigma1SQSQ, avg_sigma2, avg_sigma2SQ, avg_sigma2SQSQ]...
    = MUFI(test_set, x1,x2,z1,z2,z21,alpha1_grid,alpha2_grid, nugget_prior_1, nugget_prior_2, order_of_basis_1, order_of_basis_2,flag_prior)

% Pre-build matrix of regression functions
h1 = build_h_matrix(order_of_basis_1, x1);
h2 = build_h_matrix(order_of_basis_2, x2);
h1test = build_h_matrix(order_of_basis_1, test_set);
h2test = build_h_matrix(order_of_basis_2, test_set);

gam1 = (length(x1)-order_of_basis_1)/2; 
gam2 = (length(x2)-order_of_basis_2)/2; 

% Check constraints
if gam1  <= 1, disp(num2str(gam1)); disp('gamma1 too small, sigma1SQ does not exist'); end
if gam1  <= 1, disp(num2str(gam1)); disp('gamma1 too small, sigma1SQSQ does not exist'); end
if gam2  <= 5/2, disp(num2str(gam2)); disp('gamma2 too small, sigma2SQ does not exist'); end
if gam2  <= 5/2, disp(num2str(gam2)); disp('gamma2 too small, sigma2SQSQ does not exist'); end

% This loop computes the posterior probability of alpha according to
% Equation 4
for k1 = 1:length(alpha1_grid)
    for k2 = 1:length(alpha2_grid)
        
        
        K1 = kernelfun(x1,x1, alpha1_grid(k1), nugget_prior_1);     
        K2 = kernelfun(x2,x2, alpha2_grid(k2), nugget_prior_2);
        invK1 = inv(K1);
        invK2 = inv(K2);        
        B1 = h1'*invK1;
        B2 = h2'*invK2;
        A1 = inv(B1*h1);
        A2 = inv(B2*h2);
        
        Check_ConditionNumber = [cond(K1),cond(K2),cond(A1),cond(A2)];
        if any(Check_ConditionNumber)<10^-10  ||  any(Check_ConditionNumber) > 10^10;
            disp('Badly conditioned matrix')
        end

        C1 = invK1 - B1'*A1*B1;
        C2 = invK2 - B2'*A2*B2;
        
        a1 = 1;
        b1 = 0;
        c1 = z1'*C1*z1;
        Phi1 = c1;
        
        a2 = z21' * C2 * z21;
        b2 = z2' * C2 * z21;
        c2 = z2' * C2 * z2; 
        Phi2 = c2 - b2^2/a2;
        
        logprob(k1,k2) = (-gam1)*log(Phi1) + gammaln(gam1) -1/2*(logdet_custom(K1)-logdet_custom(A1))...
             -1/2*log((a2)) +(-gam2+1/2)*log(Phi2) + gammaln(gam2-1/2) -1/2*(logdet_custom(K2)-logdet_custom(A2) )...
             -flag_prior*log(alpha1_grid(k1))-flag_prior*log(alpha2_grid(k2));
         
         logevidence = logPLUS(logevidence, logprob(k1,k2)*(alpha1_grid(2)-alpha1_grid(1))*(alpha2_grid(2)- alpha2_grid(1)));
        

    end
end
 
probability = exp(logprob-max(logprob(:)));
probability = probability ./ sum(probability(:));
 
predictive_mean = zeros(size(test_set))';
predictive_variance = zeros(size(test_set))';
avg_beta1 = zeros([order_of_basis_1, 1]);
avg_beta1_std = zeros([order_of_basis_1, 1]);
avg_beta1_var = zeros([order_of_basis_1, 1]);
avg_beta2 = zeros([order_of_basis_2, 1]);
avg_beta2_std = zeros([order_of_basis_2, 1]);
avg_beta2_var = zeros([order_of_basis_2, 1]);
avg_rho1 = 0;
avg_rho1SQ =0;
avg_sigma1 = 0;
avg_sigma1SQ = 0;
avg_sigma1SQSQ = 0;
avg_sigma2 = 0;
avg_sigma2SQ = 0;
avg_sigma2SQSQ = 0;

% This loop computes the expectations using the posterior probability from
% the previous loop
for k1 = 1:length(alpha1_grid)
    for k2 = 1:length(alpha2_grid)
        
        K1 = kernelfun(x1,x1, alpha1_grid(k1), nugget_prior_1);     
        K2 = kernelfun(x2,x2, alpha2_grid(k2), nugget_prior_2);
        invK1 = inv(K1);
        invK2 = inv(K2);        
        B1 = h1'*invK1;
        B2 = h2'*invK2;
        A1 = inv(B1*h1);
        A2 = inv(B2*h2);
        
        Check_ConditionNumber = [cond(K1),cond(K2),cond(A1),cond(A2)];
        if any(Check_ConditionNumber)<10^-10  ||  any(Check_ConditionNumber) > 10^10;
            disp('Badly conditioned matrix')
        end

        C1 = invK1 - B1'*A1*B1;
        C2 = invK2 - B2'*A2*B2;
        
        a1 = 1;
        b1 = 0;
        c1 = z1'*C1*z1;
        Phi1 = c1;
        
        a2 = z21' * C2 * z21;
        b2 = z2' * C2 * z21;
        c2 = z2' * C2 * z2; 
        Phi2 = c2 - b2^2/a2;
        
        sigma1 = sqrt(Phi1)*gamma(gam1-1/2)/gamma(gam1); %*exp(gammaln(gam1-1/2)-gammaln(gam1));
        sigma1SQ = Phi1/(2*gam1);
        sigma1SQSQ = Phi1^2 /( (gam1-2)*(gam1-1));
        sigma2 = sqrt(Phi2)*gamma(gam2-1)/gamma(gam2-1/2);% *exp(gammaln(gam2-1)-gammaln(gam2-1/2));
        sigma2SQ = Phi2/(numel(x2)-order_of_basis_2-3);
        sigma2SQSQ = Phi2^2*exp(gammaln(gam2-1/2-2)-gammaln(gam2-1/2));
        
        rho1 = b2/a2;
        rho1SQ = sigma2SQ/a2 + (b2/a2)^2;
        beta1 = A1*B1*z1;
        beta2 = A2*B2*(z2-rho1*z21);
        
        K1test = kernelfun(x1,test_set, alpha1_grid(k1), nugget_prior_1)';
        K2test = kernelfun(x2,test_set, alpha2_grid(k2), nugget_prior_2)';
        Ktilde1 = K1test*invK1;
        Ktilde2 = K2test*invK2;
        
        g1 = h1test - Ktilde1*h1;
        g2 = h2test - Ktilde2*h2;
        u1 = Ktilde1*z1;
        u2 = Ktilde2*(z2-rho1*z21);
        predictive_mean = predictive_mean + ( (u1+g1*beta1)*rho1 + (u2+g2*beta2))*probability(k1,k2);
            
        Kpost1 =  (kernelfun(test_set,test_set, alpha1_grid(k1), nugget_prior_1) - Ktilde1*K1test');
        Kpost2 =  (kernelfun(test_set,test_set, alpha2_grid(k2), nugget_prior_2) - Ktilde2*K2test');   

        d1 = u1 + g1*beta1;
        d2 = u2 + g2*beta2 + rho1*d1;

        covariance = d2*d2' + rho1SQ*sigma1SQ *(g1*A1*g1' +Kpost1) + sigma2SQ*(g2*A2*g2' +Kpost2)...
            + (rho1SQ-rho1^2)* (Ktilde2*z21)*(Ktilde2*z21)';
        predictive_variance = predictive_variance + diag(covariance)*probability(k1,k2);
   
        
        avg_beta1 = avg_beta1 + beta1*probability(k1,k2);
        avg_beta1_std = avg_beta1_std + sigma1./sqrt(diag((A1)))*probability(k1,k2);
        avg_beta1_var = avg_beta1_var + sigma1SQ./(diag((A1)))*probability(k1,k2);
        avg_beta2 = avg_beta2 + beta2*probability(k1,k2);
        avg_beta2_std = avg_beta2_std + sigma2./sqrt(diag((A2)))*probability(k1,k2);
        avg_beta2_var = avg_beta2_var + sigma2SQ./diag((A2))*probability(k1,k2);
        avg_rho1 = avg_rho1 + rho1*probability(k1,k2);
        avg_rho1SQ = avg_rho1SQ + rho1SQ*probability(k1,k2);
        avg_sigma1 = avg_sigma1 + sigma1*probability(k1,k2);
        avg_sigma1SQ = avg_sigma1SQ + sigma1SQ*probability(k1,k2);
        avg_sigma1SQSQ = avg_sigma1SQSQ + sigma1SQSQ*probability(k1,k2);
        avg_sigma2 = avg_sigma2 + sigma2*probability(k1,k2);
        avg_sigma2SQ = avg_sigma2SQ + sigma2SQ*probability(k1,k2);
        avg_sigma2SQSQ = avg_sigma2SQSQ + sigma2SQSQ*probability(k1,k2);
 
    end
end
 
