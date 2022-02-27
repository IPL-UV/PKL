% Kernel Fair Learning
% FINAL VERSION - adapted to the suplementary variables experiment 

function [res] = PKL(Xnt,ynt,X_train,y_train,Q_train,X_val,y_val,Q_val,X_test,y_test,Q_test,mus,lambdas,sigmas,cross_val,optimize_sq)

% Length samples
ntr     = length(y_train);
nva     = length(y_val);
nte     = length(y_test);

% Centering matrices
H_train = eye(ntr) - (1/ntr)*(ones(ntr)); 
H_test  = eye(nte) - (1/nte)*(ones(nte)); 
H_val   = eye(nva) - (1/nva)*(ones(nva)); 

% Length hyperparameters
l_sigmas = length(sigmas);
l_lambdas = length(lambdas);
l_mus = length(mus);

%% Kernel Ridge Regression
%% 1) Optimize 
if cross_val
    folds = 10;
    Nnt = ntr+nva;
    cv_indices = crossvalind('Kfold',Nnt,folds);
    rmse_val_krr = zeros(length(sigmas),length(lambdas), folds);
    for k = 1:folds    
        for i=1:length(sigmas)
            K_train   = rbf(Xnt(cv_indices ~= k,:),Xnt(cv_indices ~= k,:),sigmas(i));
            K_val = rbf(Xnt(cv_indices == k,:),Xnt(cv_indices ~= k,:),sigmas(i));
            for j=1:length(lambdas)
                alpha_krr  = (lambdas(j)*eye(length(ynt(cv_indices ~= k,:))) + K_train)\(ynt(cv_indices ~= k,:));
                rmse_val_krr(i,j,k) = sqrt(mean((ynt(cv_indices == k,:) - K_val*alpha_krr).^2));
            end
        end
    end

    [~,loc] = min(rmse_val_krr(:));
    [I,J,~] = ind2sub(size(rmse_val_krr),loc);

else
    rmse_val_krr = zeros(l_sigmas,l_lambdas);

    for i=1:length(sigmas)
        K_train   = rbf(X_train,X_train,sigmas(i));
        K_val = rbf(X_val,X_train,sigmas(i));
        for j=1:length(lambdas)
            fprintf('KRR: nº sigma= %d/%d, nº lambda= %d/%d \n',i,l_sigmas,j,l_lambdas)

            % Alphas KRR
            alpha_krr  = (lambdas(j)*eye(ntr) + K_train)\(y_train);

            % RMSE
            rmse_val_krr(i,j) = sqrt(mean((y_val - K_val*alpha_krr).^2));
        end
    end

    % parameters for KRR
    % visualizing hyperparameter - rmse
    %figure(100), clf, surf(rmse_val_krr),colorbar, ylabel('Sigma'), xlabel('Lambda'), zlabel('RMSE val')

    mm = min(min(rmse_val_krr));
    [I,J] = find(rmse_val_krr==mm);
end
if length(I)>1
    % warning('more than one maximum')
    I = I(1);J = J(1);
end
best_sigma = sigmas(I); 
best_lambda = lambdas(J);

%% 2) Define the kernels
K_train  = rbf(X_train,X_train,best_sigma);
K_test = rbf(X_test,X_train,best_sigma);

%% 3) Inference
alpha_krr = (best_lambda*eye(ntr) + K_train)\(y_train); % optimal KRR weights

y_hat_krr = K_test*alpha_krr; 

%% 4) Metrices
rmse_krr = sqrt(mean((y_hat_krr - y_test).^2));

Rp_krr = corr(y_test - y_hat_krr, Q_test, 'Type','Pearson');
Rs_krr = corr(y_test - y_hat_krr, Q_test, 'Type','Spearman');

compute_kernel=true; normalize=false; nocco=false; 
nhsic_krr = computeHSIC(y_hat_krr, Q_test, compute_kernel, normalize, nocco);

%acc_wc = sqrt(mean((yte - K_test*alpha_krr).^2));
% A) HSIC(Yhat,S) versio pseudo NOCCO
%dep_wc = (1/nte^2)*alpha_krr'*K_test'*(HKq_testH*pinv(HKq_testH + epsilon*eye(nte)))*K_test*alpha_krr;

%% Physics-aware Nonparametric Regression
%% 1) Optimize  
nhsic_q = zeros(l_sigmas,1);

l_sigmas_q = 20;
sigmas_q  = logspace(-1,1,l_sigmas);
if optimize_sq && ntr == nva % fprintf('Validation and training have diferent sizes')
    for i=1:l_sigmas_q
        fprintf('PKL: nº sigma_q= %d/%d',i,l_sigmas_q)
         
        % Search for a sigma that maximizes HSIC with y_val
        Ky_val = y_val*y_val';
        Kq_val  = rbf(Q_val,Q_train,sigmas_q(i));
        % Centering
        HKy_valH = H_val*Ky_val*H_val;
        HKq_valH = H_val*Kq_val*H_train;

        % HSIC
        compute_kernel=false; normalize=false; nocco=false; 
        nhsic_q(i) = computeHSIC(HKy_valH, HKq_valH, compute_kernel, normalize, nocco);
    end    
    
    % visualizing hyperparameter - rmse
    %figure(100), clf, plot(nhsic_q),colorbar, ylabel('NHISC val'), xlabel('Sigmas_q')
    
    max_hsic = max(nhsic_q);
    [I] = find(max_hsic==nhsic_q);
    sigma_q = sigmas_q(I); 
else
    if ntr>5e2
        % if there are lots of samples, it will choose a sigma from an heuristic 
        % stimator using a fraction of the samples. If not, it will use all the
        % samples
        rp = randperm(ntr);
        xaux = X_train(rp(1:5e2),:);
        sigma_q = sqrt(.5*median(pdist(xaux).^2));
    else
        sigma_q = sqrt(.5*median(pdist(X_train).^2));
    end
end
    
%% 2) Define the kernels for Q
Kq_train = rbf(Q_train,Q_train,sigma_q);
Kq_test  = rbf(Q_test,Q_train,sigma_q);

% Centering
HK_trainH = H_train*K_train*H_train;
HKq_trainH = H_train*Kq_train*H_train;
HKq_testH = H_test*Kq_test*H_train;

%% 3) Inference
rmse_pkl = zeros(1,l_mus);
nhsic_pkl = zeros(1,l_mus);
alpha_pkl = cell(1,l_mus);

epsilon = 1e-2; %it needs to be very small -2

%HKyhatq_trainH = H_train*rbf(Q_train,X_train,sigma_q)*H_train;

for k=1:l_mus
    fprintf('PKL: nº mu= %d/%d \n',k,l_mus)
    
    % Alphas KRR
    wd = (best_lambda*eye(ntr) + K_train + mus(k)*((HKq_trainH*pinv(HKq_trainH + epsilon*eye(ntr))))*HK_trainH)\(y_train);
    %wd = (best_lambda*eye(ntr) + K_train + (mus(k)/ntr)*(Q_train'*Q_train)*HK_trainH)\(y_train);
    %wd = (best_lambda*eye(ntr) + K_train + mus(k)*HK_trainH*((HKq_trainH*pinv(HKq_trainH + epsilon*eye(ntr)))))\(y_train);
    %wd = (best_lambda*eye(ntr) + K_train + mus(k)*(HKyhatq_trainH*HKyhatq_trainH))\(y_train);
    alpha_pkl{k} = wd;
    
    %wd = (best_lambda*eye(ntr) + (eye(ntr)-mus(k)*((HKq_trainH*pinv(HKq_trainH + epsilon*eye(ntr))))*K_train + ...
    %    mus(k)*((HKq_trainH*pinv(HKq_trainH + epsilon*eye(ntr)))*HK_trainH)))\(y_train);
    
    y_hat_pkl = K_test*wd;
    
    %% 4) Metrices (RMSE, HSIC)
    rmse_pkl(k) = sqrt(mean((y_test - y_hat_pkl).^2));
    compute_kernel=true; normalize=false; nocco=false;
    nhsic_pkl(k) = computeHSIC(y_hat_pkl, Q_test, compute_kernel, normalize, nocco);
end

%% return model and parameters.
% To set range start at 1 for KRR
nhsic_pkl = nhsic_pkl./nhsic_krr;
nhsic_krr = nhsic_krr/nhsic_krr;

res.Rp_krr  = Rp_krr;
res.Rs_krr = Rs_krr;
res.acc_wc  = rmse_krr;
res.dep_wc  = nhsic_krr;
res.acc_wd  = rmse_pkl;
res.dep_wd  = nhsic_pkl;
res.sigma_q = sigma_q;
res.lambda  = best_lambda;
res.sigma   = best_sigma;
res.wc      = alpha_krr;
res.wd      = alpha_pkl;
res.epsilon = epsilon;
res.mus = mus;
res.lambdas = lambdas;
res.sigmas = sigmas;
res.sigmas_q = sigmas_q;

end