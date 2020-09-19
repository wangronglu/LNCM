function [alpha_final,Lambda_final,Beta_final,coefM_final,coefM_final_vec,min_dev_vali,delta_seq,dev_set] = ...
    SBLR_tuning_delta(W_train,X_train,y_train,W_vali,X_vali,y_vali,...
    K,eta,nreps,ndt,delta_min_ratio,maxit,fullit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SBLR_tuning_delta.m automatically find delta_max that penalizes all the
% coefficients to zero, and select the optimal delta value producing the
% smallest deviance on validation set. It returns the estimated coefficients 
% under the optimal delta.
%
% Input:
%   W_train: VxVxn_train, standardized adjacency matrix for each subject in training set.
%   X_train: n_train x m matrix, design matrix of regular covariates where the first 
%      column is all 1 corresponding to the intercept.
%   y_train: n_train x 1 vector, binary response of each subject in training set.
%   W_vali: VxVxn_vali, standardized adjacency matrix for each subject in validation set.
%   X_vali: n_vali x m matrix, design matrix of regular covariates where the first 
%      column is all 1 corresponding to the intercept.
%   y_vali: n_vali x 1 vector, binary response of each subject in validation set.
%   K: number of components in SBLR
%   eta: L1 fractional penalty factor, within (0,1]; eta=1 -> lasso
%   nreps: number of random initializations
%   ndt: number of delta values
%   delta_min_ratio: delta_min = delta_min_ratio * delta_max
%   maxit: maximum iterations (>=2),e.g. maxit=1000.
%   fullit: number of iterations that cycle all variables; after that the 
%           algorithm only updates active variables.
%           ( 2 <= fullit <= maxit, e.g. fullit = 100)
%   tol: tolerance of relative change in objective function,e.g. tol=1e-6.
%
% Output:
%   alpha_final: mx1 vector (1st entry is intercept), estimate of alpha
%                under the optimal delta
%   Lambda_final: Kx1 vector, estimate of Lambda under the optimal delta
%   Beta_final: VxK matrix, estimate of Beta under the optimal delta
%   coefM_final: VxV matrix, Beta_final * diag(Lambda_final) * Beta_final'
%   coefM_final_vec: 2 x upper triangular of coefM_final 
%   min_dev_vali: minimum deviance on validation set
%   delta_seq: delta sequence for tuning
%   dev_set: set of deviances on validation set across delta values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_train = length(y_train);
n_vali = length(y_vali);
V = size(W_train,1);
m = size(X_train,2);

%% set sequence of delta values
% choose uppper bound ----------
delta = 1;

for rep = 1:nreps
    % initialization
    rng(rep-1, 'twister')
    Beta_ini = 0.2 * rand(V,K) - 0.1;
    Lambda_ini = 0.2 * rand(K,1) - 0.1;
    alpha_ini = zeros(m,1);
    
    [~,~,Beta]...
        = SBLR_en1(n_train,V,m,W_train,X_train,y_train,K,delta,eta,maxit,fullit,...
        tol,alpha_ini,Lambda_ini,Beta_ini);
    
    if nnz(Beta)>0
        break
    end
end

while( nnz(Beta)>0 )
    delta = 2 * delta;
    
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 0.2 * rand(V,K) - 0.1;
        Lambda_ini = 0.2 * rand(K,1) - 0.1;
        alpha_ini = zeros(m,1);
        
        [~,~,Beta]...
            = SBLR_en1(n_train,V,m,W_train,X_train,y_train,K,delta,eta,maxit,...
            fullit,tol,alpha_ini,Lambda_ini,Beta_ini);
        
        if nnz(Beta)>0
            break
        end
    end
end

while( nnz(Beta)==0 )
    delta = 0.5 * delta;
    
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 0.2 * rand(V,K) - 0.1;
        Lambda_ini = 0.2 * rand(K,1) - 0.1;
        alpha_ini = zeros(m,1);
        
        [~,~,Beta]...
            = SBLR_en1(n_train,V,m,W_train,X_train,y_train,K,delta,eta,maxit,...
            fullit,tol,alpha_ini,Lambda_ini,Beta_ini);
        
        if nnz(Beta)>0
            break
        end
    end
    
end

delta_max = 2 * delta;

% set lower bound 
delta_min = delta_min_ratio * delta_max;

% set delta sequence 
delta_seq = exp(linspace(log(delta_min), log(delta_max), ndt));

%% Validation tuning for L1 penalty
dev_set = zeros(ndt,1);
prob_min = 1e-5;
prob_max = 1 - prob_min;

for j=1:ndt
    disp(['j=',num2str(j)])
    
    delta = delta_seq(j);
    
    LFmin = Inf;
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 0.2 * rand(V,K) - 0.1;
        Lambda_ini = 0.2 * rand(K,1) - 0.1;
        alpha_ini = zeros(m,1);
        
        [alpha_cand,Lambda_cand,Beta_cand,LF_cand]...
            = SBLR_en1(n_train,V,m,W_train,X_train,y_train,K,delta,eta,...
            maxit,fullit,tol,alpha_ini,Lambda_ini,Beta_ini);
        
        if (LF_cand(end) < LFmin)
            LFmin = LF_cand(end);
            alpha = alpha_cand;
            Beta = Beta_cand;
            Lambda = Lambda_cand;
        end
    end
    
    % compute deviance
    coefM = (Beta.* repmat(Lambda',[V,1]))* Beta'; % V x V
    logit = X_vali * alpha + squeeze(sum(sum(repmat(coefM,[1,1,n_vali]).* W_vali,1),2));
    prob = 1./(1+exp(-logit));
    prob = min(max(prob,prob_min),prob_max);
    dev_set(j) = -2/n_vali * sum(y_vali.*log(prob) + (1-y_vali).*log(1-prob) );
end

%% select optimal tuning parameter
[min_dev_vali,ind_opt] = min(dev_set); 
delta_opt = delta_seq(ind_opt);

% estimate model at optimal penalty factor with training data
LFmin = Inf;
for rep = 1:nreps
    % initialization
    rng(rep-1, 'twister')
    Beta_ini = 0.2 * rand(V,K) - 0.1;
    Lambda_ini = 0.2 * rand(K,1) - 0.1;
    alpha_ini = zeros(m,1);
    
    [alpha_cand,Lambda_cand,Beta_cand,LF_cand]...
        = SBLR_en1(n_train,V,m,W_train,X_train,y_train,K,delta_opt,eta,...
        maxit,fullit,tol,alpha_ini,Lambda_ini,Beta_ini);
    
    if (LF_cand(end) < LFmin)
        LFmin = LF_cand(end);
        alpha = alpha_cand;
        Beta = Beta_cand;
        Lambda = Lambda_cand;
        % disp(['rep=',num2str(rep)])
    end
end

% estimate model at selected penalty factors with full data ----------
W_s = cat(3,W_train,W_vali);
X = [X_train; X_vali];
y = [y_train; y_vali];
n = length(y);

clear W_train W_vali

% use estimates from training data under optimal penalty factor as initial
% value
[alpha_final,Lambda_final,Beta_final]...
    = SBLR_en1(n,V,m,W_s,X,y,K,delta_opt,eta,...
    maxit,fullit,tol,alpha,Lambda,Beta);

coefM_final = (Beta_final.* repmat(Lambda_final',[V,1]))* Beta_final';

% extract upper-triangular part
UTidx = triu(true(V),1); % excluding diagonal
coefM_final_vec = 2 * coefM_final(UTidx);


