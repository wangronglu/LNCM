function [delta_seq,mean_dev_set,std_dev_set,mean_AUC_set,std_AUC_set,...
    alpha0_set,alpha_set,rho_set,gamma_set,Beta_set,df_set] ...
    = LNCM_CV_tuning_delta(n,V,mW,mgW,mg2W,y,K,eta,maxit,fullit,tol,...
    nreps,ndt,delta_min_ratio,nfolds,foldid)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LNCM_CV_tuning_delta.m automatically find delta_max that penalizes all the
% coefficients to zero, and tune delta using n-fold cross validation.
% Consider 2 selection criteria: minimum CV error (deviance) and one-standard-error rule.
%
% Input:
%   n: number of subjects
%   V: number of nodes in the network
%   mW: VxVxn array, average adjacency matrix for each subject
%   mgW: VxVxn array, age-weighted average adjacency matrix for each subject
%   mg2W: VxVxn array, age-squared weighted average adjacency matrix for
%       each subject
%   y: nx1 vector, binary response of each subject
%   K: rank of the logistic regression
%   eta: elastic-net factor, within (0,1); eta=1 -> lasso
%   maxit: maximum iterations (>=2),e.g. maxit=1000.
%   fullit: number of iterations that cycle all variables; after that the 
%           algorithm only updates active variables.
%           ( 2 <= fullit <= maxit, e.g. fullit = 100)
%   tol: tolerance of relative change in objective function,e.g. tol=1e-6.
%   nreps: number of random initializations
%   ndt: number of delta values
%   delta_min_ratio: delta_min = delta_min_ratio * delta_max
%   nfolds: number of folds in CV
%   foldid: nx1 vector of values between 1 and nfolds
%
% Output:
%   delta_seq: 1xndt, delta sequence for tuning
%   mean_dev_set: ndtx1, mean CV deviance over delta values
%   std_dev_set: ndtx1, std of CV deviance over delta values
%   mean_AUC_set: ndtx1, mean CV AUC over delta values
%   std_AUC_set: ndtx1, std of CV AUC over delta values
%   alpha0_set: ndtx1, estimate of alpha0 at each delta
%   alpha_set: Kxndt, estimate of alpha at each delta
%   rho_set: Kxndt, estimate of rho at each delta
%   gamma_set: Kxndt, estimate of gamma at each delta
%   Beta_set: VxKxndt, estimate of Beta at each delta
%   df_set: ndtx1, model degree of freedom at each delta
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set sequence of delta values
% choose uppper bound ----------
delta = 1;

for rep = 1:nreps
    % initialization
    rng(rep-1, 'twister')
    Beta_ini = 0.2 * rand(V,K) - 0.1;
    alpha_ini = 0.2 * rand(K,1) - 0.1;
    rho_ini = 0.2 * rand(K,1) - 0.1;
    gamma_ini = 0.2 * rand(K,1) - 0.1;
    alpha0_ini = 0;
    
    [~,~,~,~,Beta]...
        = LNCM_en1(n,V,mW,mgW,mg2W,y,K,delta,eta,maxit,fullit,tol,alpha0_ini,...
        alpha_ini,rho_ini,gamma_ini,Beta_ini);
    
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
        alpha_ini = 0.2 * rand(K,1) - 0.1;
        rho_ini = 0.2 * rand(K,1) - 0.1;
        gamma_ini = 0.2 * rand(K,1) - 0.1;
        alpha0_ini = 0;
        
        [~,~,~,~,Beta]...
            = LNCM_en1(n,V,mW,mgW,mg2W,y,K,delta,eta,maxit,fullit,tol,alpha0_ini,...
            alpha_ini,rho_ini,gamma_ini,Beta_ini);
        
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
        alpha_ini = 0.2 * rand(K,1) - 0.1;
        rho_ini = 0.2 * rand(K,1) - 0.1;
        gamma_ini = 0.2 * rand(K,1) - 0.1;
        alpha0_ini = 0;
        
        [~,~,~,~,Beta]...
            = LNCM_en1(n,V,mW,mgW,mg2W,y,K,delta,eta,maxit,fullit,tol,alpha0_ini,...
            alpha_ini,rho_ini,gamma_ini,Beta_ini);
        
        if nnz(Beta)>0
            break
        end
    end
end

delta_max = 2 * delta;

% set lower bound 
delta_min = delta_min_ratio * delta_max;

% set delta sequence 
delta_seq = exp(linspace(log(delta_min), log(delta_max), ndt)); % 1 x ndt

%% cross validation
dev_set = zeros(ndt,nfolds);
AUC_set = zeros(ndt,nfolds);

alpha0_set = zeros(ndt,1);
alpha_set = zeros(K,ndt);
rho_set = zeros(K,ndt);
gamma_set = zeros(K,ndt);
Beta_set = zeros(V,K,ndt);

df_set = zeros(ndt,1);

prob_min = 1e-10;
prob_max = 1 - prob_min;

for j = 1:ndt
    % disp(['j=',num2str(j)])
    
    delta = delta_seq(j);
    
    for nf = 1:nfolds
        % disp(['nf= ',num2str(nf)])
        
        mW_test = mW(:,:,foldid == nf);
        mgW_test = mgW(:,:,foldid == nf);
        mg2W_test = mg2W(:,:,foldid == nf);
        y_test = y(foldid == nf);
        n_test = length(y_test);
        
        mW_train = mW(:,:,foldid ~= nf);
        mgW_train = mgW(:,:,foldid ~= nf);
        mg2W_train = mg2W(:,:,foldid ~= nf);
        y_train = y(foldid ~= nf);
        n_train = length(y_train);
        
        % train model
        LFmin = Inf;
        for rep = 1:nreps
            % initialization
            rng(rep-1, 'twister')
            Beta_ini = 0.2 * rand(V,K) - 0.1;
            alpha_ini = 0.2 * rand(K,1) - 0.1;
            rho_ini = 0.2 * rand(K,1) - 0.1;
            gamma_ini = 0.2 * rand(K,1) - 0.1;
            alpha0_ini = 0;
            
            [alpha0_cand,alpha_cand,rho_cand,gamma_cand,Beta_cand,LF_cand]...
                = LNCM_en1(n_train,V,mW_train,mgW_train,mg2W_train,y_train,K,delta,eta,...
                maxit,fullit,tol,alpha0_ini,alpha_ini,rho_ini,gamma_ini,Beta_ini);
            
            if (LF_cand(end) < LFmin)
                LFmin = LF_cand(end);
                Beta = Beta_cand;
                alpha = alpha_cand;
                rho = rho_cand;
                gamma = gamma_cand;
                alpha0 = alpha0_cand;
            end
        end
        
        % compute deviance and AUC on test set -------------------------
        bmWb = zeros(n_test,K);
        bmgWb = zeros(n_test,K);
        bmg2Wb = zeros(n_test,K);
        
        for h=1:K
            bbt_h = Beta(:,h) * Beta(:,h)'; % VxV
            bmWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n_test]).* mW_test,1),2));
            bmgWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n_test]).* mgW_test,1),2));
            bmg2Wb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n_test]).* mg2W_test,1),2));
        end
        
        logit = alpha0 + bmWb * alpha + bmgWb * rho + bmg2Wb * gamma;
        prob = 1./(1+exp(-logit));
        
        % AUC
        [~,~,~,auc] = perfcurve(logical(y_test),prob,'true');
        AUC_set(j,nf) = auc;
        
        % deviance
        prob = min(max(prob,prob_min),prob_max);
        dev_set(j,nf) = -2/n_test * sum(y_test.*log(prob) + (1-y_test).*log(1-prob) );
    end
    
    % estimate model with full data
    LFmin = Inf;
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 0.2 * rand(V,K) - 0.1;
        alpha_ini = 0.2 * rand(K,1) - 0.1;
        rho_ini = 0.2 * rand(K,1) - 0.1;
        gamma_ini = 0.2 * rand(K,1) - 0.1;
        alpha0_ini = 0;
        
        [alpha0_cand,alpha_cand,rho_cand,gamma_cand,Beta_cand,LF_cand]...
            = LNCM_en1(n,V,mW,mgW,mg2W,y,K,delta,eta,maxit,fullit,tol,alpha0_ini,...
            alpha_ini,rho_ini,gamma_ini,Beta_ini);
        
        if (LF_cand(end) < LFmin)
            LFmin = LF_cand(end);
            Beta = Beta_cand;
            alpha = alpha_cand;
            rho = rho_cand;
            gamma = gamma_cand;
            alpha0 = alpha0_cand;
        end
    end
    
    % degree of freedom
    df_set(j) = nnz(Beta) + nnz(alpha) + nnz(rho) + nnz(gamma);
    
    Beta_set(:,:,j) = Beta;
    alpha_set(:,j) = alpha;
    rho_set(:,j) = rho;
    gamma_set(:,j) = gamma;
    alpha0_set(j) = alpha0;
end

mean_dev_set = mean(dev_set,2); % ndt x 1
std_dev_set = std(dev_set,0,2); % ndt x 1

mean_AUC_set = mean(AUC_set,2); % ndt x 1
std_AUC_set = std(AUC_set,0,2); % ndt x 1