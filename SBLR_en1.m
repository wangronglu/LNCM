function [alpha,Lambda,Beta,LF]...
    = SBLR_en1(n,V,m,W,X,y,K,delta,eta,maxit,fullit,tol,alpha_ini,Lambda_ini,Beta_ini)
    
% SBLR_en1 fits the symmetric rank-K logistic bilinear regression with
% elastic-net penalty given initial values. 
% Prob(y_i = 1) = p_i
% logit(p_i) = alpha + \sum_{h=1}^K lambda_h beta_h^T W_i beta_h
% Loss function = -1/n joint log-likelihood + elastic-net penalty
%
% Input:
%   n: the number of subjects
%   V: the number of nodes in the network 
%   m: the number of covariates + 1 (intercept)
%   W: VxVxn array, standardized adjacency matrix for each subject;   
%   X: nxm matrix, design matrix of regular covariates where the first 
%      column is all 1 corresponding to the intercept.
%   y: nx1 vector, binary response of each subject
%   K: number of components in SBLR
%   delta: overall penalty factor (>0)
%   eta: L1 fractional penalty factor, within (0,1]; eta=1 -> lasso
%   maxit: maximum iterations (>=2),e.g. maxit=1000.
%   fullit: number of iterations that cycle all variables; after that the 
%           algorithm only updates active variables.
%           ( 2 <= fullit <= maxit, e.g. fullit = 100)
%   tol: tolerance of relative change in objective function,e.g. tol=1e-6.
%   alpha_ini: initial value of alpha (intercept scalar)
%   Lambda_ini: Kx1 vector 
%   Beta_ini: VxK matrix
%
% Output:
%   alpha: mx1 vector (1st entry is intercept)
%   Lambda: Kx1 vector
%   Beta: VxK matrix
%   LF: values of loss function across iterations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = alpha_ini;
Lambda = Lambda_ini;
Beta = Beta_ini;

LTidx = tril(true(V),-1);
prob_min = 1e-10;
prob_max = 1 - prob_min;

% discard inactive components
act_beta = (Beta~=0);
act_beta = sum(act_beta)>1; % 1xK vector
act_lambda = (Lambda~=0); % Kx1 vector
act_comp = act_beta' & act_lambda; % Kx1 vector
nnac = K - sum(act_comp); % number of non-active components
Beta(:,~act_comp) = zeros(V,nnac);
Lambda(~act_comp) = zeros(nnac,1);

% compute initial value of loss function
LF = zeros(maxit,1);

% compute p_i
bWb = zeros(n,K);
bbt_sum = zeros(K,1);
bbt2_sum = zeros(K,1);

for h=1:K
    if act_comp(h)
        bbt_h = Beta(:,h) * Beta(:,h)';
        bWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* W,1),2)); 
        bbt_sum(h) = sum(abs(bbt_h(LTidx)));
        bbt2_sum(h) = sum(bbt_h(LTidx).^2);
    end
end

logit = X * alpha + bWb * Lambda;
prob = 1./(1+exp(-logit));
prob = min(max(prob,prob_min),prob_max);

LF(1) = -1/n * sum(y.*log(prob) + (1-y).*log(1-prob) ) + ...
        delta * eta * bbt_sum'* abs(Lambda) + ...
        0.5 * delta * (1 - eta) * bbt2_sum'* Lambda.^2;
    
for iter = 2:fullit
    %% update Beta
    for h=1:K
        if act_comp(h)
            for u=1:V
                comp_u = Lambda(h) * squeeze(W(u,:,:))'* Beta(:,h); % nx1
                B = -2/n * sum((y-prob).* comp_u);
                A = 4/n * sum(prob.*(1-prob).* comp_u.^2);
                D1 = delta * eta * abs(Lambda(h)) * ( sum(abs(Beta(:,h))) - abs(Beta(u,h)) );
                D2 = delta * (1 - eta) * Lambda(h).^2 * ( Beta(:,h)'* Beta(:,h) - Beta(u,h)^2 );
                logit = logit - 2 * Beta(u,h) * comp_u;
                if ( (A + D2)>0 )
                    tmp1 = A * Beta(u,h)-B;
                    tmp2 = abs(tmp1) - D1;
                    if (tmp2 >0)
                        Beta(u,h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                        logit = logit + 2 * Beta(u,h) * comp_u;
                        prob = 1./(1+exp(-logit));
                    else
                        Beta(u,h) = 0;
                        prob = 1./(1+exp(-logit));
                    end
                else % A+D2==0
                    Beta(u,h) = 0;
                    prob = 1./(1+exp(-logit));
                end
            end
            % check empty
            if ( sum( Beta(:,h)~=0 ) < 2 )
                act_comp(h) = 0;
                Beta(:,h) = zeros(V,1);
                Lambda(h) = 0;
            end
        end
    end
    
    %% update Lambda
    for h=1:K
        if act_comp(h)
            bbt_h = Beta(:,h) * Beta(:,h)';
            bbt_sum(h) = sum(abs(bbt_h(LTidx)));
            bbt2_sum(h) = sum(bbt_h(LTidx).^2);
            D1 = delta * eta * bbt_sum(h);
            D2 = delta * (1 - eta) * bbt2_sum(h);
            
            bWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* W,1),2));
            B = -1/n * sum((y - prob).* bWb(:,h));
            A = 1/n * sum( prob.*(1 - prob).* bWb(:,h).^2 );
            logit = logit - Lambda(h) * bWb(:,h);
            if ( (A+D2)>0 )
                tmp1 = A * Lambda(h) - B;
                tmp2 = abs(tmp1) - D1;
                if (tmp2 >0)
                    Lambda(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                    logit = logit + Lambda(h) * bWb(:,h);
                    prob = 1./(1+exp(-logit));
                else
                    Lambda(h) = 0;
                    prob = 1./(1+exp(-logit));
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                end
            else % A+D2=0
                Lambda(h) = 0;
                prob = 1./(1+exp(-logit));
                act_comp(h) = 0;
                Beta(:,h) = zeros(V,1);
            end
        end
    end
    
    %% update alpha 
    B = -1/n * sum( repmat(y - prob,[1,m]).*X )'; % mx1
    A = (repmat(prob,[1,m]).*X)' * (repmat(1 - prob,[1,m]).*X)/n; % mxm
    if (B'* B~=0)
        if (m==1)
            if (A>0)
                alpha = alpha - sign(B) * exp( log(abs(B))-log(A));
            else % A=0
                alpha = 0;
            end
        else
            if (det(A)>0)
                alpha = alpha - A\B;
            else
                alpha = zeros(m,1);
            end
        end
   % else B=0
        % alpha = alpha
    end
    
    %% stopping rule
    % recompute logit to avoid numerical error
    logit = X * alpha + bWb * Lambda;
    prob = 1./(1+exp(-logit));
    prob = min(max(prob,prob_min),prob_max);
    
    LF(iter) = -1/n * sum(y.*log(prob) + (1-y).*log(1-prob) ) + ...
        delta * eta * bbt_sum'* abs(Lambda) + ...
        0.5 * delta * (1 - eta) * bbt2_sum'* Lambda.^2;
    
    % disp(iter)
    
    if ( ( LF(iter-1) - LF(iter) ) < tol * abs(LF(iter-1)) || isnan(LF(iter)) )
        break
    end
end

%% only update nonzero parameters
if (iter==fullit) && (fullit < maxit)
    for iter = fullit+1 : maxit
        %% update Beta
        for h=1:K
            if act_comp(h)
                for u=1:V
                    if (Beta(u,h) ~= 0)
                        comp_u = Lambda(h) * squeeze(W(u,:,:))'* Beta(:,h); % nx1
                        B = -2/n * sum((y-prob).* comp_u);
                        A = 4/n * sum(prob.*(1-prob).* comp_u.^2);
                        D1 = delta * eta * abs(Lambda(h)) * ( sum(abs(Beta(:,h))) - abs(Beta(u,h)) );
                        D2 = delta * (1 - eta) * Lambda(h).^2 * ( Beta(:,h)'* Beta(:,h) - Beta(u,h)^2 );
                        logit = logit - 2 * Beta(u,h) * comp_u;
                        if ( (A + D2)>0 )
                            tmp1 = A * Beta(u,h)-B;
                            tmp2 = abs(tmp1) - D1;
                            if (tmp2 >0)
                                Beta(u,h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                                logit = logit + 2 * Beta(u,h) * comp_u;
                                prob = 1./(1+exp(-logit));
                            else
                                Beta(u,h) = 0;
                                prob = 1./(1+exp(-logit));
                            end
                        else % A+D2==0
                            Beta(u,h) = 0;
                            prob = 1./(1+exp(-logit));
                        end
                    end
                end
                % check empty
                if ( sum( Beta(:,h)~=0 ) < 2 )
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                    Lambda(h) = 0;
                end
            end
        end
         
        %% update Lambda
        for h=1:K
            if act_comp(h)
                bbt_h = Beta(:,h) * Beta(:,h)';
                bbt_sum(h) = sum(abs(bbt_h(LTidx)));
                bbt2_sum(h) = sum(bbt_h(LTidx).^2);
                D1 = delta * eta * bbt_sum(h);
                D2 = delta * (1 - eta) * bbt2_sum(h);
                
                bWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* W,1),2));
                B = -1/n * sum((y - prob).* bWb(:,h));
                A = 1/n * sum( prob.*(1 - prob).* bWb(:,h).^2 );
                logit = logit - Lambda(h) * bWb(:,h);
                if ( (A+D2)>0 )
                    tmp1 = A * Lambda(h) - B;
                    tmp2 = abs(tmp1) - D1;
                    if (tmp2 >0)
                        Lambda(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                        logit = logit + Lambda(h) * bWb(:,h);
                        prob = 1./(1+exp(-logit));
                    else
                        Lambda(h) = 0;
                        prob = 1./(1+exp(-logit));
                        act_comp(h) = 0;
                        Beta(:,h) = zeros(V,1);
                    end
                else % A+D2=0
                    Lambda(h) = 0;
                    prob = 1./(1+exp(-logit));
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                end
            end
        end
        
       %% update alpha 
       B = -1/n * sum( repmat(y - prob,[1,m]).*X )'; % mx1
       A = (repmat(prob,[1,m]).*X)' * (repmat(1 - prob,[1,m]).*X)/n; % mxm
       if (B'* B~=0)
           if (m==1)
               if (A>0)
                   alpha = alpha - sign(B) * exp( log(abs(B))-log(A));
               else % A=0
                   alpha = 0;
               end
           else
               if (det(A)>0)
                   alpha = alpha - A\B;
               else
                   alpha = zeros(m,1);
               end
           end
      % else B=0
           % alpha = alpha
       end
    
        
       %% stopping rule
       % recompute logit to avoid numerical error
       logit = X * alpha + bWb * Lambda;
       prob = 1./(1+exp(-logit));
       prob = min(max(prob,prob_min),prob_max);
       
       LF(iter) = -1/n * sum(y.*log(prob) + (1-y).*log(1-prob) ) + ...
           delta * eta * bbt_sum'* abs(Lambda) + ...
           0.5 * delta * (1 - eta) * bbt2_sum'* Lambda.^2;
       
       % disp(iter)
       
       if ( ( LF(iter-1) - LF(iter) ) < tol * abs(LF(iter-1)) || isnan(LF(iter)) )
           break
       end
    end
end

LF = LF(1:iter);