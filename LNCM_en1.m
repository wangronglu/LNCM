function [alpha0,alpha,rho,gamma,Beta,LF,dev]...
    = LNCM_en1(n,V,mW,mgW,mg2W,y,K,delta,eta,maxit,fullit,tol,alpha0_ini,...
    alpha_ini,rho_ini,gamma_ini,Beta_ini)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LNCM_en1 fits the longitudinal network classification model with
% elastic-net penalty given initial values.
% Prob(y_i = 1) = p_i
% logit(p_i) = alpha0 + \sum_{h=1}^K 1/T_i \sum_{s=1}^{T_i}(alpha_h +
%   rho_h*g_{is} + gamma_h*g_{is}^2 ) beta_h^T W_i^{(s)} beta_h
% Loss function = -1/n joint log-likelihood + elastic-net penalty
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
%   delta: overall penalty factor (>0)
%   eta: elastic-net factor, within (0,1); eta=1 -> lasso
%   maxit: maximum iterations (>=2),e.g. maxit=1000.
%   fullit: number of iterations that cycle all variables; after that the 
%           algorithm only updates active variables.
%           ( 2 <= fullit <= maxit, e.g. fullit = 100)
%   tol: tolerance of relative change in objective function,e.g. tol=1e-6.
%   alpha0_ini: initial value of alpha0 (scalar)
%   alpha_ini: Kx1 vector 
%   rho_ini: Kx1 vector
%   gamma_ini: Kx1 vector
%   Beta_ini: VxK matrix
%
% Output:
%   alpha0: intercept of regression (scalar)
%   alpha: Kx1 vector
%   rho: Kx1 vector
%   gamma: Kx1 vector
%   Beta: VxK matrix
%   LF: values of loss function across iterations
%   dev = -2/n joint log-likelihood
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha0 = alpha0_ini;
alpha = alpha_ini;
rho = rho_ini;
gamma = gamma_ini;
Beta = Beta_ini;

LTidx = tril(true(V),-1);
prob_min = 1e-10;
prob_max = 1 - prob_min;

% discard inactive components
act_beta = (Beta~=0);
act_beta = sum(act_beta)>1; % 1xK vector
act_lambda = (abs(alpha) + abs(rho) + abs(gamma))>0; % Kx1 vector
act_comp = act_beta' & act_lambda; % Kx1 vector
nnac = K - sum(act_comp); % number of non-active components
Beta(:,~act_comp) = zeros(V,nnac);
alpha(~act_comp) = zeros(nnac,1);
rho(~act_comp) = zeros(nnac,1);
gamma(~act_comp) = zeros(nnac,1);

% compute initial value of loss function ---------------------------
LF = zeros(maxit,1);

% compute p_i
bmWb = zeros(n,K);
bmgWb = zeros(n,K);
bmg2Wb = zeros(n,K);
bbt_sum = zeros(K,1);
bbt2_sum = zeros(K,1);

for h=1:K
    if act_comp(h)
        bbt_h = Beta(:,h) * Beta(:,h)'; % VxV
        bmWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mW,1),2));
        bmgWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mgW,1),2));
        bmg2Wb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mg2W,1),2));
        bbt_sum(h) = sum(abs(bbt_h(LTidx)));
        bbt2_sum(h) = sum(bbt_h(LTidx).^2);
    end
end

logit = alpha0 + bmWb * alpha + bmgWb * rho + bmg2Wb * gamma;
prob = 1./(1+exp(-logit));
prob = min(max(prob,prob_min),prob_max);

dev = -2/n * sum(y.*log(prob) + (1-y).*log(1-prob) );
LF(1) = dev/2 + delta * eta * bbt_sum'*( abs(alpha)+abs(rho)+abs(gamma) ) + ...
        0.5 * delta * (1 - eta) * bbt2_sum'* (alpha.^2 + rho.^2 + gamma.^2);

for iter = 2:fullit
    %% update Beta
    for h=1:K
        if act_comp(h)
            for u=1:V
                comp_u = alpha(h).*(squeeze(mW(u,:,:))'* Beta(:,h)) ...
                    + rho(h).*(squeeze(mgW(u,:,:))'* Beta(:,h)) ...
                    + gamma(h).*(squeeze(mg2W(u,:,:))'* Beta(:,h)); % nx1
                B = -2/n * sum((y-prob).* comp_u);
                A = 4/n * sum(prob.*(1-prob).* comp_u.^2);
                D1 = delta * eta *( abs(alpha(h))+abs(rho(h))+abs(gamma(h)) ) * ...
                    ( sum(abs(Beta(:,h))) - abs(Beta(u,h)) );
                D2 = delta * (1 - eta) *( alpha(h).^2 + rho(h).^2 + gamma(h).^2 ) * ...
                    ( sum(Beta(:,h).^2) - Beta(u,h)^2 );
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
                alpha(h) = 0;
                rho(h) = 0;
                gamma(h) = 0;
            end
        end
    end
    
    %% update alpha, rho, gamma
    for h=1:K
        if act_comp(h)
            bbt_h = Beta(:,h) * Beta(:,h)';
            bbt_sum(h) = sum(abs(bbt_h(LTidx)));
            bbt2_sum(h) = sum(bbt_h(LTidx).^2);
            D1 = delta * eta * bbt_sum(h);
            D2 = delta * (1 - eta) * bbt2_sum(h);
            
            % update alpha(h)
            bmWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mW,1),2));
            B = -1/n * sum((y - prob).* bmWb(:,h));
            A = 1/n * sum( prob.*(1 - prob).* bmWb(:,h).^2 );
            logit = logit - alpha(h) * bmWb(:,h);
            if ( (A+D2)>0 )                
                tmp1 = A * alpha(h) - B;
                tmp2 = abs(tmp1) - D1;
                if (tmp2 >0)
                    alpha(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                    logit = logit + alpha(h) * bmWb(:,h);
                    prob = 1./(1+exp(-logit));
                else
                    alpha(h) = 0;
                    prob = 1./(1+exp(-logit));
                end
            else % A+D2=0                
                alpha(h) = 0;
                prob = 1./(1+exp(-logit));                
            end
            
            % update rho(h)
            bmgWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mgW,1),2));
            B = -1/n * sum( (y - prob).* bmgWb(:,h) );
            A = 1/n * sum( prob.*(1-prob).* bmgWb(:,h).^2);
            % D1 does not change
            % D2 does not change
            logit = logit - rho(h) * bmgWb(:,h);
            if ( (A+D2)>0 )               
                tmp1 = A * rho(h) - B;
                tmp2 = abs(tmp1) - D1;
                if (tmp2 >0)
                    rho(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                    logit = logit + rho(h) * bmgWb(:,h);
                    prob = 1./(1+exp(-logit));
                else
                    rho(h) = 0;
                    prob = 1./(1+exp(-logit));
                end
            else % A+D2=0                
                rho(h) = 0;
                prob = 1./(1+exp(-logit));
            end
            
            % update gamma(h)
            bmg2Wb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mg2W,1),2));
            B = -1/n * sum( (y - prob).* bmg2Wb(:,h) );
            A = 1/n * sum( prob.*(1-prob).* bmg2Wb(:,h).^2);
            % D1 does not change
            % D2 does not change
            logit = logit - gamma(h) * bmg2Wb(:,h);
            if ( (A+D2)>0 )                
                tmp1 = A * gamma(h) - B;
                tmp2 = abs(tmp1) - D1;
                if (tmp2 >0)
                    gamma(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                    logit = logit + gamma(h) * bmg2Wb(:,h);
                    prob = 1./(1+exp(-logit));
                else
                    gamma(h) = 0;
                    prob = 1./(1+exp(-logit));
                end
            else % A+D2=0
                gamma(h) = 0;
                prob = 1./(1+exp(-logit));
            end
            
            % check empty
            if ( ( abs(alpha(h)) + abs(rho(h)) + abs(gamma(h)) ) == 0 )
                act_comp(h) = 0;
                Beta(:,h) = zeros(V,1);
            end                 
        end
    end
    
    %% update alpha0 (intercept)
    B = -1/n * sum(y - prob);
    A = 1/n * sum( prob.*(1-prob) );
    if (B~=0)
        if (A>0)
            alpha0 = alpha0 - sign(B) * exp( log(abs(B))-log(A));           
        else % A=0
            alpha0 = 0;
        end
    % else B=0
        % alpha0 = alpha0
    end
    
    %% stopping rule
    % recompute logit to avoid numerical error
    logit = alpha0 + bmWb * alpha + bmgWb * rho + bmg2Wb * gamma;
    prob = 1./(1+exp(-logit));
    prob = min(max(prob,prob_min),prob_max);
    
    dev = -2/n * sum(y.*log(prob) + (1-y).*log(1-prob) );
    LF(iter) = dev/2 + delta * eta * bbt_sum'*( abs(alpha)+abs(rho)+abs(gamma) ) + ...
        0.5 * delta * (1 - eta) * bbt2_sum'* (alpha.^2 + rho.^2 + gamma.^2);
    disp(iter)
    if ( ( LF(iter-1) - LF(iter) ) < tol * abs(LF(iter-1)) || isnan(LF(iter))  )
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
                        comp_u = alpha(h).*(squeeze(mW(u,:,:))'* Beta(:,h)) ...
                            + rho(h).*(squeeze(mgW(u,:,:))'* Beta(:,h)) ...
                            + gamma(h).*(squeeze(mg2W(u,:,:))'* Beta(:,h)); % nx1
                        B = -2/n * sum((y-prob).* comp_u);
                        A = 4/n * sum(prob.*(1-prob).* comp_u.^2);
                        D1 = delta * eta *( abs(alpha(h))+abs(rho(h))+abs(gamma(h)) ) * ...
                            ( sum(abs(Beta(:,h))) - abs(Beta(u,h)) );
                        D2 = delta * (1 - eta) *( alpha(h).^2 + rho(h).^2 + gamma(h).^2 ) * ...
                            ( Beta(:,h)'* Beta(:,h) - Beta(u,h)^2 );
                        logit = logit - 2 * Beta(u,h) * comp_u;
                        if ( (A+D2)>0 )
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
                    alpha(h) = 0;
                    rho(h) = 0;
                    gamma(h) = 0;
                end
            end
        end
         
        %% update alpha, rho, gamma
        for h=1:K
            if act_comp(h)
                bbt_h = Beta(:,h) * Beta(:,h)';
                bbt_sum(h) = sum(abs(bbt_h(LTidx)));
                bbt2_sum(h) = sum(bbt_h(LTidx).^2);
                D1 = delta * eta * bbt_sum(h);
                D2 = delta * (1 - eta) * bbt2_sum(h);
                
                % update alpha(h)
                if (alpha(h)~=0)
                    bmWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mW,1),2));
                    B = -1/n * sum((y - prob).* bmWb(:,h));
                    A = 1/n * sum( prob.*(1 - prob).* bmWb(:,h).^2 );
                    logit = logit - alpha(h) * bmWb(:,h);
                    if ( (A+D2)>0 )
                        tmp1 = A * alpha(h) - B;
                        tmp2 = abs(tmp1) - D1;
                        if (tmp2 >0)
                            alpha(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                            logit = logit + alpha(h) * bmWb(:,h);
                            prob = 1./(1+exp(-logit));
                        else
                            alpha(h) = 0;
                            prob = 1./(1+exp(-logit));
                        end
                    else % A+D2=0
                        alpha(h) = 0;
                        prob = 1./(1+exp(-logit));
                    end
                end
                
                % update rho(h)
                if (rho(h)~=0)
                    bmgWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mgW,1),2));
                    B = -1/n * sum( (y - prob).* bmgWb(:,h) );
                    A = 1/n * sum( prob.*(1-prob).* bmgWb(:,h).^2);
                    % D1 does not change
                    % D2 does not change
                    logit = logit - rho(h) * bmgWb(:,h);
                    if ( (A+D2)>0 )
                        tmp1 = A * rho(h) - B;
                        tmp2 = abs(tmp1) - D1;
                        if (tmp2 >0)
                            rho(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                            logit = logit + rho(h) * bmgWb(:,h);
                            prob = 1./(1+exp(-logit));
                        else
                            rho(h) = 0;
                            prob = 1./(1+exp(-logit));
                        end
                    else % A+D2=0
                        rho(h) = 0;
                        prob = 1./(1+exp(-logit));
                    end
                end
                
                % update gamma(h)
                if (gamma(h)~=0)
                    bmg2Wb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* mg2W,1),2));
                    B = -1/n * sum( (y - prob).* bmg2Wb(:,h) );
                    A = 1/n * sum( prob.*(1-prob).* bmg2Wb(:,h).^2);
                    % D1 does not change
                    % D2 does not change
                    logit = logit - gamma(h) * bmg2Wb(:,h);
                    if ( (A+D2)>0 )
                        tmp1 = A * gamma(h) - B;
                        tmp2 = abs(tmp1) - D1;
                        if (tmp2 >0)
                            gamma(h) = sign(tmp1) * exp( log(tmp2) - log(A+D2) );
                            logit = logit + gamma(h) * bmg2Wb(:,h);
                            prob = 1./(1+exp(-logit));
                        else
                            gamma(h) = 0;
                            prob = 1./(1+exp(-logit));
                        end
                    else % A+D2=0
                        gamma(h) = 0;
                        prob = 1./(1+exp(-logit));
                    end
                end
                
                % check empty
                if ( ( abs(alpha(h)) + abs(rho(h)) + abs(gamma(h)) ) == 0 )
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                end
            end
        end
        
       %% update alpha0 (intercept)
        B = -1/n * sum(y - prob);
        A = 1/n * sum( prob.*(1-prob) );
        if (B~=0)
            if (A>0)
                alpha0 = alpha0 - sign(B) * exp( log(abs(B))-log(A));
            else % A=0
                alpha0 = 0;
            end
        % else B=0
            % alpha0 = alpha0
        end
        
        %% stopping rule
        % recompute logit to avoid numerical error
        logit = alpha0 + bmWb * alpha + bmgWb * rho + bmg2Wb * gamma;
        prob = 1./(1+exp(-logit));
        prob = min(max(prob,prob_min),prob_max);
        
        dev = -2/n * sum(y.*log(prob) + (1-y).*log(1-prob) );
        LF(iter) = dev/2 + delta * eta * bbt_sum'*( abs(alpha)+abs(rho)+abs(gamma) ) + ...
            0.5 * delta * (1 - eta) * bbt2_sum'* (alpha.^2 + rho.^2 + gamma.^2);
        disp(iter)
        if ( ( LF(iter-1) - LF(iter) ) < tol * abs(LF(iter-1)) || isnan(LF(iter))  )
            break
        end
    end
end

LF = LF(1:iter);