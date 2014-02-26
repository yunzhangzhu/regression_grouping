% solves minimize 1/(2n)*|Y-X*beta|^2+\lambda_1*|beta| + 
% \lambda_2*sum_{(j,j') \in E} |beta_j*w_j-beta_j'*w_j'|

function [beta, history] = grouping_convex(X,Y,lambda1,lambda2,E,wts,rho,alpha)
t_start = tic;

MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-3;

[n,p] = size(X);
g = size(E,2);
X = X ./ sqrt(n);
Y = Y ./ sqrt(n);
XtY = X'*Y;

QUIET = 0;

if ~QUIET
    fprintf('solving a regression problem with %d predictors and %d edges \n', p, g);
end

L1 = sparse(1:g,E(1,:),wts(E(1,:)),g,p);
L2 = sparse(1:g,E(2,:),wts(E(2,:)),g,p);
D = rho.*(speye(p) + 2*(L1'*L1 + L2'*L2)); 

% caching the factors
[L U] = factor(X,D);

if ~QUIET
    fprintf('%3s\t%10s\n', 'iter','objective');
end

alpha1 = zeros(p,1);
alpha2 = zeros(g,1);
alpha3 = zeros(g,1);
u1 = zeros(p,1);
u2 = zeros(g,1);
u3 = zeros(g,1);

% ADMM iteration starts  

for k = 1:MAX_ITER
  % beta update
  q = XtY+rho*(alpha1-u1)+L1'*(rho*(alpha2+alpha3-u2-u3))...
  -L2'*(rho*(alpha2-alpha3-u2+u3));
  
  if (n >= p)
    beta = U \ (L \ q);
  else
    d = 1./diag(D);
    beta = d.*q - d.*(X' * (U \ (L \ (X * (d.*q)))));
  end
  
  L1beta = L1 * beta;
  L2beta = L2 * beta;
  
  % alpha update
  alpha1_old = alpha1;
  alpha2_old = alpha2;
  alpha3_old = alpha3;
            
  %relaxation update:
  beta1_hat = alpha*beta + (1-alpha)*alpha1_old;
  F_beta2_hat = alpha*(L1beta-L2beta)+(1-alpha)*alpha2_old;
  F_beta3_hat = alpha*(L1beta+L2beta)+(1-alpha)*alpha3_old;
            
  alpha1 = shrinkage(beta1_hat + u1,lambda1/rho);
  alpha2 = shrinkage(F_beta2_hat + u2,lambda2/rho);
  alpha3 = F_beta3_hat + u3;
            
  % u udpate
  u1 = u1 + beta1_hat - alpha1;
  u2 = u2 + F_beta2_hat - alpha2;
  u3 = u3 + F_beta3_hat - alpha3;
  
  % termination check
  r_norm = sqrt(norm(beta-alpha1)^2+...
  norm(L1beta-L2beta-alpha2)^2+...
  norm(L1beta+L2beta-alpha3)^2);
  s_norm = rho*sqrt(norm(alpha1_old - alpha1)^2+...
  norm(L1'*(alpha2_old-alpha2)-L2'*(alpha2_old-alpha2))^2+...
  norm(L1'*(alpha3_old-alpha3)+L2'*(alpha3_old-alpha3))^2);
  eps_pri = sqrt(p+2*g)*ABSTOL+...
  RELTOL*max(sqrt(norm(beta)^2+...
  norm(L1beta-L2beta)^2+...
  norm(L1beta+L2beta)^2),...
  sqrt(norm(alpha1)^2+...
  norm(alpha2)^2+norm(alpha3)^2));
  eps_dual = sqrt(p+2*g)*ABSTOL+...
  RELTOL*rho*sqrt(norm(u1)^2+norm(L1'*u2-L2'*u2)^2+...
  norm(L1'*u3+L2'*u3)^2);
  
  if ~QUIET
    % save the objective and reporting
    history.objective(k) = objective(X, Y, L1, L2,...
      sparse(alpha1),lambda1,lambda2);
    fprintf('%3d\t%10.2f\n', k, history.objective(k));
  end
  
  % termination check
  if ((r_norm < eps_pri) && (s_norm < eps_dual))
    beta = sparse(alpha1);
    % grouping equal components %
    index1 = find(alpha2==0);
    if (size(index1,1) ~= 0)
        temp = find(alpha2==0);
        temp(temp > g) = temp(temp > g) - g;
        ee = E(:,temp);
        ff = sparse([ee(1,:),ee(2,:)],[ee(2,:),ee(1,:)],ones(2*size(ee,2),1),p,p);
        [S, C] = graphconncomp(ff,'DIRECTED',false);
        for i = 1:S
            index2 = find(C == i);
            if (beta(index2(1)) ~= 0)
                beta(index2) = sign(beta(index2)).*mean(abs(beta(index2)));
            end
        end
    end
    break;
  end
end
if ~QUIET 
    toc(t_start);
end
end