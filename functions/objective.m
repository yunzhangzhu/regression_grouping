function obj = objective(X, Y, L1, L2, beta,lambda1,lambda2)
    n = size(X,1);
    obj = ( .5*sum((X*beta - Y).^2) + lambda1*norm(beta,1)...
        + lambda2*norm(L1*beta-L2*beta,1) );
end