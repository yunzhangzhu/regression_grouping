function [L U] = factor(X, D)
    [n, p] = size(X);
    if ( n >= p )    % if skinny
       L = chol( X'*X + D, 'lower' );
    else            % if fat
        d = 1./diag(D);
        L = chol( speye(n) + (bsxfun(@times,X,d')) * X', 'lower');
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end