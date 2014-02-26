% demo for solving high dimensional
% regression with lasso and grouping penalty

clear all;
clc
addpath('functions/');
rand('state',0); randn('state',0);
num_of_err_edge = 1;
num_of_active_vars = 10;
g1 = num_of_active_vars * num_of_err_edge + num_of_active_vars - 1;

% generating training data
n = 1000;
p = 20000;
cor = .9;
err_var = 1;
X = randn(n,p);
beta0 = zeros(p,1);
beta0(1:num_of_active_vars) = 3;
for i = 2:num_of_active_vars
    X(:,i) = sqrt(cor)*X(:,1) + sqrt(1-cor)*X(:,i);
end
Y = X * beta0 + sqrt(err_var) * randn(n, 1);

% Note E is 2 x g matrix.  
J = [2:num_of_active_vars,randsample((num_of_active_vars+1):p,...
    num_of_active_vars * num_of_err_edge,true)];
I = ones(1,g1);
I(num_of_active_vars:end) = repmat(1:num_of_active_vars,...
    num_of_err_edge,1);
E1 = [I;J];

num_extra_edges = 19980;
E2 = [11:(11+num_extra_edges);(12:(12+num_extra_edges))];
E = [E1,E2];
g = size(E,2);
lambda_max = max(X'*Y)/n;
lambda1 = .01 * lambda_max;
lambda2 = .01 * lambda_max;
wts = ones(p,1);
alpha = 1.6;
rho = .2;

[beta history] = grouping_convex(X,Y,lambda1,lambda2,E,wts,rho,alpha);


