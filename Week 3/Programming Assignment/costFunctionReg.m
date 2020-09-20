function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% similar to costfunction.m
h = sigmoid(X * theta);
% cost term
cost = (1/m) * sum((-y .* log(h))-((1-y).* log(1-(h))));

% regularize term - skipping theta 
% "end" keyword here refers to the last element in matriz we can subset the matrix
regCost = (lambda / (2*m)) * norm(theta([2:end])) ^2;

%identical to CostFunction.m
grad = (1/m) .* X' * (h-y);

% regularize gradient term -skipping theta(0)
regGrad = (lambda / m) .* theta;
regGrad(1) = 0;

% Regularisation just consists of summing the original cost
% with the regularized termi
J = cost + regCost;
grad = grad + regGrad;

% =============================================================

end
