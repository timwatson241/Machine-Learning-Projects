function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values

m = length(y); % number of training examples

H = X*theta;%the definition of H for logistic regression
inside_func= (H-y).^2;%inside part of the formula for J - all operations are element-wise - returns a vector
B = sum(inside_func);%sum that vector
theta_sum = sum((lambda/(2*m))*theta.^2)-(lambda/(2*m))*(theta(1).^2);%remove the summation for the first element
J = (1/(2*m))*B+theta_sum;%calculate the cost

grad = zeros(size(theta));%initialize the "grad" vector
grad =(1/m).*(X'*(H-y)); %compute non regularized gradient
temp=grad(1); %save the first term
grad =(1/m).*(X'*(H-y))+(lambda/m).*theta;%calculate the cost multiplied by the data value element-wise, return a vector - this is the regularized grad vector
grad(1)=temp; %replace the first term by the non-regularized first term

grad = grad(:);


end
