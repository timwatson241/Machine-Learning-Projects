function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

H = sigmoid(X*theta);%the definition of H for logistic regression
inside_func= -y.*log(H)-(1-y).*log(1-H);%inside part of the formula for J - all operations are element-wise - returns a vector
B = sum(inside_func);%sum that vector
theta_sum = sum((lambda/(2*m))*theta.^2)-(lambda/(2*m))*(theta(1).^2);%remove the summation for the first element
J = (1/m)*B+theta_sum;%calculate the cost

grad = zeros(size(theta));%initialize the "grad" vector
grad =(1/m).*(X'*(H-y)); %compute non regularized gradient
temp=grad(1); %save the first term
grad =(1/m).*(X'*(H-y))+(lambda/m).*theta;%calculate the cost multiplied by the data value element-wise, return a vector - this is the regularized grad vector
grad(1)=temp; %replace the first term by the non-regularized first term

grad = grad(:);

end
