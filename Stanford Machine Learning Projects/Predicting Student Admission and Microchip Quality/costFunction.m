function [cost, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples
H = 1./(1+exp(-X*theta));%the definition of H for logistic regression
inside_func= -y'*log(H)-(1-y)'*log(1-H);%inside part of the formula for J - all operations are element-wise - returns a vector
B = sum(inside_func);%sum that vector
cost = (1/m)*B;%calculate the cost

grad = zeros(size(theta));%initialize the "grad" vector
for i = 1:size(X,2)%for every column in the data set X
    W = (H-y).*X(:,i);%calculate the cost multiplied by the data value element-wise, return a vector
    B = (1/m)*sum(W);%sum that vector and multiply by 1/m
    grad(i)=B; %assign the result to the corresponding row of the grad vector
end
grad 
cost
end
