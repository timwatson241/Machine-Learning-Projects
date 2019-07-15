function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


m = length(y); % number of training examples
H = 1./(1+exp(-X*theta));%the definition of H for logistic regression
inside_func= -y'*log(H)-(1-y)'*log(1-H);%inside part of the formula for J - all operations are element-wise - returns a vector
B = sum(inside_func);%sum that vector
theta_sum = sum((lambda/(2*m))*theta.^2)-(lambda/(2*m))*(theta(1).^2);%remove the summation for the first element
J = (1/m)*B+theta_sum;%calculate the cost

grad = zeros(size(theta));%initialize the "grad" vector
%for first element
W = (H-y).*X(:,1);%calculate the cost multiplied by the data value element-wise, return a vector
B = (1/m)*sum(W);%sum that vector and multiply by 1/m
grad(1)=B; %assign the result to the corresponding row of the grad vector
    
for i = 2:size(X,2)%for every column in the data set X except for first element
    W = (H-y).*X(:,i);%calculate the cost multiplied by the data value element-wise, return a vector
    B = (1/m)*sum(W);%sum that vector and multiply by 1/m
    grad(i)=B+(lambda/m)*theta(i); %assign the result to the corresponding row of the grad vector
end

end
