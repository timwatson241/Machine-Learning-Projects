function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J=(1/2)*sum((R.*(((X*Theta')-Y).^2)),'all')+(lambda/2)*(sum(Theta.^2,'all'))+(lambda/2)*(sum(X.^2,'all'));

X_grad = (((X*Theta')-Y).*R)*Theta+X.*lambda;
Theta_grad = (((X*Theta')-Y).*R)'*X+Theta.*lambda;

grad = [X_grad(:); Theta_grad(:)];

end
