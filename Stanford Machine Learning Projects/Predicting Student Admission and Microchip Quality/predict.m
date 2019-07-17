function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

for i = 1:length(X) %go through each entry
    if sigmoid(X(i) * theta) >= 0.5 %if the prob that they pass is >=0.5
        p(i) = 1; %set the label of the corresponding entry in the p vector to 1
    else
        p(i) = 0; %set the label of the corresponding entry in the p vector to 0
    end
end 

end
