function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p_all = zeros(size(X, 1), 1);
p = zeros(size(X, 1), 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

%isolate the row j of the all_thetas and the i row of the X data points
%multiply them together
%apply sigmoid to get the probability that X row i corresponds to number
%j, repeat for each j
%save those probabilities in a matrix p_all
%use max function to figure out the max probability of each row and the
%column id that corresponds to it

for i = 1:size(X,1)
    for j = 1:size(all_theta)
        z = all_theta(j,:)*X(i,:)';
        p_all(i,j)=sigmoid(z);
    end
    [M,I] = max(p_all(i,:),[],2);
    p(i)=I;
end

end
