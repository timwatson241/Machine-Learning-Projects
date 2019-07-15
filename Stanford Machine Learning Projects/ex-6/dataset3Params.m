function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
Possible_C= [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
Possible_Sigma= [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]

error = zeros(length(Possible_C),length(Possible_Sigma))

for i = 1:length(Possible_C)
    for j = 1:length(Possible_Sigma)
        
        model= svmTrain(X, y, Possible_C(i), @(x1, x2) gaussianKernel(x1, x2, Possible_Sigma(j)));
        
        predictions = svmPredict(model, Xval);
        
        error(i,j) = mean(double(predictions ~= yval));
    end
end

[min_val,idx]=min(error(:));
[row,col]=ind2sub(size(error),idx);

C = Possible_C(row);
sigma = Possible_Sigma(col);

end
