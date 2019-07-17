function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup variables
m = size(X, 1);
n = size(X, 2);  

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%add 1 to the start of each row of the training set
X = [ones(m, 1) X];

%Create matrix for the y values with a 1 at the correct column
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%calculate H
a1=X;
z2=a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];%add 1 to the top of the a2 vector
z3 = a2 * Theta2'; 
H = sigmoid(z3);

%calculate J
inside_func= -y_matrix.*log(H)-(1-y_matrix).*log(1-H);%inside part of the cost function
Reg_term_theta1=Theta1(:,2:end).^2; %remove the first column from each matrix and square it element wise
Reg_term_theta2=Theta2(:,2:end).^2;
J = (1/m) * sum(sum(inside_func)) + (lambda/(2*m))*(sum(Reg_term_theta1,'all') + sum(Reg_term_theta2,'all'));

% Part 2: Implement the backpropagation algorithm to compute the gradients

d3 = H-y_matrix;
d2 = (d3*Theta2) .* (a2.*(1-a2));
d2 = d2(:,2:end);

Delta1 = d2'*a1;
Delta2 = d3'*a2;

Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;

%grad = [D1(:);D2(:)]
% Part 3: Implement regularization with the cost function and gradients.

Theta1(:,1) = 0; %Set first column to 0
Theta2(:,1) = 0;

Theta1 = (lambda/m)*Theta1;
Theta2 = (lambda/m)*Theta2;

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
