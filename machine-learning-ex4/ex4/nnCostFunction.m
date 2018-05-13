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
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%y_matrix  size: num_labels x  m
%make [1; 4 ; 10 ] ---> [[1;0;0...],[0;0;0;1...] ]

% 10 x 10000
y_matrix = [];

for i =1:m
	y_matrix_i = zeros(num_labels,1);
	y_matrix_i(y(i)) = 1;
	y_matrix = [y_matrix,y_matrix_i];
end

% 10000x401
alpha_1 = [ones(m,1),X];
% 10000x401 x 401 x 25
z_2 = alpha_1 * Theta1'; 
alpha_2 = sigmoid(z_2);

%  10000 x 26
alpha_2 = [ones(m,1),alpha_2];

%  10000 x 26 x 26 x 10
z_3 = alpha_2 * Theta2';
% 1000 x 10
alpha_3 = sigmoid(z_3);



log_h_x = log(alpha_3);


% 10000 x 10 x 10 x 10000 
J_matrix =  (-log(alpha_3) * y_matrix - log(1 - alpha_3) * (1 - y_matrix)) * 1/m;

%结果10000x1000 取矩阵的对角线的和即可 完美
J =  sum(diag(J_matrix));


%加上正则项

% 去掉bias featues 即 1st column  25x401 --> 25x400
newTheta1 = Theta1(:,2:end);

% 去掉bias featues 即 1st column  10x26 --> 10x25
newTheta2 = Theta2(:,2:end);


regulation = (lambda / (2*m)) *(sum(sum(newTheta1 .* newTheta1)) +...

sum(sum(newTheta2 .* newTheta2)) );

J= J + regulation;


% backpropagation计算梯度

% 10000 x 10
err_3 = alpha_3  - y_matrix';  


% (26 x 10 x 10 x 10000)'  .*   (10000 x 26) =  10000 x 26
err_2 = (Theta2' * err_3' )' .* sigmoidGradient([ones(m,1),z_2]);

% remove epsilon(0) for every layer
% 10000 x 25
err_2 = err_2(:,2:end);

% (401 x 25 x 25 x 10000)'  .*   (10000 x 401) =  10000 x 401
err_1 = (Theta1' * err_2')' .* sigmoidGradient([ones(m,1) X]);

% remove epsilon(0) for every layer
% 10000 x 400
err_1 = err_1(:,2:end);





% 25 x 401 = 25x401 +   (401x10000 x 10000 x 25 )'
Theta1_grad =( Theta1_grad  + (alpha_1' * err_2 )' ) / m;

Theta1_b = [zeros(size(Theta1),1),Theta1(:,2:end)];



% 10 x 26 = 10x26 + ( 26 x 10000 x 10000 x 10)
Theta2_grad =( Theta2_grad  + (alpha_2' * err_3 )' ) / m;

Theta2_b = [zeros(size(Theta2),1),Theta2(:,2:end)];



% add regulation
Theta1_grad = Theta1_grad + (lambda/m)*Theta1_b;

Theta2_grad = Theta2_grad + (lambda/m)*Theta2_b;























% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
