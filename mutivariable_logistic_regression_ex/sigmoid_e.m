function [val] = sigmoid_e(X) 
	val = 1 ./ (1 + e.^(-X));
end