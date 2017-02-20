function [theta, J_history, thetaHistory] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

thetaHistory = theta';

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % 1 x N
    
    %the difference between the predicted values and the real ones
    diff = X * theta - y;
   
    %apply the derived function of the cost, to find the direction to go
    delta = (1/m) * sum((diff .* X));
    
    %update theta with the delta, according to the learning rate
    theta = theta .- (alpha*delta)';
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    thetaHistory = [thetaHistory; theta'];

end

end
