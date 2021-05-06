function [ Y, L, U ] = runMultiLayer( X, W, V )
% RUNMULTILAYER Calculates output and labels of the net
%
%    Inputs:
%              X - Data samples to be classified (matrix)
%              W - Weights of the hidden neurons (matrix)
%              V - Weights of the output neurons (matrix)
%
%    Output:
%              Y - Output for each sample and class (matrix)
%              L - The resulting label of each sample (vector) 
%              U - Activation of hidden neurons (vector)

% Add your own code here
S = X*W; % Calculate the weighted sum of input signals (hidden neuron)

%U = (size(S,2) , tanh(S); 
U = tanh(S);
U(:,end+1) = 1;% Calculate the activation of the hidden neurons (use hyperbolic tangent)
Y = U*V; % Calculate the weighted sum of the hidden neurons
%tanh(activation function) 
% V = 1x3 
% U =
% S 3x2 
% basera storleken på U på storleken på V

% Calculate labels
[~, L] = max(Y,[],2);

end

