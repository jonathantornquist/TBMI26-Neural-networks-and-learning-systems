function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);

for j=1:length(X)
    distance_arr = zeros(length(XTrain),2);%matrix with 2 columns: distance and label
for i=1:length(XTrain) 
Distance = sqrt(sum((X(j,:) - XTrain(i,:)).^2)); %eucl distance to every point
distance_arr(i,1) = Distance; %put every distance into array
distance_arr(i,2) = LTrain(i,1); %combine distance with label
end 
sorted_distance_arr = sortrows(distance_arr,1); %sort lowest to highest distance, labels attached to distance
k_vote = sorted_distance_arr(1:k, 2);
LPred(j,1) = mode(k_vote); %classify as K nearest neighbor using built in matlab mode function -> if draw it picks lowest value
end

end

