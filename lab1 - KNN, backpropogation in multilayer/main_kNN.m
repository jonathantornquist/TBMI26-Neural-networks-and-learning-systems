%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
plotCase(X,D)

%% Select a subset of the training samples

numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
% XBin2 = XBins{2};
% XBin3 = XBins{3};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% Add your own code to setup data for training and test here

%Loop and plot to Find the optimal k 
N=30; %Searching through more k:s, slows down the code. 
vector_with_k_and_acc=zeros(N,2); 
for k_counter=1:N 
 vector_with_k_and_acc(k_counter,2)=k_counter;
 
%For cross validation 
acc_vector=zeros(numBins,1);
counter=numBins;
for i=1:(numBins-1) %Crossvalidation, combins all 2+1 combinations from the bins
for j=1:(numBins-i)
XBinComb = combineBins(XBins, [i,i+j]);
LBinComb = combineBins(LBins, [i,i+j]);

XTrain = XBinComb;
LTrain = LBinComb;
XTest  = XBins{counter};
LTest  = LBins{counter};

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.

% Set the number of neighbors
k = k_counter; 

% Classify training data

LPredTrain = kNN(XTrain, k, XTrain, LTrain);

LPredTest  = kNN(XTest , k, XTrain, LTrain);

%% Calculate The Confusion Matrix and the Accuracy
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

% The confucionMatrix
cM = calcConfusionMatrix(LPredTest, LTest);

% The accuracy
acc = calcAccuracy(cM);
acc_vector(counter,1)=acc; 
counter=counter-1;
end 
end

cM;
acc=sum(acc_vector)/numBins;
vector_with_k_and_acc(k_counter,1)=acc;
end

max_acc=max(vector_with_k_and_acc(:,1))

plot(1:N,vector_with_k_and_acc(:,1))
title('Accuracy with different k-values')

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], k);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
