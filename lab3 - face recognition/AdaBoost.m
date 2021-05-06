%% Hyper-parameters
clc;
clear all;
% Number of randomized Haar-features
nbrHaarFeatures = 130; 
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers =70;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
d = ones(nbrTrainImages, 1) * (1/nbrTrainImages); %Initialize weight, set start weights to 1/N

adaVar = zeros(nbrWeakClassifiers, 3); %initialize matrix for AdaBoost variables
                                       %threshold, polarity, feature
                                 

for c = 1:nbrWeakClassifiers %iterate through all classifiers,
                             %we will find the nbrWeakClassifiers best classifiers
                             %start by initializing all variables
                             
    minErr = 1000000; %initialize large error
    threshold = 0; %Initialize threshold
    polarity = 0; %Initialize polarity
    feature = 0; %Initialize counter for feature
    minAlpha = 0; %initialize alpha
    h = 0;
    
    for f = 1:nbrHaarFeatures %iterate through all features
                              %for each feature we will find the best classification
        tao = xTrain(f,:);    %set tao to feature f for every data point
        
        %p = 1;
        
        for t = tao %iterate through all thresholds, to find best threshold for feature f
    
            p = 1; %
            pC = WeakClassifier(t,p,xTrain(f,:)); %pC = 1 or -1, classify using threshold tao
            e = WeakClassifierError(pC,d,yTrain); %calculate error of classification
            
           if e > 0.5 %if error is bigger than 0.5,
               p = -p; %switch polarity,
               e = 1-e; %get less error
           end
           
           if e < minErr %if we get an improvement, update variables
               minErr = e;
               minAlpha = 0.5 * log((1-minErr)/minErr);
               threshold = t;
               polarity = p;
               feature = f;
               h = p*pC;
           end
        end
    end
    
    d = d .* exp(-minAlpha * yTrain .* h)'; %calculate new weights
    d = d ./ sum(d); %Normalize
    
    adaVar(c,1) = threshold; %update matrix of optimal AdaBoost variables for classifier c
    adaVar(c,2) = polarity;
    adaVar(c,3) = feature;
    Alpha(c) = minAlpha; %NEW code Saves the minimum alpha for each classification.
    
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

%Train evaluation
for c = 1:nbrWeakClassifiers
   Train_eval(c,:) = WeakClassifier((adaVar(c,1)),(adaVar(c,2)),xTrain(adaVar(c,3),:));

   for i = 1:nbrTrainImages   
    
       %OLD CODE
       %  %Train_classification(i) = mode(Train_eval(:,i));
       
       %NEW CODE, uses alpha 
       for a=1:c
       aW(a)= Alpha(a)*Train_eval(a,i);
       end 
       weigthed_decision= sum(aW);
       aW(:)=0;

       if  weigthed_decision>= 0
           Train_classification(i)=1;
       else 
             Train_classification(i)=-1;
       end 
        %%%--------------------------------   
 
 

    Train_result(i) = Train_classification(i) ~= yTrain(i);
   end
   
   Train_acc(c) = 1 - sum(Train_result)/nbrTrainImages;
   end; 

Final_train_acc = Train_acc(nbrWeakClassifiers);
%Test evaluation
for c = 1:nbrWeakClassifiers
   Test_eval(c,:) = WeakClassifier((adaVar(c,1)),(adaVar(c,2)),xTest(adaVar(c,3),:));
   for i = 1:nbrTestImages
  
     %  OLD CODE, cant use mode 
     %Test_classification(i) = mode(Test_eval(:,i));
     %NEW code 
     for a=1:c
       aW(a)= Alpha(a)*Test_eval(a,i);
       end 
       weigthed_decision= sum(aW);
       aW(:)=0;

       if  weigthed_decision>= 0
           Test_classification(i)=1;
       else 
             Test_classification(i)=-1;
       end 
       %%%--------------------------------
    
    Test_result(i) = Test_classification(i) ~= yTest(i);
   end
   Test_acc(c) = 1 - sum(Test_result)/nbrTestImages;
end
Final_test_acc = Test_acc(nbrWeakClassifiers);

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
figure(4)
subplot(1,2,1)
plot(1:nbrWeakClassifiers,Train_acc)
title('Accuracy of training wrt number of weak classifiers')

subplot(1,2,2)
plot(1:nbrWeakClassifiers,Test_acc)
title('Accuracy of test wrt number of weak classifiers')




%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

figure(5);
colormap gray;

%Plot some incorrect faces
c = 1;
incorrect = 1;
while incorrect <= 15
    if Test_classification(c) ~= yTest(c)
        subplot(6,5,incorrect), imagesc(testImages(:,:,c));
        axis image;
        axis off;
        incorrect = incorrect + 1;
    end
    c = c + 1;
end

%Plot some randomly chosen (of last 2500 images) incorrect non-faces
c = nbrTestImages;
while incorrect <= 30
    if Test_classification(c) ~= yTest(c)
        subplot(6,5,incorrect), imagesc(testImages(:,:,c));
        axis image;
        axis off;
        incorrect = incorrect + 1;
    end
    c = c - 1;
end


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
figure(7);
colormap gray;

%Plot 20 features
for f = 1 : 20
    subplot(5,4,f), imagesc(haarFeatureMasks(:,:,adaVar(f,3)));
    axis image;
    axis off;
end




