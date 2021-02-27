clear all; clc;
load("smileTrain.mat");
load("neutralTrain.mat");



faces = [smileTrain,neutralTrain];
numFaces = size(faces,2);

h = 64; w = 64;


meanFace = mean(faces, 2);
faces = faces - repmat(meanFace, 1, numFaces);

% Perform Singular Value Decomposition
[u,d,v] = svd(faces,0);

% Pull out eigen values and vectors
eigVals = diag(d);
eigVecs = u;

% Plot the mean sample and the first three principal components
figure(1); imagesc(reshape(meanFace, h, w)); title('Mean Face');colormap(gray);
figure(2);
subplot(1, 3, 1); imagesc(reshape(u(:, 1), h, w)); colormap(gray);title('First Eigenface');
subplot(1, 3, 2); imagesc(reshape(u(:, 2), h, w)); colormap(gray);title('Second Eigenface');
subplot(1, 3, 3); imagesc(reshape(u(:, 3), h, w)); colormap(gray);title('Third Eigenface');

figure(4);
for i=1:16
    subplot(4,4,i);
    imagesc(reshape(u(:,i),h,w)); colormap(gray);
    
end

% The cumulative energy content for the m'th eigenvector is the sum of the energy content
%across eigenvalues 1:m
for i = 1:length(eigVals)
energy(i) = sum(eigVals(1:i));
end
propEnergy = energy./energy(end);

% Determine the number of principal components required to model 90% of data variance
percentMark = min(find(propEnergy > 0.9));
% Pick those principal components
eigenVecs = u(:, 1:percentMark);

% Do something with them; for example, project each of the neutral and smiling faces onto
% the corresponding eigenfaces
;
smileFaces = smileTrain; neutralFaces = neutralTrain;

smileWeights = eigenVecs' * smileFaces;
neutralWeights = eigenVecs' * neutralFaces;


%% TEST KISMI
%ilk 200erkek,son 200kiz
load('smileTest.mat');
load('neutralTest.mat');
smileFaces = smileTest;
neturalFaces = neutralTest;

testFaces = [smileFaces,neutralFaces];
testMean = mean(testFaces,2);
testFaces = testFaces - repmat(testMean, 1, size(testFaces,2));


testWeights = eigenVecs' * testFaces;
% testWeights = pca(testFaces);

testMatch = zeros(1,165);
for i = 1:165
weightDiff = repmat(testWeights(:, i), 1, numFaces) - [smileWeights,neutralWeights];
[val, ind] = min(sum(abs(weightDiff), 1));
testMatch(i) = ind;
end

acc       = sum(testMatch(1:21)<191)/165 + sum(testMatch(21:end)>192)/165
acc_smile   = sum(testMatch(1:21)<191)/21 
acc_neutral = sum(testMatch(21:end)>191)/144


