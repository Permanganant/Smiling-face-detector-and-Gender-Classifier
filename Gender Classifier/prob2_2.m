load("test.mat");
load("training.mat");

faces = TRAINING;
numFaces = size(faces,2);

h = 36; w = 36;

meanFace = mean(faces, 2);
faces = faces - repmat(meanFace, 1, numFaces);

% Perform Singular Value Decomposition
[u,d,v] = svd(faces, 0);

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
for i = 1:size(faces,1)
energy(i) = sum(eigVals(1:i));
end
propEnergy = energy./energy(end);

% Determine the number of principal components required to model 90% of data variance
percentMark = min(find(propEnergy > 0.9));
% Pick those principal components
eigenVecs = u(:, 1:percentMark);

% Do something with them; for example, project each of the neutral and smiling faces onto
% the corresponding eigenfaces

% eigenVecs(:,:) = -1 * eigenVecs(:,:);
menFaces = faces(:,1:2500); womenFaces = faces(:,2501:end);

menWeights = eigenVecs' * menFaces;
womenWeights = eigenVecs' * womenFaces;

% menWeights = pca(menFaces);
% womenWeights = pca(womenFaces);


figure(3)
plot3(menWeights(1,:),menWeights(2,:),menWeights(3,:),'r.','MarkerSize',11);
hold on
plot3(womenWeights(1,:),womenWeights(2,:),womenWeights(3,:),'b.','MarkerSize',11);
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')
legend('men','women')


%% TEST KISMI
%ilk 200erkek,son 200kiz
testFaces = TEST;
testMean = mean(testFaces,2);
testFaces = testFaces - repmat(testMean, 1, size(testFaces,2));

C = testFaces'*testFaces;

testWeights = eigenVecs' * testFaces;
% testWeights = pca(testFaces);

testMatch = zeros(1,400);
for i = 1:400
weightDiff = repmat(testWeights(:, i), 1, numFaces) - [menWeights,womenWeights];
[val, ind] = min(sum(abs(weightDiff), 1));
testMatch(i) = ind;
end

acc       = sum(testMatch(1:200)<2500)/400 + sum(testMatch(201:end)>2500)/400
acc_men   = sum(testMatch(1:200)<2500)/200 
acc_women = sum(testMatch(201:end)>2500)/200


