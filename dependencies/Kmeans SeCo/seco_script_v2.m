
clear all
close all
clc
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % SCRIPT TO COMPUTE THE SECO FRAMEWORK WITH K-MEANS % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Raul Casana 13/03/2017
% Two sections added 15/03/2017: Final labels and LDA-based visualization
% Other cluster algorithms can be applied substituiting the kmeans function

%% Load data

% % % % Example Iris data % % % %
load iris_dataset
data = irisInputs';
class = irisTargets(1,:)' + 2*irisTargets(2,:)' + 3* irisTargets(3,:)';
% % % % % % % % % % % % % % % % %


% The scripts uses the names of data and class to work. Observations in row
% and features in columns.

% load('E:\PROJECT MOST UP TO DATA STUFF\secomatlab.mat') 
% data = Book1uneditedJARMOS5;
% class = your class;

%% Preprocess data

% [data, mu, sigma] = zscore(data); % Optional step

[m,n] = size(data);

%% Pre-define number of clusters for K-means

k = 2:10;
% k = [2,3,4,5,6,7]; % Or create a custom number of clusters
ktotal = length(k);

%% SeCo parameters

kmsamples = 500; % Number of K-means samples (the higher the better chances 
% to obtain good results by random initialization)
topsep = 50; % Number of best separation solutions to select.

%% KM Sampling

labels = zeros(m,kmsamples,ktotal);
for i=1:kmsamples
    for j=1:ktotal
        labels(:,i,j)= kmeans(data,k(j)); % See documentation for more complex arguments.
    end
end

%% Measuring separation

% This is the Separation considering a single cluster (trivial solution).
SSQ0 = sum(sum((data - ones(m,1)*mean(data,1)).^2)); % Total sum of squares

SSQdata = zeros(kmsamples,ktotal);

for i=1:kmsamples
    for j=1:ktotal
        % Computing the number of clusters in datatest
        clusters = unique(labels(:,i,j));
        
        SSQ = 0; % Within cluster sum of squares, aggregating each cluster contribution
        for h=1:length(clusters)
            ind = (labels(:,i,j) == clusters(h));
            
            centroid = mean(data(ind,:),1);
            
            SSQk = data(ind,:) - ones(size(data(ind,:),1),1)*centroid;
            SSQ = SSQ + sum(sum(SSQk.^2));
        end
        
        SSQdata(i,j) = SSQ0 - SSQ; % Substract Total SS minus Within cluster SS
    end
end

%% Select the best solutions by Separation

[SSQdatasorted, SSQdatasorted_ind] = sort(SSQdata, 'descend');

aux = 1:topsep; % Auxiliar vector for selecting index of the best SSQ sol.
SSQdatabest = SSQdatasorted(aux,:); % We use later this filtered solutions.

bestSSQsolutions = zeros(m,topsep,ktotal);
for j=1:ktotal
    bestSSQsolutions(:,:,j) = labels(:,SSQdatasorted_ind(aux,j),j);
end

%% Measuring the Concordance
% Computing the median of Cramer's V in the bestSSQsolutions

cvdata = zeros(topsep,ktotal);
for j=1:ktotal
    for i=1:topsep
        cvdata(i,j) = cramer(bestSSQsolutions(:,i,j),bestSSQsolutions(:,aux~=i,j));
    end
end

%% SeCo plot

% auxiliar matrix to makea gscatter plot
clusterlabel = ones(topsep,1)*k;
secodata = [cvdata(:),SSQdatabest(:),clusterlabel(:)];
markers = 'ox*sdv+^<>ph'; colors = 'mrgbkc';

figure
gscatter(secodata(:,1),secodata(:,2),secodata(:,3),colors,markers,8)
xlabel('Concordance')
ylabel('Separation')
title('SeCo framework')
grid minor

% If separation has big values, consider log scale in y axis just applying
% logs on secodata(:,2).

%% Pick the best solution
% Choose the appropiate k based on SeCo plot.
k0 = 8; % This is a manual process. Indicate the k number, not it's index.
aux2 = 1:ktotal;
k0_ind = aux2(k==k0); % That gives us the index of the vector of k.

% Our best estimate solution is considering the most separated of the
% chosen k0. The matrix 'bestSSQsolutions' is already sorted by max
% separation, so the first element corresponds with the most separated.

final_labels = bestSSQsolutions(:,1,k0_ind); % We extract the labels from the matrix of labels.

%% Visualization based on a LDA (from Paulo)

% If you have the real labels you can uncomment next line, to compare them
% with our solution.

% c3dvis([data,class]);


% Visualization using a 3d projection of the data using the final labels
% with a LDA-bsed method (maximazing the separation of the labels)
c3dvis([data,final_labels]);


%% PCA representation
% In case you want compare different solutions with fixed observations.
[coeff1,score1,latent1,tsquared1,explained1,mu1] = pca(data);

figure
scatter3(score1(:,1),score1(:,2),score1(:,3),10,final_labels)
% legend('show')
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
grid on

