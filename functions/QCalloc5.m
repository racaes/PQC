function [labels, centroids, K] = ...
    QCalloc5(dataV,V,ERR)
% Function that allocate the data close to each potential minima and
% identifies each cluster
% 
% INPUTS:
% - dataV: Data allocated in potential minima after the SGD
% - V: Last values of potential
% - ERR: Tolerance to discriminate distance between different clusters
% 
% OUTPUTS:
% - labels: Labels identifying each cluster
% - centroids: Average position of each minima
% - K: Number of clusters
% - margin: Distance to the next closest point out of the cluster.

% Modified 25/07/16 to change the inter-cluster threshold parameter.
% Modified 28/07/16 to avoid outliers as single clusters, 'refine' option.
% 
% New function created on 28/02/2017 with Community finding algorithm.
% https://es.mathworks.com/matlabcentral/fileexchange/45867-community-detection-toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% if isdir('D:\Google Drive\matlab functions\CDTB')
%     addpath(genpath('D:\Google Drive\matlab functions\CDTB'))
% elseif isdir('C:\Users\cmprcasa\Google Drive\matlab functions\CDTB')
%     addpath(genpath('C:\Users\cmprcasa\Google Drive\matlab functions\CDTB'))
% end

[m,n] = size(dataV);

lambda1 = mean(sum(dataV.^2,2).^0.5);
lambda2 = mean(abs(V));

newdata = [dataV./lambda1, V./lambda2];
dist2 = squareform(pdist(newdata,'squaredeuclidean'));
sig = max(ERR,0.1);
Amat = exp(-dist2./sig^2);

labels = GCModulMax1(Amat);
total = unique(labels);

K = length(total);
centroids = zeros(K,n);

for i=1:K
    index = labels == total(i);    
    centroids(i,:) = mean(dataV(index,:),1);
end

end