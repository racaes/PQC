function [K,labels,maxERR,Slocal,centroids]=QC3main_eig_v5(datagen,noise, local,datallo,QCsetup)
% Main function to call the QC3, stablish the noise and the local
% covariance, introduce the QC3 and its gradient descent.

if ~exist('noise','var')
    noise = 0.05; % Noise to 1% by default
end

if ~exist('local','var')
    local = 0.1; % Noise to 15% by default
end

[m,n] = size(datagen);
%% Selection the k-nearest neighbours and the noise mean.
dist2 = squareform(pdist(datagen,'euclidean')); % d2(i,j) is the distance^2 between i and j
[dist2sort,dist2ind] = sort(dist2);

% % % To save time just compute the r
% % Noise calculus
% sknn = zeros(m-1,m);
% for j=1:(m-1)
%     sknn(j,:) = mean(dist2sort(2:j+1,:),1);
% end

% Local covariance calculus
knn = ceil((m-1)*local);
localknn = dist2ind(2:(knn+1),:);

% mindist = sknn(ceil((m-1)*noise),:).^2';
if knn>1
    mindist =1/n * sum(dist2sort(2:ceil((m-1)*noise)+1,:).^2,1)'/(knn-1);
else
    mindist =1/n * sum(dist2sort(2:ceil((m-1)*noise)+1,:).^2,1)'/knn;
end

Ueig = zeros(m,n^2);
Seig = zeros(m,n);
for i=1:m
    aux = datagen(localknn(:,i),:)-ones(knn,1)*datagen(i,:);
    if knn>1
        [U,S,~] = svd((aux'*aux)/(knn-1));
    else
        [U,S,~] = svd((aux'*aux)/knn);
    end
    tempdiag =  diag(S);
    tempdiag(tempdiag<=mindist(i)) = mindist(i);
    Seig(i,:) = tempdiag;
    Ueig(i,:) = U(:);
end
Slocal = [Ueig,Seig];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

[Dnew,Vlast,~,~,~,maxERR,maxERRdist] = ...
    graddescQC3(datagen,Slocal,datallo,QCsetup);

[labels, centroids, K] = QCalloc5(Dnew,Vlast,maxERR);
