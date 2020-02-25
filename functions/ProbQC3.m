function [Pjoint, Px, Pk, Px_k, Pk_x] = ProbQC3(ri,Slocal,r,labels)
% function qc2sigmavar that applies QC to a normalized wave function
% purpose: performing quantum clustering in n dimensions.
% Modified by Raul on 08/08/2016.
% input:
%       ri - a vector of points in n dimensions
%       q - the factor q which determines the clustering width
%       r - the vector of points to calculate the potential for. equals ri if not specified
% output:
%       V - the potential
%       P - the wave function
%       E - the energy
%       dV - the gradient of V
% example: [V,P,E,dV] = qc ([1,1;1,3;3,3],5,[0.5,1,1.5]);
% see also: qc2d

%close all;
if ~exist('r','var')
    r=ri;
end;

[pointsNum,dims] = size(ri);
calculatedNum=size(r,1);

% Default local covariance
if ~exist('Slocal','var')
    local = 0.2; % Local to 20% by default
    noise = 0.02; % Noise to 2% by default
    
    dist2 = squareform(pdist(ri,'euclidean')); % d2(i,j) is the distance^2 between i and j
    [dist2sort,dist2ind] = sort(dist2);
    
%     % Noise calculus
%     sknn = zeros(pointsNum-1,pointsNum);
%     for j=1:(pointsNum-1)
%         sknn(j,:) = mean(dist2sort(2:j+1,:),1);
%     end
    
    % Local covariance calculus
    knn = ceil((pointsNum-1)*local);
    localknn = dist2ind(2:(knn+1),:);
    Ueig = zeros(pointsNum,dims^2);
    Seig = zeros(pointsNum,dims);
    
   % mindist = sknn(ceil((pointsNum-1)*noise),:)';
%     mindist = mean(dist2sort(2:ceil((pointsNum-1)*noise)+1,:),1).^2';
if knn>1
    mindist =1/n * sum(dist2sort(2:ceil((pointsNum-1)*noise)+1,:).^2,1)'/(knn-1);
else
    mindist =1/n * sum(dist2sort(2:ceil((pointsNum-1)*noise)+1,:).^2,1)'/knn;
end

    for i=1:pointsNum
        aux = ri(localknn(:,i),:)-ones(knn,1)*ri(i,:);
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
end

Ueig = Slocal(:,1:end-dims);
Seignoise = Slocal(:,end-dims+1:end);

% Auxiliar vector to vectorise the multiple matrix multiplication
aux = zeros(dims^2,dims);
for j=1:dims
    i1=(j-1)*dims+1;
    i2=j*dims;
    aux(i1:i2,j) = ones(dims,1);
end

uniquelabels = unique(labels);
K = length(uniquelabels);

% Pre-allocating variables depending on r points
Pjoint=zeros(calculatedNum,K);
Pk = zeros(1,K);


%run over all the points and calculate for each the P, V and dV
for point = 1:calculatedNum
    % parfor point = 1:calculatedNum
    for k=1:K
        
        tempri = ri(labels == uniquelabels(k),:);
        templength = size(tempri,1);
        tempUeig = Ueig(labels == uniquelabels(k),:);
        tempSeignoise = Seignoise(labels == uniquelabels(k),:);
        
        % Marginal probability over K
        Pk(1,k) = size(tempri,1)/pointsNum;
        
        
        rdif = repmat(r(point,:),templength,1)-tempri; % diference vector: r-ri
        rdifU = (repmat(rdif,1,dims).*tempUeig)*aux;  % vector rdif in Ueig base
        z = -0.5*sum( (rdifU.^2)./tempSeignoise ,2); % exp(z)
        singlePoint = 1./sqrt(prod(2*pi*tempSeignoise,2)).*exp(z);
        
        % Joint probability on r(point) per K cluster
        Pjoint(point,k) = sum(singlePoint,1)/pointsNum;       
        
    end
end;
% Marginal probability over x
Px = sum(Pjoint,2);

% Conditional probability of x over k
Px_k = Pjoint./ (ones(calculatedNum,1)*Pk);

% Conditional probability of k over x
Pk_x = Pjoint./ (Px * ones(1,K));
