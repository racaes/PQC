function [Pjoint, Px, Pk, Px_k, Pk_x] = probQC(ri,sigma,r,labels)
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

%default sigma as 10% of knn
if ~exist('sigma','var')
    % d2(i,j) is the distance^2 between i and j
    dist2 = squareform(pdist(ri,'euclidean'));
    [dist2sort,~] = sort(dist2);
    sknn = zeros(pointsNum-1,pointsNum);
    for i=1:(pointsNum-1)
        sknn(i,:) = mean(dist2sort(2:i+1,:),1);
    end
    sigma = sknn(round(pointsNum*0.1),:)';
end;

if size(sigma,1)<size(sigma,2)
    sigma = sigma';
    if size(sigma,2)~=1
        error('Sigma is not a column vector')
    end
end

% Cluster number
uniquelabels = unique(labels);
K = length(uniquelabels);

% Pre-allocating variables depending on r points
Pjoint=zeros(calculatedNum,K);
Pk = zeros(1,K);


%run over all the points and calculate for each the P, dP2, V and dV
for point = 1:calculatedNum
    
    for k=1:K
        
        tempri = ri(labels == uniquelabels(k),:);
        templength = size(tempri,1);
        tempsigma = sigma(labels == uniquelabels(k),:);
        
        % Marginal probability over K
        Pk(1,k) = size(tempri,1)/pointsNum;
        
        % Vector of distances^2 from r(point) to each ri
        D2 = sum( (ones(templength,1)*r(point,:) - tempri).^2, 2);
        
        % Each term of wave function.
        singlePoint = 1./(2*pi.*tempsigma.^2).^(dims/2).*exp(-0.5*D2.*tempsigma.^(-2));
        
        % Joint probability on r(point) per K cluster
        Pjoint(point,k) = sum(singlePoint,1)/pointsNum;
        
    end    
end
% Marginal probability over x
Px = sum(Pjoint,2);

% Conditional probability of x over k
Px_k = Pjoint./ (ones(calculatedNum,1)*Pk);

% Conditional probability of k over x
Pk_x = Pjoint./ (Px * ones(1,K));
