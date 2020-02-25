function [V,Pn,E,dV] = qc2sig2(ri,sigma,r)
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

% Pre-allocating variables depending on r points
P=zeros(calculatedNum,1);
Ed2per2sigma2=zeros(calculatedNum,1);
Edxpersigma2=zeros(calculatedNum,dims);
Ediffd2per2sigma2=zeros(calculatedNum,dims);

%run over all the points and calculate for each the P, dP2, V and dV
for point = 1:calculatedNum
% parfor point = 1:calculatedNum
    % Vector of distances^2 from r(point) to each ri
    D2 = sum( (ones(pointsNum,1)*r(point,:) - ri).^2, 2);
    
    % Each term of wave function.
    singlePoint = 1./(2*pi.*sigma.^2).^(dims/2).*exp(-0.5*D2.*sigma.^(-2));
    
    % Each term of expected values.
    d2per2sigma2 = 0.5* D2 .* sigma.^(-2) .* singlePoint;
    sigma2inv = sigma.^(-2) .* singlePoint;
    % Wave function on r(point)
    P(point) = sum(singlePoint,1);
    
    % Expected values
    Ed2per2sigma2(point) = sum(d2per2sigma2,1)./ P(point);
    
    % Auxiliar factors of gradient V (matrix m x n)
    dx = ones(pointsNum,1)*r(point,:) - ri;
    auxgrad = sigma.^(-2).*( 1-0.5*D2.*sigma.^(-2)).* singlePoint;
    
    Edxpersigma2(point,:)= sum(dx.*(sigma2inv*ones(1,dims)))./P(point);
    Ediffd2per2sigma2(point,:) = sum(dx.*(auxgrad*ones(1,dims)))./P(point);
    
end;
% P(P==0) = min(P(P~=0)); % To avoid conflicts with 1/0
Pn = P/pointsNum;
V = -dims/2 + Ed2per2sigma2;

dV = Ediffd2per2sigma2 + Edxpersigma2.*( Ed2per2sigma2 *ones(1,dims) );

E = -min(V);
V = V + E;

