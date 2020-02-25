function [V,Pn,E,dV] = qc3_eig_v3(ri,Slocal,r)
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
    noise = 0.2; % Noise to 1% by default
    local = 0.1; % Noise to 15% by default
    
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
    
%     mindist = sknn(ceil((pointsNum-1)*noise),:)';
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

% Aux index matrix to obtain Ueig transpose in row form:
t = reshape(1:dims^2, dims, dims);
t = t';
t = t(:);
Ueigt = Ueig(:,t);

% Pre-allocating variables depending on r points
P = zeros(calculatedNum,1);
V = zeros(calculatedNum,1);
dV = zeros(calculatedNum,dims);

% In case of labels allocation, just need to create unique(labels) and look
% for the pointsNum in each label before the genpoint for.

%run over all the points and calculate for each the P, V and dV
for point = 1:calculatedNum
    % parfor point = 1:calculatedNum
    
    rdif = repmat(r(point,:),pointsNum,1)-ri; % diference vector: r-ri
    % method to multiply two matrices: rdif_mxn * Ueig_n2x1 (nxn)
    rdifU = (repmat(rdif,1,dims).*Ueig)*aux;  % vector rdif in Ueig base: rdif*Ueig  
    z = -0.5*sum( (rdifU.^2)./Seignoise ,2); % exp(z)
    singlePoint = 1./sqrt(prod(2*pi*Seignoise,2)).*exp(z);
    
    % Maux = rdif * sigma-1 (m x d); this term is repeated quite often.
    Maux = (repmat(rdifU./Seignoise, 1,dims).*Ueigt)*aux;
    
    % tracelong = trace( Maux'*Maux)
    % % Old expresion_v2:    tracelong = sum( (rdifU./Seignoise).^2 ,2);
    tracelong = sum( Maux.^2 ,2);

    % Old expresion:
    %trper2dim = sum(Seignoise,2)/dims/2; % Error detectec, not 1/dim !!
    trper2dim = sum(Seignoise,2)/2;
    trinv = sum(1./Seignoise,2);
    
    % Tracelong derivative: 2*rdifu*S^-2*Ueigt
    trlongderiv = 2*(repmat(rdifU.*(Seignoise.^(-2)), 1,dims).*Ueigt)*aux;
    
    % Wave function on r(point)
    P(point) = sum(singlePoint,1);
    V(point) = sum( trper2dim.*singlePoint.*(tracelong-trinv),1)/P(point);    
    
    % S^-1*X = (U*Seig*U')^-1*X = U'^-1 * Seig^-1 * U^-1 * X
    % But U*U'=U'*U = I, so S^-1*X = U'^-1 * Seig^-1* U^-1 * X
    dV1p1 = sum( repmat(trper2dim.*singlePoint, 1,dims).*trlongderiv - ...
        repmat(trper2dim.*tracelong.*singlePoint, 1,dims).* Maux  ,1);
    dV1p2 = sum( trper2dim.*tracelong.*singlePoint, 1);
    dVp0 = sum( repmat(singlePoint, 1,dims).* Maux ,1);
    
    dV2p1 = sum( repmat(-trper2dim.*trinv.*singlePoint, 1,dims).* Maux ,1);
    dV2p2 = sum( trper2dim.*trinv.*singlePoint,1);
    
    dV(point,:) = dV1p1/P(point)+dV1p2.*dVp0.*P(point).^-2 - dV2p1/P(point)-...
        dV2p2.*dVp0.*P(point).^-2;
    
    
end;
Pn = P/pointsNum;
E = -min(V);
V = V + E;

