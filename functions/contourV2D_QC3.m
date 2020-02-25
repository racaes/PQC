function contourV2D_QC3(datagen, noise, local, gridsize, N, wf, min0, max1, points,grad)
% Function to represent data potential unsing only 2 PC of U matrix (SVD).
% MOD: 10/08/2016 to include the new QC with variable sigma: qc2sig2.

% if size(datagen,2)>2
%     warning('Data has more than 2 dimensions. Only 2 PC considered')
% end

% [U,~,~] = svd(datagen,0);
% data = U(:,1:2);
data = datagen(:,1:2);
transparency = 0.3;
if ~exist('noise','var')
    noise = 0.001; % Noise to 1% by default
end

if ~exist('local','var')
    local = 0.01; % Noise to 15% by default
end

[m,n] = size(datagen);
% Selection the k-nearest neighbours and the noise mean.
dist2 = squareform(pdist(datagen,'euclidean')); % d2(i,j) is the distance^2 between i and j
[dist2sort,dist2ind] = sort(dist2);

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


%%
if ~exist('gridsize','var')
    gridsize = 30;
end

if ~exist('N','var')
    N = 30;
end

if ~exist('wf','var')
    wf = false;
end

if ~exist('min0','var')
    min0 = false;
end

if ~exist('max1','var')
    max1 = false;
end

if ~exist('points','var')
    points = true;
end

if ~exist('grad','var')
    grad = false;
end

datagrid = zeros(gridsize,2);
datagrid(:,1) = linspace(min(data(:,1)), max(data(:,1)),gridsize);
datagrid(:,2) = linspace(min(data(:,2)), max(data(:,2)),gridsize);

datagrid2 = zeros(gridsize^2,1);
[gridx, gridy] = meshgrid(datagrid(:,1),datagrid(:,2));
datagrid2(:,1) = reshape(gridx,[],1);
datagrid2(:,2) = reshape(gridy,[],1);

[V,P,E,dV] = qc3_eig_v3(data,Slocal,datagrid2);

% Vo = V./max(V);
if ~min0
    V = V - E;
end

if max1
    V = V./max(V);
end


gridP = reshape(P,gridsize,gridsize);
gridV = reshape(V,gridsize,gridsize);

if wf
    
    h=figure('Name', 'Wave function');
    set(h,'Position',[156 186 880 762]);
    contour3(gridx, gridy, gridP,N)
    colormap(hsv)
    alpha(transparency)
    hold on
    plot(data(:,1),data(:,2),'*k')
    xlabel('X1')
    ylabel('X2')
    zlabel('P(X)')
    title(['Wave funtion.'])% Sigma: ', num2str(mean(Slocal))])
    
    
    h=figure('Name', 'Wave function');
    set(h,'Position',[156 186 880 762]);
    surf(gridx, gridy, gridP)
    alpha(transparency)
    colormap(cool)
    hold on
    plot(data(:,1),data(:,2),'*k')
    xlabel('X1')
    ylabel('X2')
    zlabel('P(X)')
    title(['Wave funtion.'])% Sigma: ', num2str(mean(Slocal))])

end

if grad
    h=figure('Name', 'Gradient');
    dVn = normr(dV);
    set(h,'Position',[156 186 880 762]);
    contour(gridx, gridy, gridV,N)
    alpha(transparency)
    hold on
    plot(data(:,1),data(:,2),'*k')
    quiver(datagrid2(:,1), datagrid2(:,2), -dVn(:,1), -dVn(:,2))

    xlabel('X1')
    ylabel('X2')
    title(['grad(V)'])% Sigma: ', num2str(mean(Slocal))])
end

h=figure('Name', 'Potential');
set(h,'Position',[156 186 880 762]);
contour3(gridx, gridy, gridV,N)
alpha(transparency)
if points
    hold on
    plot(data(:,1),data(:,2),'*k')
end
xlabel('X1')
ylabel('X2')
zlabel('V(X)')
title(['Potential. Max(V):', num2str(max(V)),'. Min(V):', num2str(min(V)),...
    '. Points:', num2str(size(data,1))])

h=figure('Name', 'Potential');
set(h,'Position',[156 186 880 762]);

surf(gridx, gridy, gridV)
alpha(transparency)
if points
    hold on
    plot(data(:,1),data(:,2),'*k')
end
xlabel('X1')
ylabel('X2')
zlabel('V(X)')
title(['Potential. Max(V):', num2str(max(V)),'. Min(V):', num2str(min(V)),...
    '. Points:', num2str(size(data,1))])
end
