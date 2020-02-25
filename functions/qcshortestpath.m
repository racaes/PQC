function [paths,ok] = qcshortestpath (alldata,distallV,c0,c1, minstep,K)

% SHORTEST PATH function
m = size(alldata,1)-K;

[distallsort,distallind] = sort(distallV);
[~,undoind] = sort(distallind);
% tic
knnlimit = 1;
steps = 30;
dist1= Inf;
while dist1==Inf
    % Limit nodes till noise knn
    distallfilt = distallsort;
    knnlimit = (knnlimit^(1/3)+1/steps)^3 + 1;
    test = floor(min([knnlimit,m]));
    if test<m
        distallfilt(test:end,:)=0;
    end
    redodistall = zeros(size(undoind,1));
    for q=1:size(undoind,1)
        redodistall(:,q) = distallfilt(undoind(:,q),q);
    end
    
    D = sparse(redodistall);
    [dist1, path1, ~] = graphshortestpath(D, m+c0, m+c1);
end
% toc
if ~exist('minstep','var')
    minstep = 0.05;
end


pairdist = zeros(length(path1)-1,1);
extrapaths = [];
for q=1:length(path1)-1
    pairdist(q) = distallV(path1(q),path1(q+1));
    if pairdist(q)>minstep
        x = alldata(path1(q),:);
        y = alldata(path1(q+1),:);
        alphasteps = ceil(pairdist(q)/minstep);
        alpha = linspace(0,1,alphasteps)';
        temppath = ones(alphasteps,1)*x + alpha*(y-x);
        extrapaths = [extrapaths;temppath];
    end
end
ok = dist1 == sum(pairdist);
paths = [alldata(path1,:);extrapaths];


% figure
% scatter(paths(:,1),paths(:,2),'b')
% hold on
% scatter(extrapaths(:,1),extrapaths(:,2),'b')


% h = view(biograph(D, [],'ShowWeights','on'));
% h = view(biograph(tril(D),[],'ShowArrows','off','ShowWeights','on'));
% [dist, path, pred] = graphshortestpath(tril(D), 4, 151,'Directed',false);


