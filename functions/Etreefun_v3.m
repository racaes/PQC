function [Etree, labelstree, tracksol,Klabel0,label0,Etree2] = Etreefun_v3(dEr,labels,ERR)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% This function is an iterative method to merge clusters by minimum energy
% INPUTS:
% - dEr: Matrix indicating the sufficient energy needed to cluster jump.
%     Element dEr(i,j) means energy from going from cluster i to cluster j.
% - labels: These are the labels column vector, identified as different clusters
% - ERR: It is the threshold to consider the premerging step, where we assume
%     all clusters pairwise path within this range should be the same solution
%     
% OUTPUTS:
% - Etree: Shows the results of each jump made, where:
%     column 1 is the origin cluster
%     column 2 is the target cluster
%     column 3 is the sufficient energy to do the jump
%     column 4 is the accumulated energy
%     column 5 indicates if the jump corresponds to a leveling clusters, this
%         means that i goes to j, but previously, j went to i. So this jump does
%         not apport new information, and both clusters are at the same level.
%     column 6 indicates the number of different labels at this stage
%     column 7 indicates if this merge step introduces a new solution. Later,
%         only the steps with new solutions will be represented.
% - labelstree: Indicates the matrix of all solutions with real merging labels,
%     where each column vector is the labels solutions at this stage. Each column
%     means a new merge step. 
% - tracksolutions: shows the square matrix of different solutions in each step.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%% Iterative method to merge clusters by minimum energy step
k = size(dEr,1);
dErtemp = dEr + diag(nan(1,k));
fullclusters = zeros(k,k); % True indicates the jumps made
% estimatedrows = round(k*(k-1)/2); % Risk of create more rows than needed.
Etree = zeros(1,7); % Under estimation of rows for safety
count = 0;

%% Pre-merging step before Energy jumps
% Consider all clusters with V < ERR
% This process merges all clusters which are really close to consider as 1.
for i=1:k
    for j=1:k
       if dErtemp(i,j) < ERR
           count = count +1;
           Emin = dErtemp(i,j);
           Etree(count,1:5) = [i, j, Emin, ERR, 1];
           dErtemp(i,j) = NaN;
           fullclusters(i,j) = 1;
           if isnan(dErtemp(j,i))
               Etree(count,5) = 0;
           end           
       end
    end
end
Emin0 = ERR;
indexlabel0 = count;

%% Bucle for Energy jumps with E > 0 initially
while ~all(isnan(dErtemp(:)))
    count = count + 1;
    Emasked = dErtemp;
    Emasked(any(dErtemp==0, 2),:) = NaN;
    if ~any(~isnan(Emasked(:)))
        break;
    end
    
    [Emin, Eind] = min(Emasked(:));
    [K0, K1] = ind2sub(size(dErtemp),Eind);
    
    fullclusters(K0,K1) = 1;
    Etree(count,1:5) = [K0, K1, Emin, Emin+Emin0,1];
    
    dErtemp = dErtemp - Emin*(~any(dErtemp==0, 2))*ones(1,k);
    
    if fullclusters(K1,K0)==1
        dErtemp(K1,K0) = NaN;
        dErtemp(K0,K1) = NaN;
        Etree(count,5) = 0; % When is cero means that is a leveling jump.
    end
    
    % This section tries to record the jumps made from K1 to another
    % clusters, in such way that K0 jumps to tnew(last K1 destination)
    t = K1;
    tnew0 = zeros(1,1); % Create a generic vector
    while any(Etree(Etree(:,5)==1,1)==t) && fullclusters(t,K0)==0 % After 
        % K0 to K1 jump, does it exist any jump from t(K1) to other cluster
        % and also t(K1) to K0 not be made previously? If true, do while.
        tnew = Etree(Etree(:,1)==t & Etree(:,5)==1,2);
        if any(ismember(tnew,tnew0))
            break;
        end
        count = count + 1;
        Etree(count,1:5) = [K0, tnew(1), 0, Emin+Emin0, 1]; % tnew(1) is the
        % first jump that fullfiles these conditions.
        t = tnew(1);
        tnew0 = [tnew0;tnew];
    end
    Emin0 = Emin + Emin0;
end

%% Merging labels based on label summation in a diagonal matrix
clustermerging = logical(diag(ones(k,1)));
% tracksolutions = zeros(k,k,size(Etree,1));
tracksol = struct([]);


for i=1:size(Etree,1)
    K0 = Etree(i,1);
    K1 = Etree(i,2);
    likeK0 = find(all(repmat(clustermerging(K0,:),size(clustermerging,1),1)...
        == clustermerging,2));
    likeK1 = find(all(repmat(clustermerging(K1,:),size(clustermerging,1),1)...
        == clustermerging,2));
    mergesolution = clustermerging(K0,:) | clustermerging(K1,:);
    
    clustermerging(likeK0,:) = repmat(mergesolution,size(likeK0,1),1);
    clustermerging(likeK1,:) = repmat(mergesolution,size(likeK1,1),1);
    tracksol(i).clmerg = clustermerging;
    Etree(i,6) = size(unique(clustermerging,'rows'),1);
end

%% Identifying which jumps imply a new label solution
Etree(1,7) = 1;
for i=2:size(Etree,1)
%     Etree(i,7) = ~isequal(tracksolutions(:,:,i-1),tracksolutions(:,:,i));
    Etree(i,7) = ~isequal(tracksol(i-1).clmerg, tracksol(i).clmerg);
end

%% Function to generate the clusters solutions over each jump (columns)
mergesteps = sum(Etree(:,7)==1);
indextree = find(Etree(:,7)==1);
aux = 1:k;
m = size(labels,1);
labelstree = zeros(m,mergesteps+1);
labelstree(:,1) = labels;
for i=1:mergesteps
    % Create an unique matrix with the elements (clusters) to include as
    % one label. Then the labels are named by steps of temp rows.
    temp = logical(unique(tracksol(indextree(i)).clmerg,'rows'));
    for j=1:size(temp,1)
        auxtemp = aux(temp(j,:));
        indexsolutions = false(m,1);
        for f=1:length(auxtemp)
           indexsolutions = indexsolutions | labels==auxtemp(f); 
        end
        labelstree(indexsolutions,i+1) = min(auxtemp);
    end
end

%% Identifying the premerge solution, using indexlabel0.
if indexlabel0>0
    label0 = zeros(m,1);
%     temp = logical(unique(tracksolutions(:,:,indexlabel0),'rows'));
    temp = logical(unique(tracksol(indexlabel0).clmerg,'rows'));
    for j=1:size(temp,1)
        auxtemp = aux(temp(j,:));
        indexsolutions = false(m,1);
        for f=1:length(auxtemp)
            indexsolutions = indexsolutions | labels==auxtemp(f);
        end
        label0(indexsolutions) = min(auxtemp);
    end
    Klabel0 = size(temp,1);
else
    label0 = labels;
    Klabel0 = length(unique(labels));
end

%%
Etree2 = zeros(size(labelstree,2),2);
% Etree(1,2) = Etree0(1,4)./max(Etree0(Etree0(:,7)==1,4));
% Etree(2:end,2) = Etree0(Etree0(:,7)==1,4)./max(Etree0(Etree0(:,7)==1,4));

% No normalizing energies
Etree2(1,2) = Etree(1,4);
Etree2(2:end,2) = Etree(Etree(:,7)==1,4);


for i=1:size(Etree2)
    Etree2(i,1) = length(unique(labelstree(:,i)));
end

end
