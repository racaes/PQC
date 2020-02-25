function [problab,outlier_ind,outlier_ratio,LL] = proballoc(datagen,sigmaknn,label0,outlier)
% Function to allocate the clusters according the maximum probability given
% the data: max( P(K|X))

if ~exist('outlier','var')
    outlier = 0.05;
end;


[~, ~, ~, Px_k, Pk_x_match] = probQC(datagen,sigmaknn,datagen,label0);
[Pk_x, Pk_x_index] = max(Pk_x_match,[],2);
lab0=unique(label0);
problab = lab0(Pk_x_index);
LL = sum(log(Pk_x));
[maxPx_k, ~] = max(Px_k,[],2);
outlier_ind = maxPx_k < outlier;
% This only works when datallo is datagen
outlier_ratio = sum(outlier_ind)/size(datagen,1);
