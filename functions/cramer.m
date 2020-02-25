function [Cmed, C, pvalmed, pval] = cramer(classref, classother)
% This function computes the Cramer's V pairwise respect the classref and
% other classes.
% Input: Vectors of cluster assignments
% Output: median of cramer coef. and the vector of all pairwise solutions

[m1, n1] = size(classref);
[m2,n2]=size(classother);

if n1~=1
    warning('Class of reference has to be a vector of dimension 1.')
end

if m1~=m2
    warning('Vectors to compare have different number of rows.')
end
C=zeros(n2,1);
pval=zeros(n2,1);
for i=1:n2
    ct = crosstab(classref,classother(:,i));
    N = sum(sum(ct));
    Ei =sum(ct,1);
    Ej = sum(ct,2);
    chimat = ((ct - Ej*Ei./N).^2)./(Ej*Ei./N);
    C(i) = sqrt( sum(sum(chimat))./(N * (min(size(ct))-1)));
    pval(i) = 1 - chi2cdf(sum(chimat(:)),(size(ct,1)-1)*(size(ct,2)-1));
end
pvalmed = nanmedian(pval);
Cmed = nanmedian(C);

end
