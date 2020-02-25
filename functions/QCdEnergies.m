function dE = QCdEnergies(datagen,sigma,centroids,K,QCsetup)

if size(sigma,2)>1
   QC3 = true;
else
    QC3 = false;
end

alldata = [datagen; centroids];
if QC3
    [Vall,~,Eall] = qc3_eig_v3(datagen,sigma,alldata);
else
    [Vall,~,Eall] = qc2sig2(datagen,sigma,alldata);
end

lambda1 = mean(sum(alldata.^2,2).^0.5);
lambda2 = mean(abs(Vall - Eall));

alldataV = [alldata./lambda1, (Vall - Eall)./lambda2];

distallV = squareform(pdist(alldataV,'euclidean'));

% tic

dE = zeros(K);
for i=1:K-1
    for j=i+1:K
        path = qcshortestpath(alldata,distallV,i,j, QCsetup.Minstep,K);
        
        if QC3
            [Vi,~,Ei,~] = qc3_eig_v3(datagen,sigma,path);
        else
            [Vi,~,Ei,~] = qc2sig2(datagen,sigma,path);
        end
        Vi = Vi - Ei;
        dE(i,j) = max(Vi) - Vi(1);
        dE(j,i) = max(Vi) - Vi(end);
    end
end
%     toc
% dEr = dE./max(max(dE));
% E2merge = (max(dE(:))-min(dE(:)))*QCsetup.Tolmerge;

% E2merge = max(maxERR,QCsetup.Emerge);
end
