function [K,labels,maxERR,sigmaknn,centroids]=QC2main_v1(datagen,sigmaknn,datallo,QCsetup)

[Dnew,Vlast,tracking,~,~,maxERR,maxERRdist] = ...
    graddescVsig(datagen,sigmaknn,datallo,QCsetup);

[labels, centroids, K] = QCalloc5(Dnew,Vlast,maxERR);

