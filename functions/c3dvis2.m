% function [ output_args ] = Untitled( input_args )
% %UNTITLED Summary of this function goes here
% %   Detailed explanation goes here
function [dataout, v_M1, d_M1] = c3dvis2(inputdata, kdim)
%InputsClusterVisualisation_new.m
%Inputs; Inps; RInputs;

%%%%PJL

%%%Read in data and set the global variables
% Load the data as a flat file with additional columns with cluster indices
% filedata = importdata(datafile);

% [ndata, header]=xlsread('data3.xls');
ndata = inputdata;
% Read in all relevant input variables ...
Inputs=ndata(:,1:end-1);
% ... and the corresponding cluster allocations
init_alloc=ndata(:,end);
%%
[npar, nvar] =size(Inputs);
Inputs = Inputs-ones(npar,1)*mean(Inputs);
prot_alloc = init_alloc; % Labels
%%%%PJL

Major_axes_thres=0.0; %10.0;
Scaling_factor = 1.0; %1.5 for Marketing 1.0 for Daliadata
Scaling_fact = 1.0; %1.5 for marketing data  600.0 for Daliadata

% Read-in data and cluster allocation vector
Data=Inputs;

if nvar<3
    kdim = nvar;
end

Euclid=0;
if Euclid==1
    % Normalise data to unit Euclidean length
    [npar, nvar]=size(Inputs);
    for j=1:npar
        Data(j,:)=Inputs(j,:)/max(sqrt(Inputs(j,:)*Inputs(j,:)'), 0.000001);
    end
end; %Euclid=2;

%Cluster_alloc=init_alloc;
b = size(prot_alloc,2); % R: b uses to be = 1
Cluster_alloc = prot_alloc(:,b);

d = unique(Cluster_alloc);


Cluster_number = length(d);

[N_Inputs, Inp_dimension]=size(Data);

% Set Prototypes to cluster means
P = zeros(Cluster_number, Inp_dimension);
for i=1:Cluster_number
    Cluster_Inputs = Data(Cluster_alloc==d(i),:);
    P(i,:) = mean(Cluster_Inputs);
end;

% Enable following trainKM/ART2005Nott
%%%%PJL [share segment]=sort(-market_share);
% Enable following trainKM/ART2005Nott
%%%%PJL d=segment(1:Cluster_number);
% R: This defines the maximum elements per each direction in the prototypes
major_axes=ceil(P(d,:)-Major_axes_thres);
major_axes=sum(major_axes);
major_axes=min(major_axes,1);
major_axes=find(major_axes>0);

% R: Order observations by labels
Cluster_Data = zeros(N_Inputs, Inp_dimension);
j = 1;
for i = 1:Cluster_number
    ind = Cluster_alloc==d(i);
    Cluster_Data(j:j+sum(ind)-1,:) = Data(ind,:);
    j = j+ sum(ind);
end;

%Define cluster performance indices using scatter matrices
Data_dimension = size(Cluster_Data,2);

Sw = zeros(Data_dimension,Data_dimension);
Sb = Sw;
Cluster_mean = zeros(Cluster_number,Data_dimension);
Cluster_size = zeros(Cluster_number,1);
Overall_mean = mean(Cluster_Data);
for i=1:Cluster_number
    X_i = Data(Cluster_alloc==d(i),:);
    clust_size = size(X_i,1);
    Cluster_size(i) = clust_size;
    Cluster_mean(i,:) = mean(X_i);
    X_i = X_i-ones(clust_size,1)*Cluster_mean(i,:);
    Sw = Sw+X_i'*X_i;
    Sb = Sb+Cluster_size(i)*(Cluster_mean(i,:)-Overall_mean)'*(Cluster_mean(i,:)-Overall_mean);
end;
% Sep_Matrix=inv(Sw)*Sb;
Sep_Matrix = Sw\Sb;

fprintf('\n Original data');
fprintf('\n Cluster performance indices: tr(Sb)= %5.2f tr(Sw)= %5.2f tr(Sb)/tr(Sw)= %5.2f',trace(Sb),trace(Sw),trace(Sb)/trace(Sw));
fprintf('\n Invariant criterion:         tr(inv(Sw)*Sb)= %5.2f\n',trace(Sep_Matrix));


% print out detailed results
% Sw1=Sw;
% [v_Sw1, d_Sw1] = eig(Sw);
% Sb1 = Sb;
% M1=inv(Sw)*Sb;
M1 = Sw\Sb;
[v_M1, d_M1] = eig(M1);
% J1=trace(M1);
%
dataout = Data* v_M1;
% Project using the first 3 eigenvalues of the scatter matrix
[evalue, index] = sort(diag(d_M1),'descend');
VV = v_M1(:,index(1:kdim));
% VV_scatter_matrix = VV;%!!!
fprintf('\n The first 3 eigenvalues of the scatter matrix from %2d explain %4.1f of the total\n',(nvar-1), 100*sum(evalue(index(1:kdim)))/sum(evalue(index)));

% Cluster_numbers clusters
%
%

if kdim > 1
    figure;
    
    P3 = P*VV;
    % % Variable orth_axes does not exist!!
    legendplot = cell(1,1);
    j1 = 1;
    for i=1:Cluster_number
        temp = Data(Cluster_alloc==d(i),:)*VV;
        %%%  For new data
        %    temp=New_data_vector*VV;
        %%%%
        ii =  rem(i-1,12)+1;
        if kdim ==2
            switch ii
                case 1
                    plot(temp(:,1),temp(:,2),'r+');
                    hold all;
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'r--');
                    grid on
                case 2
                    plot(temp(:,1),temp(:,2),'go');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'g--');
                case 3
                    plot(temp(:,1),temp(:,2),'ms');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'m--');
                case 4
                    plot(temp(:,1),temp(:,2),'c*');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'c--');
                case 5
                    plot(temp(:,1),temp(:,2),'bv');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'b--');
                case 6
                    plot(temp(:,1),temp(:,2),'kp');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'k--');
                case 7
                    plot(temp(:,1),temp(:,2),'y.');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'y--');
                case 8
                    plot(temp(:,1),temp(:,2),'kx');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'k--');
                case 9
                    plot(temp(:,1),temp(:,2),'kd');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'k--');
                case 10
                    plot(temp(:,1),temp(:,2),'kh');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'k--');
                case 11
                    plot(temp(:,1),temp(:,2),'k<');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'k--');
                case 12
                    plot(temp(:,1),temp(:,2),'k>');
                    plot([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],'k--');
            end;
            
        else
            switch ii
                case 1
                    plot3(temp(:,1),temp(:,2),temp(:,3),'r+');
                    hold all;
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'r--');
                    grid on
                case 2
                    plot3(temp(:,1),temp(:,2),temp(:,3),'go');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'g--');
                case 3
                    plot3(temp(:,1),temp(:,2),temp(:,3),'ms');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'m--');
                case 4
                    plot3(temp(:,1),temp(:,2),temp(:,3),'c*');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'c--');
                case 5
                    plot3(temp(:,1),temp(:,2),temp(:,3),'bv');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'b--');
                case 6
                    plot3(temp(:,1),temp(:,2),temp(:,3),'kp');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'k--');
                case 7
                    plot3(temp(:,1),temp(:,2),temp(:,3),'y.');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'y--');
                case 8
                    plot3(temp(:,1),temp(:,2),temp(:,3),'kx');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'k--');
                case 9
                    plot3(temp(:,1),temp(:,2),temp(:,3),'kd');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'k--');
                case 10
                    plot3(temp(:,1),temp(:,2),temp(:,3),'kh');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'k--');
                case 11
                    plot3(temp(:,1),temp(:,2),temp(:,3),'k<');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'k--');
                case 12
                    plot3(temp(:,1),temp(:,2),temp(:,3),'k>');
                    plot3([0,Scaling_factor*P3(i,1)],[0,Scaling_factor*P3(i,2)],[0,Scaling_factor*P3(i,3)],'k--');
            end;
            
        end
        legendplot{j1} = ['Cluster ',num2str(d(i))];
        legendplot{j1+1} = ['Centroid ',num2str(d(i))];
        j1 = j1+2;
    end;
    % legend(legendplot)
    %%
    %%
    
    %% Show major categories
    for cat=1:length(major_axes)
        cat_axis = zeros(1,nvar);
        cat_axis(major_axes(cat))=1;
        cat3=cat_axis*VV;
        %% Normalise category axes
        %  cat3=cat3/sqrt(cat3*cat3');
        if kdim ==2
            plot([0,Scaling_fact*cat3(1)],[0,Scaling_fact*cat3(2)]);
            H=text(Scaling_fact*cat3(1),Scaling_fact*cat3(2),num2str(major_axes(cat)));
        else
            plot3([0,Scaling_fact*cat3(1)],[0,Scaling_fact*cat3(2)],[0,Scaling_fact*cat3(3)]);
            H=text(Scaling_fact*cat3(1),Scaling_fact*cat3(2),Scaling_fact*cat3(3),num2str(major_axes(cat)));
        end
        set(H,'Color','b');
        legendplot{j1} = ['Eigvect',num2str(cat)];
        j1 = j1+1;
    end
    % legend('show')
    legend(legendplot,'Location','best')
    title('Multiple discriminant analysis, first 3 eigenvalues scatter matrix');
end

%% Define cluster performance indices using scatter matrices
Data3lambda=Data*VV;
Data_dimension = size(Data3lambda,2);
Sw=zeros(Data_dimension,Data_dimension);
Sb=Sw;
Cluster_mean=zeros(Cluster_number,Data_dimension);
Cluster_size=zeros(Cluster_number,1);
Overall_mean=mean(Data3lambda);
for i=1:Cluster_number
    X_i = Data3lambda(Cluster_alloc==d(i),:);
    clust_size = size(X_i,1);
    Cluster_size(i) = clust_size;
    Cluster_mean(i,:) = mean(X_i);
    X_i = X_i-ones(clust_size,1)*Cluster_mean(i,:);
    Sw = Sw+X_i'*X_i;
    Sb=Sb+Cluster_size(i)*(Cluster_mean(i,:)-Overall_mean)'*(Cluster_mean(i,:)-Overall_mean);
end
Sep_Matrix = Sw\Sb;
fprintf('\n Original data projected onto 3D using the largest 3 eigenvalues of the scatter matrices in the original domain');
fprintf('\n Cluster performance indices: tr(Sb)= %5.2f tr(Sw)= %5.2f tr(Sb)/tr(Sw)= %5.2f',trace(Sb),trace(Sw),trace(Sb)/trace(Sw));
fprintf('\n Invariant criterion:         tr(inv(Sw)*Sb)= %5.2f\n',trace(Sep_Matrix));




end

