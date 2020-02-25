clc
clear all
close all

%% Load data

opt = 1;
switch opt
    case 1
        load('datasets4.mat','data1')
        data = data1.data;
        class = data1.class;
    case 2
        load('datasets4.mat','data2')
        data = data2.data;
        class = data2.class;
    case 3
        load('datasets4.mat','data3')
        data = data3(1).data;
        class = data3(1).class;
    case 4
        load('datasets4.mat','data4')
        data = data4(1).data;
        class = data4(1).class;
    case 5
        load('datasets4.mat','data5')
        data = data5(1).data;
        class = data5(1).class;        
end

%% Preprocessing

% % Data from JMLR in 'datasets4.mat' is already preprocessed.
% Only zscoring if features are not related themselves as a distance. Not
% in Euclidean embedding!!
[data1,mu,s] = zscore(data);
lambda = mean(sqrt(sum(data.^2,2)));
data = data./lambda;

%% Plot data

if size(data,2)==3
    figure;
    scatter3(data(:,1), data(:,2), data(:,3), 20, class)
    xlabel('X1')
    ylabel('X2')
    zlabel('X3')
elseif size(data,2)==2
    figure;
    gscatter(data(:,1),data(:,2),class)
    grid minor
    title(['Data #', num2str(opt)])
    xlabel('X1')
    ylabel('X2')
else
    for c_col=1:size(class,2)
        figure;
        [coeff, score, latent] = pca(data);
        scatter3(score(:,1),score(:,2),score(:,3), 20, class(:,c_col))
        title(['PCA Data #', num2str(opt), 'Class column: ', num2str(c_col)])
        grid minor
        xlabel('PCA1')
        ylabel('PCA2')
        zlabel('PCA3')
    end
end

%% Histograms 2D
figure
h = histogram2(data(:,1), data(:,2),'FaceColor','flat');
colorbar

%% QC SETUP
QCsetup = struct;
QCsetup.steps = 1000;
QCsetup.eta = 0.005; % Default is eta = 0.001;
QCsetup.b1 = 0.9;
QCsetup.b2 = 0.999;
QCsetup.ep = 1e-8;
QCsetup.showProgress = 0;
QCsetup.track = 0;
QCsetup.ERR = 1e-3;
QCsetup.Minstep = 0.025; % Maximum distance to travel in centroids paths
QCsetup.Emerge = 0.001; % Absolute energy to merge K, minimum is ERR after SGD

%% Select QC model
% QC3 = false; % QC2
QC3 = true; % QC3
q = 2;
if QC3 == true
    q = 3;
end

snr = 1;
lambda = mean(sqrt(sum(data.^2,2)));
data = data./lambda;
datagen = data;
datallo = data;

%% Scan %knn
scan_knn = 1;
if scan_knn == true
    qtity2 = 5;
    qtile = linspace(0.07,0.35,qtity2);
else
    qtity2 = 1;
    qtile = 0.15;
end

%% Scan dE
scan_dE = 1;
if scan_dE == true
    qtity1 = 20;
    energy = logspace(-3.5,2,qtity1);
else
    qtity1 = 1;
    energy = 0.01;
end

%% QC ANALYSIS
Energies = struct([]);
clusters = zeros(qtity1,qtity2,3);
ANLLdata  = zeros(qtity1,qtity2);
maxERRdata  = zeros(qtity1,qtity2);
datalabels = zeros(size(data,1),qtity1,qtity2);

if size(class,2)==2
    crdata = zeros(qtity1,qtity2,2);
    jsdata = zeros(qtity1,qtity2,2);
else
    crdata = zeros(qtity1,qtity2);
    jsdata = zeros(qtity1,qtity2);
end

for i=1:qtity2
    
    noise = qtile(i)*snr;
    local = qtile(i);
    
    if ~QC3
        dist2 = squareform(pdist(datagen,'euclidean'));
        [dist2sort,~] = sort(dist2);
        m1 = size(datagen,1);
        sigmaknn = mean(dist2sort(2:ceil((m1-1)*local)+1,:),1)';
        
        [K,labels,maxERR,sigma,centroids]=QC2main_v1(datagen,...
            sigmaknn,datallo,QCsetup);
    else
        [K,labels,maxERR,sigma,centroids]=QC3main_eig_v5(datagen,...
            noise, local,datallo,QCsetup);
    end
    
    dE = QCdEnergies(datagen,sigma,centroids,K,QCsetup);
    Energies(i).dE = dE(~eye(size(dE,1)));
    
    for j=1:qtity1
        
        %         disp(opt*100000+qct*10000+100*i+j)
        [~, ~, ~,K0,label0,Etree] = Etreefun_v3(dE,labels,energy(j));
        
        if j==1
            Energies(i).knn = Etree;
        end
        
        if QC3 == false
            [~, ~, ~, ~, Pk_x] = probQC(datagen,sigma,datallo,label0);
            [Pk_x_max1, Pk_x_index] = max(Pk_x,[],2);
            lab0=unique(label0);
            problabqc3 = lab0(Pk_x_index);
            
        else
            [~, ~, ~, ~, Pk_x] = ProbQC3(datagen,sigma,datallo,label0);
            [Pk_x_max1, Pk_x_index] = max(Pk_x,[],2);
            lab0=unique(label0);
            problabqc3 = lab0(Pk_x_index);
        end
        
        clusters(j,i,1) = K;
        clusters(j,i,2) = K0;
        clusters(j,i,3) = length(unique(problabqc3));
        ANLLdata(j,i) = -sum(log(Pk_x_max1))/size(Pk_x_max1,1);
        maxERRdata(j,i) = maxERR;
        datalabels(:,j,i)= problabqc3;

        if size(class,2)==2
            jsdata(j,i,1) = myClustMeasure2(problabqc3,class(:,1));
            jsdata(j,i,2) = myClustMeasure2(problabqc3,class(:,2));
            
            crdata(j,i,1) = cramer(problabqc3,class(:,1));
            crdata(j,i,2) = cramer(problabqc3,class(:,2));
        else
            jsdata(j,i) = myClustMeasure2(problabqc3,class(:,1));
            crdata(j,i) = cramer(problabqc3,class(:,1));
        end
    end
end


%% SAVE results?
saveok = 0;
if saveok == true
    save(['PQC_main_Dat',num2str(opt),'_QC',num2str(q),'.mat'],'clusters',...
        'ANLLdata','maxERRdata','jsdata','crdata','qtile','energy','Energies')
end

%% PLOT RESULTS

%% Single solution for 2D data
if ~scan_knn && ~scan_dE == true
    figure
    gscatter(data(:,1),data(:,2),problabqc3)
    title(['QC',num2str(q),': JS = ',num2str(myClustMeasure2(problabqc3,class(:,1)),3),...
        ', qtile = ',num2str(qtile),', dE = ',num2str(energy,2)])
    grid minor
end

%% Probabilistic map for 2D data
if ~scan_knn && ~scan_dE == true && size(datagen,2) == 2
    gridsize = 30;
    datagrid = zeros(gridsize,2);
    datagrid(:,1) = linspace(min(datagen(:,1)), max(datagen(:,1)),gridsize);
    datagrid(:,2) = linspace(min(datagen(:,2)), max(datagen(:,2)),gridsize);
    
    datagrid2 = zeros(gridsize^2,1);
    [gridx, gridy] = meshgrid(datagrid(:,1),datagrid(:,2));
    datagrid2(:,1) = reshape(gridx,[],1);
    datagrid2(:,2) = reshape(gridy,[],1);
    
    if QC3 == false
        [Pjoint, ~, ~, Px_k_grid, Pk_x_grid] = probQC(datagen,sigma,datagrid2,label0);
    else
        [Pjoint, ~, ~, Px_k_grid, Pk_x_grid] = ProbQC3(datagen,sigma,datagrid2,label0);
    end
        
    Px_k_max = max(Px_k_grid,[],2);
    
    outliers = 0;
    if outliers == true
        outlier = zeros(gridsize^2,1);
        outlier(Px_k_max < 0.05) = 1;
        gridoutlier = reshape(outlier,gridsize,gridsize);
    end
    
    % % % % % % % % % P(K|X) % % % % % % % % %
    leg = [];
    leg{size(Pk_x_grid,2)} = [];
    h=figure('Name', 'P(K|X) map');
    set(h,'Position',[156 186 880 762]);
    for i=1:size(Pk_x_grid,2)
        gridProb = reshape(Pk_x_grid(:,i),gridsize,gridsize);
        surf(gridx, gridy, gridProb, repmat(i,gridsize,gridsize))
        %     colormap(cool)
        hold all
        leg{i} = num2str(i);
    end
    
    if outliers == true
        surf(gridx, gridy, gridoutlier, zeros(gridsize,gridsize))%, repmat(i,gridsize,gridsize))
        leg{i+1} = 'Outlier';
    end
    title('P(K|X)')
    legend(leg)
    
    % % % % % % % % % P(X|K) Heat map % % % % % % % % %
    colormap default
    aux = Px_k_max;
    aux(aux>1) = 1;
    v = [0, 0.01,0.05,0.1,0.2,0.35,0.5,0.75,1];
    h=figure('Name', 'P(X|K) map');
    gridProb = reshape(aux,gridsize,gridsize);
    contourf(gridx, gridy, gridProb,v,'ShowText','on')
    contourcbar
    title('max_K P(X|K)')
    
end


%% ANLL 3D
if scan_knn && scan_dE == true
    
    dataname = {'Art.#1','Art.#2','Crabs','Olive','Seeds'};
    qcname = {'QC_{knn}^{prob}','QC_{cov}^{prob}'};
    
    ANLLmod = ANLLdata(:);
    K = reshape(clusters(:,:,3),[],1);
    %         ANLLmod(K==1) = 1;
    ANLLmod(K==1) = max(ANLLmod);
    ANLLmod = reshape(ANLLmod, size(ANLLdata));
    
    h = figure;
    surf(qtile,log10(energy),clusters(:,:,3))
    alpha(0.3)
    ylabel('log_{10}(E threshold)')
    xlabel('%knn')
    zlabel('#K')
    title(['# Clusters ',qcname{q-1}])
%     title(['# Clusters ',qcname{q-1},'. ',dataname{r},' data.'])
    %         savefig(h,['JMLR_R',num2str(r),'_QC',num2str(q),'_clusters.fig']);
    
    h = figure;
    surf(qtile,log10(energy),ANLLdata)
    alpha(0.3)
    ylabel('log_{10}(E threshold)')
    xlabel('%knn')
    zlabel('ANLL')
    title(['ANLL ',qcname{q-1}])
%     title(['ANLL ',qcname{q-1},'. ',dataname{r},' data.'])
    %         savefig(h,['JMLR_R',num2str(r),'_QC',num2str(q),'_ANLL.fig']);
    
    h = figure;
    surf(qtile,log10(energy),ANLLmod)
    alpha(0.3)
    ylabel('log_{10}(E threshold)')
    xlabel('%knn')
    zlabel('ANLLmod')
    title(['ANLLmod ',qcname{q-1}])
%     title(['ANLLmod ',qcname{q-1},'. ',dataname{r},' data.'])
    %         savefig(h,['JMLR_R',num2str(r),'_QC',num2str(q),'_ANLLmod.fig']);
    
    if size(jsdata,3)==2
        h = figure;
        surf(qtile,log10(energy),jsdata(:,:,1))
        alpha(0.3)
        ylabel('log_{10}(E threshold)')
        xlabel('%knn')
        zlabel('Jaccard1')
        title(['Jaccard1 ',qcname{q-1}])
%         title(['Jaccard1 ',qcname{q-1},'. ',dataname{r},' data.'])
        
        h = figure;
        surf(qtile,log10(energy),jsdata(:,:,2))
        alpha(0.3)
        ylabel('log_{10}(E threshold)')
        xlabel('%knn')
        zlabel('Jaccard2')
        title(['Jaccard2 ',qcname{q-1}])
%         title(['Jaccard2 ',qcname{q-1},'. ',dataname{r},' data.'])
        
    else
        h = figure;
        surf(qtile,log10(energy),jsdata)
        alpha(0.3)
        ylabel('log_{10}(E threshold)')
        xlabel('%knn')
        zlabel('Jaccard')
        title(['Jaccard ',qcname{q-1}])
%         title(['Jaccard ',qcname{q-1},'. ',dataname{r},' data.'])
    end
    
end

%%
