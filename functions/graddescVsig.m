function [Dnew,Vnew,tracking,trackdata,trackV,maxERR,maxERRdist] = ...
    graddescVsig(datagen,sigma,datallo,QCsetup)
% function graddesc(xyData,q,[steps])
% purpose: performing quantum clustering in and moving the
%          data points down the potential gradient
% INPUTS: 
%  - datage: Data generating the potential
%  - q: Paramater that controls the gaussian smoothness 1/(2*sigma^2)
%  - datallo: Data to allocate in the potential
%  - Next values correspond to the QCsetup struct:
% 
%  - steps: Minimum number of steps to assure the gradient descent
%  - eta: Learning rate
%  - b1: Parameter that controls the momentum decay in ADAM SGD
%  - b2: Parameter that controls the variance decay in ADAM SGD
%  - ep: Smoothness term in RMS when 1st and 2nd momentums are very low
%  - showProgress: To enable the graphics display
%  - track: To store the solutions per step during SGD
%  - ERR: Convergence error
%        
% OUTPUTS:
% - D: Location of data after SGD
% - V: Potential values at the end
% - tracking: Convergence per SGD step
% - trackdata: Data positions per SGD step
% - trackV: Potentials per SGD step
%
%
% Mod Raul 21/07/2016: Using ADAM SGD when eta~=0 and ADADELTA when eta=0.
% Mod Raul 01/08/2016: Potential convergence criterion has been added.
% Warning: ADADELTA show problems of convergence. Use ADAM (default)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Default parameters
if ~isfield(QCsetup,'steps')
    QCsetup.steps = 100;
end
if ~isfield(QCsetup,'eta')
    QCsetup.eta = 0.001;
end
if ~isfield(QCsetup,'b1')
    QCsetup.b1=0.9;
end
if ~isfield(QCsetup,'b2')
    QCsetup.b2=0.999;
end
if ~isfield(QCsetup,'ep')
    QCsetup.ep = 1e-8;
end
if ~isfield(QCsetup,'showProgress')
    QCsetup.showProgress=0;
end
if ~isfield(QCsetup,'track')
    QCsetup.track=0;
end
if ~isfield(QCsetup,'ERR')
    QCsetup.ERR = 1e-4;
end

% Assigning parameter values according QCsetup struct.
steps = QCsetup.steps;
eta = QCsetup.eta; % Default is eta = 0.001;
b1 = QCsetup.b1;
b2 = QCsetup.b2;
ep = QCsetup.ep;
showProgress = QCsetup.showProgress;
track = QCsetup.track;
ERR = QCsetup.ERR;


% Variables initialitation
[Vold,~,E,dV] = qc2sig2(datagen,sigma,datallo);
Vold = Vold - E;
Dold = datallo;
[m,n] = size(Dold);

if track
    % Convergence control
    tracking = zeros(steps,4); % Info about convergence criteria
    trackdata = zeros(m,n,steps); % Info about data at each step
    trackV = zeros(m,steps); % Potential in each step
    trackdata(:,:,1) = datallo; % Initial data allocation
    trackV(:,1) = Vold; % Initial potential
else
    tracking = [];
    trackdata = [];
    trackV = [];
end

if showProgress && track
    % Mesh generation for ploting the potential
    gridsize = 25;
    datagrid(:,1) = linspace(min(datagen(:,1)), max(datagen(:,1)),gridsize);
    datagrid(:,2) = linspace(min(datagen(:,2)), max(datagen(:,2)),gridsize);
    if n>2
        datagrid(:,3) = linspace(min(datagen(:,3)), max(datagen(:,3)),gridsize);
        [gridx, gridy,gridz] = meshgrid(datagrid(:,1),datagrid(:,2),datagrid(:,3));
        datagrid2(:,1) = reshape(gridx,[],1);
        datagrid2(:,2) = reshape(gridy,[],1);
        datagrid2(:,3) = reshape(gridz,[],1);
    elseif n==2
        [gridx, gridy] = meshgrid(datagrid(:,1),datagrid(:,2));
        datagrid2(:,1) = reshape(gridx,[],1);
        datagrid2(:,2) = reshape(gridy,[],1);
    end
    [Vmesh,~,~,dVgrid] = qc2sig2(datagen(:,1:size(datagrid2,2)),sigma,datagrid2);
    Vmesh = Vmesh./max(Vmesh);
    
    if n==2
        gridV = reshape(Vmesh,gridsize,gridsize);
    elseif n>=3
        gridV = reshape(Vmesh,gridsize,gridsize,gridsize);
        gridx = gridx(:,:,round(gridsize/2));
        gridy = gridy(:,:,round(gridsize/2));
        gridV = gridV(:,:,round(gridsize/2));
    end
      
    
    % Plot figures
    h=figure('Name', 'Potential_performance');
    set(h,'Position',[156 186 880 762]);
    
    subplot(2,2,1)
    contour(gridx, gridy, gridV)
    hold on
    plot(datagen(:,1),datagen(:,2),'*k')
    xlabel('PC1')
    ylabel('PC2')
    title('Potential and allocation')
    hold off
    
    subplot(2,2,2)
    contour(gridx, gridy, gridV)
    hold on
    quiver(datagrid2(:,1), datagrid2(:,2), -dVgrid(:,1), -dVgrid(:,2))
    xlabel('PC1')
    ylabel('PC2')
    title('-Grad(V)')
    hold off
    
    subplot(2,2,3)
    refline([0,ERR])
    grid minor
    title('max( distance(Dew - Dold))')
    
    subplot(2,2,4)
    refline([0,ERR])
    title('max( abs(Vold - Vnew))')
    grid minor
end



convergence = ERR+1;
convergenceV = ERR+1;
if eta~=0 % ADAM SGD
    m0 = zeros(m,n);
    v0 = zeros(m,n);
else % ADADELTA SGD
    Eg20 = zeros(m,n);
    Edx20 = zeros(m,n);
    dx = zeros(m,n);    
end
t = 0; % Steps counter

while (t <= steps) && (( convergence >= ERR) || ( convergenceV >= ERR))
    t = t+1;
    
    if eta~=0
    mt = b1*m0 + (1-b1)*dV; % gradient with decay term (momentum)
    vt = b2*v0 + (1-b2)*dV.^2; % grad variance (no centered) w/decay
    
    me = mt / (1 - b1^t); % Bias correction to first moment
    ve = vt / (1 - b2^t); % Bias correction to second moment    
    
    Dnew = Dold - eta.*me./(ve.^0.5 +ep); % Data allocation update        
        
    m0 = mt; % Update old variables
    v0 = vt; 
    else
        Eg2 = b1*Eg20 + (1-b1)*dV.^2;
        Edx2 = b1*Edx20 + (1-b1)*dx.^2;
        
        dx = - sqrt(Edx2 + ep)./sqrt(Eg2 + ep).*dV;
        Dnew = Dold + dx;
        
        Eg20 = Eg2; % Update old variables
        Edx20 = Edx2;
    end
    [V,~,E,dV] = qc2sig2(datagen,sigma,Dnew);
    Vnew = V - E;
    convergence = max(sqrt(sum((Dnew - Dold).^2,2)));
    convergenceV = max(abs(Vold - Vnew));
    convergence_mean = mean(sqrt(sum((Dnew - Dold).^2,2)));
    Dold = Dnew;
    Vold = Vnew;
    
    if track
        % Convergence control
        tracking(t,:)= [t,convergence, convergence_mean, convergenceV];
        trackdata(:,:,t+1) = Dnew;
        trackV(:,t+1) = Vnew;
    end
    
    
    if showProgress && track
        subplot(2,2,1)
        contour(gridx, gridy, gridV)
        hold on
        plot(Dnew(:,1),Dnew(:,2),'*k')
        xlabel('PC1')
        ylabel('PC2')
        title('Potential and allocation')        
        hold off
        
        subplot(2,2,2)
        contour(gridx, gridy, gridV)
        hold on
        quiver(Dnew(:,1), Dnew(:,2), -dV(:,1), -dV(:,2))
        xlabel('PC1')
        ylabel('PC2')
        title('-Grad(V)')
        hold off
         
        subplot(2,2,3)
        semilogy(tracking(1:t,1),tracking(1:t,2))
        title('max( distance(Dew - Dold))')
        grid minor
        
        subplot(2,2,4)
        semilogy(tracking(1:t,1),tracking(1:t,4))
        title('max( abs(Vold - Vnew))')
        grid minor
    
        drawnow
    end
    
end
maxERR = max([convergenceV,ERR]);
maxERRdist = max([convergence,ERR]);

end

