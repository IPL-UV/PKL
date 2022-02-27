% PKL example 2
% Hyperspectral and vegetation in situ data

%% Consistency with ancillary in situ data

clear; clc; close all;
rand('seed',42)
randn('seed',42)

%% Set default graphics  
% general graphics, this will apply to any figure you open
set(0, ...
    'DefaultFigureColor', 'w', ...
    'DefaultAxesLineWidth', 1.5, ...
    'DefaultAxesXColor', 'k', ...
    'DefaultAxesYColor', 'k', ...
    'DefaultAxesFontUnits', 'points', ...
    'DefaultAxesFontSize', 18, ...
    'DefaultAxesFontName', 'Helvetica', ...
    'DefaultLineLineWidth', 4, ...
    'DefaultTextFontUnits', 'Points', ...
    'DefaultTextFontSize', 18, ...
    'DefaultTextFontName', 'Helvetica', ...
    'DefaultAxesBox', 'off', ...
    'DefaultAxesTickLength', [0.015 0.025],...
    'Defaulttextinterpreter','latex',...  
    'DefaultAxesTickLabelInterpreter','latex',...
    'DefaultLegendInterpreter','latex');

% set the tickdirs to go out - need this specific order
set(0, 'DefaultAxesTickDir', 'out');
set(0, 'DefaultAxesTickDirMode', 'manual');

%% Dataset: SPARC0304
% TYPE: real data

% X (covariables): radiances, dimensions -> 62 features (channels) x 135 samples
% Y (target): LAI *or* Chla, dimensions -> 1 feature x 135 samples 
% S (anciliary variable): fCOVER *or* LAI, dimensions -> 1 feature x 135 samples 
load SPARC0304.mat 

%% Select the target and ancillary variable
% exp_1: LAI (y) and fCOVER (s)  
% exp_2: Chla (y) and LAI (s)

expHV = 'exp_1';
%expHV = 'exp_2';

if strcmp(expHV,'exp_1')
    fprintf('exp_1: LAI (y) and fCOVER (s)')
    % Assign, standarize variables
    X = zscore(Spectra');
    Y = zscore(Chla'); 
    S = zscore(LAI');
    N = length(Y);
    % Increase the capacity of the model -> Widen the hypothesis space
    X = [X,ones(N,1)];
    
    %figure(1), plot((Y)), hold on, plot((S)),hold off, legend('Chla','LAI')
    
elseif strcmp(expHV,'exp_2')
    fprintf('exp_2: Chla (y) and LAI (s)')
    X = zscore(Spectra');
    Y = zscore(LAI');
    S = zscore(fcover');
    N = length(Y);
    % Increase the capacity of the model -> Widen the hypothesis space
    X = [X,ones(N,1)];
 
    %figure(1), plot(zscore(Y)), hold on, plot(zscore(S)), hold off, legend('LAI','fcover')

else
    print('No experiment selected')
    return
end

%% Split data: train-test-val
rp = randperm(N);
ntr = round(0.4*N); 
nva = round(0.2*N);
% train
Xtr = X(rp(1:ntr),:);
Ytr = Y(rp(1:ntr),:);
Str = S(rp(1:ntr),:);

% val
Xva = X(rp(ntr+1:nva+ntr),:);
Yva = Y(rp(ntr+1:nva+ntr),:);
Sva = S(rp(ntr+1:nva+ntr),:); 

%test 
Xte = X(rp(nva+ntr+1:end),:);
Yte = Y(rp(nva+ntr+1:end),:);
Ste = S(rp(nva+ntr+1:end),:); 

% no test(for crossvalidation)
Xnt = X(rp(1:nva+ntr),:);
Ynt = Y(rp(1:nva+ntr),:);

%% MODEL
%% Range of the hyperparameters (problem specific)
l_lambdas = 25; 
lambdas = logspace(0.1,0.5,l_lambdas);
l_sigmas = 25;
sigmas  = logspace(-2,2,l_sigmas);
l_mus = 250;
mus = [-logspace(-0.55,-5,l_mus) 0 logspace(-5,-0.55,l_mus)];
l_mus = length(mus);  

z = find(mus == 0);

%% Apply the model
cross_val = true;
optimize_sq = false; % Set true to optimize sigma for S (only if length(val) = length(train) 

[res] = PKL(Xnt,Ynt,Xtr,Ytr,Str,Xva,Yva,Sva,Xte,Yte,Ste,mus,lambdas,sigmas,cross_val,optimize_sq);

%% Inference
% if not using the cross val version
K = rbf(X,Xtr,res.sigma); % projection over all X

rmse = zeros(1,l_mus);
for i=1:length(res.wd)
    yhat = K*res.wd{i};
    ass = assessment(S,yhat,'regress');
    rmse(i) = sqrt(mean((yhat-Y).^2));
end
[~,miny] = min(rmse);

%% Figures
% Assistant figure to predefine the colormmap 
marker_size = 75;
figure(123), scatter(res.mus,res.mus,marker_size,res.mus,'filled'), title('Assistant figure')
co = colormap(orangewhitered(length(res.mus))); colorbar

% Exp. figure
figure(2),clf,
scatter(res.acc_wd,res.dep_wd,75,res.mus,'filled')
ylabel('HSIC'),xlabel('RMSE $[mg/m^3]$'), colormap(co), colorbar
grid on, hold on
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])
%plot(res.acc_wd(z),res.dep_wd(z),'+', 'color', 'k','HandleVisibility','off')

if strcmp(expHV,'exp_1')
    print(figure(2),'-depsc','LAI(y)_fCOVER(s).eps')
else 
    print(figure(2),'-depsc','Chla(y)_LAI(s).eps')
end

% Prediction
%figure(3)
%plot(Y,'r'), hold on, plot(K*res.wd{miny},'b'), plot(K*res.wd{z},'g'), hold off, legend('target', 'PKL','KRR')