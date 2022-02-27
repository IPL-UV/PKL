% PKL example 1
% Hyperspectral and vegetation in situ data

%% Consistency with models for biophysical parameter estimation.

clear; clc; close all;
rand('seed',123)
randn('seed',123)

%% PLOT COOL GRAPHS
% general graphics, this will apply to any figure you open
% (groot is the default figure object).
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

co      = [185, 202, 195;
              67, 74, 66;
              255, 149, 10;
              84, 20, 144]/256;
          
%% Dataset: SeaBAM
% TYPE: real data
% X : logarithm in base 10 of radiances (design matrix)
% y : logartihm in base 10 of ocean chlorophYll content
% y=f(t), X = g(t) both aproximately sigmoidal in log-log scale
% s : predictions should be similar to other models

load SeaBAM.mat

% Preprocess the data, stardandize 
Y = zscore(Y);
X(:,1) = zscore(X(:,1));
X(:,2) = zscore(X(:,2));
X(:,3) = zscore(X(:,3));
X(:,4) = zscore(X(:,4));
X(:,5) = zscore(X(:,5));

%% Define the theorical models 

%  Obtain radiances from converted covariables
R412 = 10.^X(:,1);
R443 = 10.^X(:,2);
R490 = 10.^X(:,3);
R510 = 10.^X(:,4);
R555 = 10.^X(:,5);

N = length(Y);

% a simple linear regression
w = pinv(X)*Y; Ylr = X*w;
ResLR = assessment(Y,Ylr,'regress');

% s: generate model outputs to which we should be *dependent* to
% Comments: some models have an additional constant value (they appear commented later). This should be added
% and not doing so is an added bias. We do not add them because our
% independent variable is the log10 of the chl concentration. Parameter
% model outputs are converted to this scale and if we do not remove this
% value, for some examples, we enconter the log10 of a negative
% argument. Which is undefined. 

% Morel-1 XXX
R = log10(R443./R555);
YTEhat1 = 0.2492-1.768*R;
ResMorel1 = assessment(Y,YTEhat1,'regress');

% Morel-3
R = log10(R443./R555);
YTEhat2 = 0.20766 -1.82878*R +0.75885*R.^2 -0.73979*R.^3;
ResMorel3 = assessment(Y,YTEhat2,'regress');

% CalCOFI 2-band cubic
R = log10(R490./R555);
YTEhat3 = 0.450 -2.860*R+ 0.996*R.^2 -0.3674*R.^3;
ResCalcofi2c = assessment(Y,YTEhat3,'regress');

% CalCOFI 2-band linear XXX
R = log10(R443./R555);
YTEhat4 = 0.444-2.431*R;
ResCalcofi2l = assessment(Y,YTEhat4,'regress');

% OC2 XXX
R = log10(R490./R555);
YTEhat = 10.^(0.341 - 3.001*R + 2.811*R.^2 -2.041*R.^3);    %-0.040
YTEhat5 = log10(YTEhat);
ResOC2 = assessment(Y,YTEhat5,'regress');

% OC4 XXX
R = log10(max([R443, R490, R510]')'./R555);
YTEhat = 10.^(0.4708 - 3.8469*R + 4.5338*R.^2 -2.4434*R.^3);  %  -0.0414
YTEhat6 = log10(YTEhat);
ResOC4 = assessment(Y,YTEhat6,'regress');

%% Increase the capacity of the model -> Widen the hypothesis space
X = [X,ones(N,1)];

%% Define train-test-val data
rp = randperm(N);  

ntr = round(.15*N); 
nva = round((N-ntr)/2);
% train
Xtr = X(rp(1:ntr),:);
Ytr = Y(rp(1:ntr),:);

% val
Xva = X(rp(ntr+1:nva+ntr),:);
Yva = Y(rp(ntr+1:nva+ntr),:); 

%test 
Xte = X(rp(nva+ntr+1:end),:);
Yte = Y(rp(nva+ntr+1:end),:);

% no test(for crossvalidation)
Xnt = X(rp(1:nva+ntr),:);
Ynt = Y(rp(1:nva+ntr),:);

%% MODEL
% Define the figures for all
figure(1),clf
%figure(2), clf, plot(Y), hold on 

% Execute the model for each parametric model of the chl 
for ii=1:4
    if ii==1
        YYY = YTEhat1;
    elseif ii==2
        YYY = YTEhat3;
    elseif ii==3
        YYY = YTEhat5;
    elseif ii==4
        YYY = YTEhat6;
    end
    %
    % Normalize S to compare with other Ss 
    S = zscore(YYY);
    Str = S(rp(1:ntr),:);
    Sva = S(rp(ntr+1:nva+ntr),:);
    Ste = S(rp(nva+ntr+1:end),:); 
    %figure(11), plot(S), hold on, legend('Morel1','CalCOFI','OC2','OC4')
    
    %% Range of the hyperparameters (problem specific)
    l_lambdas = 20; 
    lambdas = logspace(-3,3,l_lambdas);
    l_sigmas = 10;
    sigmas  = logspace(-3,3,l_sigmas);
    l_mus = 45;
    esp = logspace(-3,-1.,l_mus); 
    mus = [0, -esp]; 
    lmus = length(mus);

    z = find(mus == 0);

    %% Apply the model
    cross_val = false;
    optimize_sq = false; % Set true to optimize sigma for S (only if length(val) = length(train) 

    [res] = PKL(Xnt,Ynt,Xtr,Ytr,Str,Xva,Yva,Sva,Xte,Yte,Ste,mus,lambdas,sigmas,cross_val,optimize_sq);
    
    %% Inference
    K = rbf(X,Xtr,res.sigma); % projection over all X

    rmse = zeros(1,l_mus);
    for i=1:length(res.wd)
        yhat = K*res.wd{i};
        ass = assessment(S,yhat,'regress');
        rmse(i) = sqrt(mean((yhat-Y).^2));
    end
    [~,miny] = min(rmse);
    %
    best_mu = find(min(res.acc_wd) == res.acc_wd);
    %
        %%
        figure(1),
        plot(res.acc_wd,res.dep_wd,'color', co(ii,:))
        hold on  
        plot(res.acc_wd(mus==0),res.dep_wd(mus==0),'square', 'color', 'k','HandleVisibility','off','MarkerSize',15)
                 
end
%% Figures
% 1) 
figure(1), grid on
ylabel('HSIC'),xlabel('RMSE $[mg/m^3]$')
legend('Morel1','CalCOFI','OC2','OC4','location','northeast')
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])
