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
% R = log10(max([R443'; R490'; R510'])'./R555);
R = log10(max([R443, R490, R510]')'./R555);
YTEhat = 10.^(0.4708 - 3.8469*R + 4.5338*R.^2 -2.4434*R.^3);  %  -0.0414
YTEhat6 = log10(YTEhat);
ResOC4 = assessment(Y,YTEhat6,'regress');

%% Increase the capacity of the model -> Widen the hypothesis space
% Hypothesis space: set of functions the model can choose from
% Capacity: ability to fit a wide variety of functions
% Hypothesis -> is the object; Capacity -> is the property of the object 

% We consider the affine functions, linear functions with an intercept term (usually called bias).
% By adding new terms of the dependent variables while retaining the relationship between
% w and the independent variables, Y. 
X = [X,ones(N,1)];

% Comment: As we use Kernel methods, we are dealing in the end with
% nonparametric models (the size of the parameter vector is not finite and
% fixed). Nonparametric models are the most extreme case of arbitrarily high
% capacity. 

%% Define train-test-val data
% The training and test data are generated by a probability distribution 
% over the dataset (a process known as data-generating process) 
% Typically we make a set of assumptions, the i.i.d assumptions 
% -> The training and test set are INDEPENDENT of each other
% -> The training and test set are IDENTICALLY DISTRIBUTED

% This assumptions enable us to describe the data-generating process with a
% probability distribution, the data-generating distribution. And this 
% distribution is later used to generate the training and test data. 

% Here we sample using a distribution that arises from this assumptions,
% the uniform distribution is used to separate the data.
% As the function: Yreal(X) remains the same for all the samples, by using a 
% uniform distribution we obtain train and test data identically distributed. 
rp = randperm(N); 

% train data: data to obtain the parameters
% validation data: data to train hyperparameters that control model capacity
% test data: data used to obtain te performance measure, not seen by the model

% Comment: we can't use the training set to obtain the best hyperparameters
% as they will always choose the maximum possible model capacity, resulting in overfitting. We wouldn't be regularizing
% We can't also use all Y for the test set, as the error would account for the train data which is 
% better modeled.  

ntr = round(.15*N); %<0.5 for better results, otherwise we are giving the model too much information
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
figure(2), clf, plot(Y), hold on 

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
    figure(11), plot(S), hold on, legend('Morel1','CalCOFI','OC2','OC4')
    
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
    % if not using the cross val version
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
    % Display lambda and sigma to make sure that the model is the same for
    % mu = 0 for all theoric parametrizations
    disp([num2str(ii),' Sigma = ',num2str(res.sigma), ', Lambda = ',num2str(res.lambda), ', \mu inflexió = ', num2str(best_mu)])
    %
        %%
        figure(1),
        plot(res.acc_wd,res.dep_wd,'color', co(ii,:))
        hold on  
        plot(res.acc_wd(mus==0),res.dep_wd(mus==0),'square', 'color', 'k','HandleVisibility','off','MarkerSize',15)
        
        % Adding color gratiation for each plot
%         z = zeros(size(res.acc_wd));
%         capa1 = [linspace(1,0.1,length(res.dep_wd));linspace(1,0.1,length(res.dep_wd))]*co(ii,1);
%         capa2 = [linspace(1,0.1,length(res.dep_wd));linspace(1,0.1,length(res.dep_wd))]*co(ii,2);
%         capa3 = [linspace(1,0.1,length(res.dep_wd));linspace(1,0.1,length(res.dep_wd))]*co(ii,3);
%         lineColor = cat(3,capa1,capa2); %repmat(co(ii,:),length(res.acc_wd),1));  % This is the color, it has the size of "res.acc_wd" in this case.
%         lineColor = cat(3,lineColor,capa3);
%         surface([res.acc_wd;res.acc_wd], [res.dep_wd;res.dep_wd], [z;z], lineColor,...
%             'FaceColor', 'no',...       % Don't bother filling faces with color
%             'EdgeColor', 'interp',... % Use interpolated color for edges
%             'LineWidth', 4);
%         
        %%
        figure(2), plot(K*res.wd{best_mu})
        %
        fixX = 0.259; % valor de RMSE per al tall
        fixY = 1.1; % valor de HSIC per al tall
        [~,indexY] = min(abs(res.acc_wd-fixX));
        [~,indexX] = min(abs(res.dep_wd-fixY));
        fixant_rmse(ii,1) = res.acc_wd(indexX);
        fixant_hsic(ii,1) = res.dep_wd(indexY);
        disp([num2str(ii), ' pendent = ',  num2str((res.acc_wd(indexY) - res.acc_wd(1))/(res.dep_wd(indexY) - res.dep_wd(1)))])

        % disp(['\mu fix X = ', num2str((indexX)), ' \mu fix Y = ',  num2str(indexY)])
        % disp(['mu inflexió = ', num2str(mus(best_mu))])
end
%% Figures
% 1) 
figure(1), grid on
% Per a posar les linies
    AX = gca;
    %plot(fixX*ones(2,1),AX.YLim,'-.','Color',[0.25 0.25 0.25],'linewidth',2,'HandleVisibility', 'off')
    %plot(AX.XLim,fixY*ones(2,1),'-.','Color',[0.25 0.25 0.25],'linewidth',2,'HandleVisibility', 'off')
ylabel('HSIC'),xlabel('RMSE $[mg/m^3]$')
legend('Morel1','CalCOFI','OC2','OC4','location','northeast')
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])

% 2)
figure(2)
legend('Morel1','CalCOFI','OC2','OC4','location','southeast'), ylabel('$Y-\hat{Y}$')

% 3)
figure(3), clf
yyaxis left
    hb = bar(1:4,[fixant_hsic nan(4,1)]);
    ylabel('NHSIC')
    ylim([1 1.04])
    hb(1).FaceColor = co(3,:);
    set(gca, 'YColor', 'k')
    ax1 = gca;
    l1=line('parent',ax1,'xdata',1:4,'ydata',NaN(1,4),'color',co(3,:));
yyaxis right
    hb = bar(1:4,[nan(4,1) fixant_rmse]);
    ylabel('RMSE')
    hb(2).FaceColor = co(4,:);
    ylim([0.256 0.259])
    yticks([0.256 0.257 0.258 0.259])
    set(gca, 'YColor', 'k')
    ax2 = gca;
    l2=line('parent',ax2,'xdata',1:4,'ydata',NaN(1,4),'color',co(4,:));
legend( [l1;l2] , {'NHSIC','RMSE'} );
xticklabels({'Morel1','CalCOFI','OC2','OC4'})
xlabel('Models')
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])