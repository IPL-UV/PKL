% PKL example 3
% Modeled temperatures and internal variability index

%% FKL applied to modeled temperatures afected by internal variability
% Input data = DJF mean temperature anomalies obtained by different models, each
% model with an ensemble of runs. Runs correspond to historical simulations and RCP
% simulations with a conterfactual climate configu  rations.  

% Output data = fraw (forced temperature response). This is the anomaly
% without the internal variability. It is obtained averaging multiple runs
% of each simulation configuration (of each ensemble) that is modified by extremely small
% amounts. 

%% >>> Misc
clc, clear, close all; 
% Set the seed 
rand('seed', 123)
randn('seed', 123)

%% >>> PLOT COOL GRAPHS
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

%% >>> Load data
X = readtable('./Data/LENS_tas_DJF_X_ct.csv', 'ReadVariableNames', false, 'HeaderLines', 1);
Y = readtable('./Data/LENS_tas_DJF_Y_ct.csv', 'ReadVariableNames', false, 'HeaderLines', 1);
M = readtable('./Data/LENS_tas_DJF_M_ct.csv', 'ReadVariableNames', false, 'HeaderLines', 1);

%% Get non empty variables
% get ancillary variables
nino34  = Y.(11); 

% Filter for cells with the string 'NA'
noNA_nino34   = ~strcmp('NA', nino34);
non_empty = noNA_nino34;

%% Get indexs of models for train, test and val. One for each
model_names = unique(M.Var4(non_empty));
n_models = length(model_names);
idx_model_train = 3; 
idx_model_val = 2;
idx_model_test = 1; 

name_model_train = model_names{idx_model_train};
name_model_val = model_names{idx_model_val};
name_model_test = model_names{idx_model_test};

idx_train = strcmp(name_model_train, M.Var4);
idx_val = strcmp(name_model_val, M.Var4);
idx_test =  strcmp(name_model_test, M.Var4);

%% Separate data
% Ancillary 
% Train 
train_nino34  = [cellfun(@str2num, nino34(idx_train & non_empty))];

% Test 
test_nino34  = [cellfun(@str2num, nino34(idx_test & non_empty))];

% Val 
val_nino34  = [cellfun(@str2num, nino34(idx_val & non_empty))];

% Target
y_train = Y(idx_train & non_empty, 3).(1);
y_val = Y(idx_val & non_empty, 3).(1);
y_test  = Y(idx_test & non_empty, 3).(1); 

% Variables
X_train = X(idx_train & non_empty, :);
X_val = X(idx_val & non_empty, :);
X_test = X(idx_test & non_empty, :);

% remove the text colum of X (first one)
X_train = X_train{:,2:end};
X_val = X_val{:,2:end};
X_test = X_test{:,2:end};

idx_year = 141;

M_train = M(idx_train & non_empty, 8);
years_train = str2num(cell2mat(M_train{1:idx_year,:}));
M_test = M(idx_test & non_empty, 8);
years_test = str2num(cell2mat(M_test{1:idx_year,:}));
M_val = M(idx_val & non_empty, 8);
years_val = str2num(cell2mat(M_val{1:idx_year,:}));

%% Visualize the variables
if false
    % ENSO index to train, validate and test the model.
    figure(1), clf, plot(train_nino34), hold on,
    plot(val_nino34), plot(test_nino34), hold off
    ylabel('ENSO index DJF')
    legend({name_model_train, name_model_val, name_model_test}, 'Location','best')
    
    % Data used to train, validate and test the model. Target
    figure(1), clf, plot(years_train, y_train(1:idx_year,1)), hold on,
    plot(years_val, y_val(1:idx_year,1)), plot(years_test, y_test(1:idx_year,1)), hold off
    ylabel('anomalies mean DJF TAS')
    legend({name_model_train, name_model_val, name_model_test}, 'Location','best')
    title('target')

    % Data used to train, validate and test the model. Variables: raw values
    figure(2), clf, plot(years_train, mean(X_train(1:idx_year,1),2)), hold on,
    plot(years_val, mean(X_val(1:idx_year,1),2)), plot(years_test, mean(X_test(1:idx_year,1),2)), hold off
    ylabel('anomalies mean DJF TAS global')
    legend({name_model_train, name_model_val, name_model_test}, 'Location','best')
    title('variables')

    % Target, variables 
    figure(3), clf, 
    plot(years_train, y_train(1:idx_year,1)), hold on,
    plot(years_train, mean(X_train(1:idx_year,1),2)), hold off
    ylabel('anomalies mean DJF TAS')
    legend({'wout internal variability', 'w internal variability'}, 'Location','best')
    title(name_model_train)

    % correlation
    residuals = y_train(1:idx_year,1) - mean(X_train(1:idx_year,1),2);
    figure(5), clf, 
    hold on,
    scatter(residuals, train_nino34(1:idx_year,1), 'filled'),
    ylabel('y-X')
    xlabel('S')
    R2 = corr(residuals, train_nino34(1:idx_year,1));
    legend({['R(S, y-X): ', num2str(R2)]}, 'Location','best')
    title(name_model_train)
    
    % train vs test ENSO index
    figure(6), plot(years_train, train_nino34(1:idx_year,1)), hold on,
    plot(years_train, test_nino34(1:idx_year,1))
    ylabel('mean DJF NINO34')
    xlim([min(years_train), max(years_train)]), legend({'train nino34','test nino34'}, 'Location','best')
end

%% Preprocess the data
X_train = zscore(X_train);
X_val = zscore(X_val);
X_test = zscore(X_test);
%
y_train = zscore(y_train);
y_val = zscore(y_val);
y_test = zscore(y_test);
%
train_nino34 = zscore(train_nino34);
test_nino34 = zscore(test_nino34);
val_nino34  = zscore(val_nino34);

% no test(for crossvalidation)
Xnt = 0;
Ynt = 0; 

%% Additional variables

ntr = length(y_train);
nva = length(y_val);
nte = length(y_test);

% Centering matrices
H_train = eye(ntr) - (1/ntr)*(ones(ntr)); 
H_test  = eye(nte) - (1/nte)*(ones(nte)); 
H_val   = eye(nva) - (1/nva)*(ones(nva));

all_S_train = [train_nino34];
all_S_test = [test_nino34];
all_S_val = [val_nino34];
all_S_names = {'nino34'};

%% MODEL
%% Range of the hyperparameters (problem specific)
l_lambdas = 25; 
lambdas = logspace(0.1,0.5,l_lambdas);
l_sigmas = 25;
sigmas  = logspace(-2,2,l_sigmas);
l_mus= 10;
mus = +[0 logspace(-2, 1.25, l_mus)];
lmus = length(mus);  
z = find(mus == 0);

rmse = zeros(lmus, 1);
nhsic = zeros(lmus, 1); 

S_train = all_S_train(:, 1);
S_test = all_S_test(:, 1);
S_val = all_S_val(:, 1);
S_name = all_S_names{1}        
     
%% Apply the model
cross_val = false;
optimize_sq = false; % Set true to optimize sigma for S (only if length(val) = length(train) 
[res] = PKL(Xnt,Ynt,X_train,y_train,S_train,X_val,y_val,S_val,X_test,y_test,S_test,mus,lambdas,sigmas,cross_val,optimize_sq);

[~,idx_min_acc] = min(res.acc_wd);
[~,idx_min_hsic] = min(res.dep_wd);
[~,idx_max_acc] = max(res.acc_wd);
[~,idx_max_hsic] = max(res.dep_wd);

rmse(:, 1) = res.acc_wd;
nhsic(:, 1) = res.dep_wd; 

idx_mins_idx_acc(:, 1) =  idx_min_acc;
idx_mins_idx_hsic(:, 1) = idx_min_hsic;

wc(:, 1) = res.wc;
wd(:, 1) = res.wd;

%% Figures
QteQte = S_test*S_test';
HQteQteH = H_test*QteQte*H_test;
K = rbf(X_test,X_train, res.sigma);

%% Scatter HSIC vs RMSE
figure(17), 
plot(rmse(:,1), nhsic(:,1), 'color', co(4,:)), hold on
ylabel('HSIC'),xlabel('RMSE $[K]$'), 
grid on, 
plot(rmse(1,1), nhsic(1,1), 'square', 'color', 'k','HandleVisibility','off','MarkerSize',15)
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])

%% KRR
figure(110), clf,
R1 = corr(K*wc(:,1), S_test);
scatter(S_test, K*wc(:,1), 3), hold on
pred_standard = K*wc(:,1);
lmodel = fitlm(S_test, K*wc(:,1));
coeff = lmodel.Coefficients.Estimate;
line_standard = coeff(1,1) + coeff(2,1)*S_test;
plot(S_test, line_standard)
%
HSIC_krr = (1/nte^2)*trace(HQteQteH*(K*wc(:,1))*(K*wc(:,1))');
%
xlabel('$s \ [K]$'), ylabel('$\hat{y} \ [K]$'), 
legend(['$\rho (\hat{y}, s)$ = ', num2str(round(R1,2)), newline, '$HSIC{(\hat{y}, s)}$ = ', num2str(round(HSIC_krr,5))], 'location', 'best')
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])

%% PKL
figure(210), clf,
R2 = corr(K*wd{idx_mins_idx_hsic(:,1), 1}, S_test);
HSIC = nhsic(idx_mins_idx_hsic(:,1),1);
scatter(S_test, (K*wd{idx_mins_idx_hsic(:,1), 1}), 3), hold on
pred_pkl = (K*wd{idx_mins_idx_hsic(:,1), 1});
lmodel = fitlm(S_test, (K*wd{idx_mins_idx_hsic(:,1), 1}));
coeff = lmodel.Coefficients.Estimate;
line_pkl = coeff(1,1) + coeff(2,1)*S_test;
plot(S_test, line_pkl)
%
HSIC_pkl = (1/nte^2)*trace(HQteQteH*(K*wd{idx_mins_idx_hsic(:,1), 1})*(K*wd{idx_mins_idx_hsic(:,1), 1})');
% 
xlabel('$s \ [K]$'),ylabel('$\hat{y} \ [K]$'),
legend(['$\rho (\hat{y}, s)$ = ', num2str(round(R2,2)), newline, '$HSIC{(\hat{y}, s)}$ = ', num2str(round(HSIC_pkl,4))], 'location', 'best')
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])

%% To make the previous figures pretty:
% Uncoment/comment one of the below 
%% KRR
% % fig . a
XX = [S_test, pred_standard];
xmin = min(S_test); xmax = max(S_test);
ymin = min(pred_standard); ymax = max(pred_standard);
sigma = 0.1;
line = line_standard;
RR = R1;
plotHSIC = HSIC_krr;

%% PKL
% fig . b
XX = [S_test, pred_pkl];
xmin = min(S_test); xmax = max(S_test);
ymin = min(pred_pkl); ymax = max(pred_pkl);
sigma = 0.04;
line = line_pkl;
RR = R2;
plotHSIC = HSIC_pkl;

%%
% Auxiliar variables
Ng = 18; % bins per dim
nn = linspace(xmin,xmax,Ng);
rr = linspace(ymin,ymax,Ng);
h1=[]; h2=[]; h3 = [];

% Scatter
figure(2),
h1 = plot(XX(:,1),XX(:,2),'.','Markersize',10,'color',[0 0 0]);
ylim([-0.2, 0.18])
xlim([-3, 2.6])
hold on

% Line
plot(S_test, line)
% Density estimation
[x1,x2] = ndgrid(linspace(min(nn),max(nn),Ng),linspace(min(rr),max(rr),Ng));
x1 = x1(:,:)'; x2 = x2(:,:)'; xi = [x1(:) x2(:)];
PDF  = mvksdensity(XX,xi,'Bandwidth',[sigma,sigma],'Kernel','normpdf');
zz = reshape(PDF,Ng,Ng);
zz = imgaussfilt(zz, 2.5);
[~,h3] = contour(nn,rr,zz,3);
h3.LineWidth = 3;
xlabel('$s \ [K]$'), ylabel('$\hat{y} \ [K]$'), 
legend(['$\rho (\hat{y}, s)$ = ', num2str(round(RR,2)), newline, '$HSIC{(\hat{y}, s)}$ = ', num2str(round(plotHSIC,4))], 'location', 'southwest')
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 18 13]); %x_width=18cm y_width=13cm
set(gcf, 'units','centimeters', 'position',[0 0 18 13])
hold off