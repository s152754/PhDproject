clear all; close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Title: Uncertainty quantification                                   %%%
%%% Author: Aricia Rinkens                                              %%%
%%% Description: UQ to compare between model(s) and to experiments.     %%%   
%%% Output:                                                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nexp = 10;
%% Input files
%%%%%%%%%%%% PVP cases %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% experiment
ExpF1V1         = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_PVPF1V1.csv");
TimeF1V1     = ExpF1V1.data(1:end-1,1:nexp);
RoutF1V1     = ExpF1V1.data(1:end-1,nexp+1:2*nexp);
muRF1V1     = ExpF1V1.data(1:end-1,end-1);
stdRF1V1    = ExpF1V1.data(1:end-1,end);

% experiment
ExpF2V1         = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_PVPF2V1.csv");
TimeF2V1     = ExpF2V1.data(1:end-1,1:nexp);
RoutF2V1     = ExpF2V1.data(1:end-1,nexp+1:2*nexp);
muRF2V1     = ExpF2V1.data(1:end-1,end-1);
stdRF2V1    = ExpF2V1.data(1:end-1,end);

% experiment
ExpF1V2         = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_PVPF1V2.csv");
TimeF1V2     = ExpF1V2.data(1:end-1,1:nexp);
RoutF1V2     = ExpF1V2.data(1:end-1,nexp+1:2*nexp);
muRF1V2     = ExpF1V2.data(1:end-1,end-1);
stdRF1V2    = ExpF1V2.data(1:end-1,end);

% experiment
ExpF2V2        = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_PVPF2V2.csv");
TimeF2V2     = ExpF2V2.data(1:end-1,1:nexp);
RoutF2V2     = ExpF2V2.data(1:end-1,nexp+1:2*nexp);
muRF2V2     = ExpF2V2.data(1:end-1,end-1);
stdRF2V2    = ExpF2V2.data(1:end-1,end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Glycerol F V1
ExpGlyV1F    = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_GlyV1F.csv");
TimeGlyV1F   = ExpGlyV1F.data(1:end-1,1:nexp);
RoutGlyV1F   = ExpGlyV1F.data(1:end-1,nexp+1:2*nexp);
muRGlyV1F    = ExpGlyV1F.data(1:end-1,end-1);
stdRGlyV1F   = ExpGlyV1F.data(1:end-1,end);

% Glycerol F025 V1
ExpGlyV1F025    = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_GlyV1F025.csv");
TimeGlyV1F025   = ExpGlyV1F025.data(1:end-1,1:nexp);
RoutGlyV1F025   = ExpGlyV1F025.data(1:end-1,nexp+1:2*nexp);
muRGlyV1F025    = ExpGlyV1F025.data(1:end-1,end-1);
stdRGlyV1F025   = ExpGlyV1F025.data(1:end-1,end);

% Glycerol F05 V1
ExpGlyV1F05    = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_GlyV1F05.csv");
TimeGlyV1F05   = ExpGlyV1F05.data(1:end-1,1:nexp);
RoutGlyV1F05   = ExpGlyV1F05.data(1:end-1,nexp+1:2*nexp);
muRGlyV1F05    = ExpGlyV1F05.data(1:end-1,end-1);
stdRGlyV1F05   = ExpGlyV1F05.data(1:end-1,end);

% Glycerol F025 V2
ExpGlyV2F025    = importdata("C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\RoutExp_GlyV2F025.csv");
TimeGlyV2F025   = ExpGlyV2F025.data(1:end-1,1:nexp);
RoutGlyV2F025   = ExpGlyV2F025.data(1:end-1,nexp+1:2*nexp);
muRGlyV2F025    = ExpGlyV2F025.data(1:end-1,end-1);
stdRGlyV2F025   = ExpGlyV2F025.data(1:end-1,end);
%% Model comparison

%%%%%%%%%% Glycerol  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
% title('Glycerol, 10 experiments per case')
plot(TimeGlyV1F,RoutGlyV1F,'b-o','LineWidth',1)
hold on
plot(TimeGlyV1F025,RoutGlyV1F025,'k-o','LineWidth',1)
plot(TimeGlyV1F05,RoutGlyV1F05,'r-o','LineWidth',1)
plot(TimeGlyV2F025,RoutGlyV2F025,'g-o','LineWidth',1)
xlabel('$t$[s]','Interpreter','latex')
ylabel('$R$[m]','Interpreter','latex')
ax = gca; 
ax.FontSize = 18;
% legend({"F = 2.8 N, V = big", "F = 5.69 N, V = small", "F = 8.64 N, V = small","F = 5.69 N, V = big"},'Interpreter','latex','Location','Southeast')

figure()
errorbar(TimeGlyV1F(:,1),muRGlyV1F,2*stdRGlyV1F,'b','LineWidth',1.5)
hold on
errorbar(TimeGlyV1F025(:,1),muRGlyV1F025,2*stdRGlyV1F025,'k','LineWidth',1.5)
errorbar(TimeGlyV1F05(:,1),muRGlyV1F05,2*stdRGlyV1F05,'r','LineWidth',1.5)
errorbar(TimeGlyV2F025(:,1),muRGlyV2F025,2*stdRGlyV2F025,'g','LineWidth',1.5)
xlabel('$t$[s]','Interpreter','latex')
ylabel('$R$[m]','Interpreter','latex')
ax = gca; 
ax.FontSize = 18;
legend({"F = 2.8 N, V = big", "F = 5.69 N, V = small", "F = 8.64 N, V = small","F = 5.69 N, V = big"},'Interpreter','latex','Location','Southeast')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% PVP cases %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure()
% title('Glycerol, Case I, 10 experiments')
plot(TimeF1V1,RoutF1V1,'b-o','LineWidth',1)
hold on
plot(TimeF2V1,RoutF2V1,'k-o','LineWidth',1)
plot(TimeF1V2,RoutF1V2,'r-o','LineWidth',1)
plot(TimeF2V2,RoutF2V2,'g-o','LineWidth',1)
xlabel('$t$[s]','Interpreter','latex')
ylabel('$R$[m]','Interpreter','latex')
ax = gca; 
ax.FontSize = 18;
% legend({"F = 2.8 N, V = big", "F = 8.64 N, V = big", "F = 2.8 N, V = small","F = 8.64 N, V = small"},'Interpreter','latex','Location','Southeast')

figure()
errorbar(TimeF1V1(:,1),muRF1V1,2*stdRF1V1,'b','LineWidth',1.5)
hold on
errorbar(TimeF2V1(:,1),muRF2V1,2*stdRF2V1,'k','LineWidth',1.5)
errorbar(TimeF1V2(:,1),muRF1V2,2*stdRF1V2,'r','LineWidth',1.5)
errorbar(TimeF2V2(:,1),muRF2V2,2*stdRF2V2,'g','LineWidth',1.5)
xlabel('$t$[s]','Interpreter','latex')
ylabel('$R$[m]','Interpreter','latex')
ax = gca; 
ax.FontSize = 18;
legend({"F = 2.8 N, V = big", "F = 8.64 N, V = big", "F = 2.8 N, V = small","F = 8.64 N, V = small"},'Interpreter','latex','Location','Southeast')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

