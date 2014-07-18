%DEMO_MINIMAL  Demonstration with minimal commands needed for a quick GP modeling
%
%  Description
%    This demo shows how to make a quick data analysis using 2-3
%    GPstuff commands.
%
%    For Gaussian regression only gp_optim and gp_plot are needed.
%    For non-Gaussian data only gp_set, gp_optim and gp_plot are needed.
%
%    Default values for covariance functions, priors and optimisation
%    are used for a quick data analysis.
%
% Copyright (c) 2014 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%% 1D regression %%
% The El Niño Southern Oscillation (ENSO) climate data
S = which('demo_minimal');
L = strrep(S,'demo_minimal.m','demodata/enso.txt');
data=load(L);
x = data(:,2);
y = data(:,1);

figure
% plot data
plot(x,y,'r.')
hold on
% create a default GP with Gaussian likelihood and integrate over the parameters
gp=gp_ia([],x,y);
% plot predictions
gp_plot(gp,x,y);
hold off

%% 2D regression %%
% See also DEMO_REGRESSION1
S = which('demo_minimal');
L = strrep(S,'demo_minimal.m','demodata/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

figure
% create a default GP with Gaussian likelihood and integrate over the parameters
gp=gp_ia([],x,y);
% plot predictions
gp_plot(gp,x,y);
% plot data
hold on
plot3(x(:,1),x(:,2),y,'r*')

%% 2D classification %%
% See also DEMO_CLASSIFIC
S = which('demo_minimal');
L = strrep(S,'demo_minimal.m','demodata/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

figure
% create a default GP with a likelihood suitable for classification
gp=gp_set('lik',lik_logit());
% integrate over the latent values and parameters
gp=gp_ia(gp,x,y);
% plot conditional and joint prediction
gp_plot(gp,x,y);
view(2)
% plot data
hold on
plot3(x(y<0,1),x(y<0,2),ones(sum(y<0),1),'yx','MarkerSize',6,'LineWidth',2)
plot3(x(y>0,1),x(y>0,2),ones(sum(y>0),1),'go','MarkerSize',6)
hold off
