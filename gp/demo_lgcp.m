% DEMO_LGCP    Demonstration for a log Gaussian Cox process
%              with inference via EP or Laplace approximation
%
%    Description 

%    Log Gaussian Cox process (LGCP) is a model for non-homogoneus
%    point-process in which the log intensity is modelled using
%    Gaussian process. LGCP can be modelled using log GP and
%    Poisson observation model in a discretized space. 
%
%    The model constructed is as follows:
%
%    The number of occurences of the realised point pattern within cell w_i
%
%         y_i ~ Poisson(y_i| |w_i|exp(f_i))
%
%    where |w_i| is area of cell w_i and f_i is the log intensity.
%
%    We place a zero mean Gaussian process prior for f =
%    [f_1, f_2,...,f_n] ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is
%    covariance function and th its parameters, hyperparameters. We
%    place a hyperprior for hyperparameters, p(th).
%
%    The inference is conducted via EP or Laplace, where we find Gaussian
%    approximation for p(f| th, data), where th is the maximum a
%    posterior (MAP) estimate for the hyper-parameters.
%
%    See also  LGCP, DEMO_SPATIAL2

% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% =====================================
% 1D-example
% =====================================
figure(1)
% Coal disaster data
S = which('demo_lgcp');
L = strrep(S,'demo_lgcp.m','demos/coal.txt');
x=load(L);
lgcp(x,'hyperint','CCD')
line([x x],[4.7 5],'color','k')
line(xlim,[4.85 4.85],'color','k')
xlim([1850 1963])
title('The coal mine disaster data, estimated intensity, and 90% interval')
xlabel('Year')
ylabel('Intensity')

% =====================================
% 2D-example
% =====================================
figure(2)
S = which('demo_lgcp');
L = strrep(S,'demo_lgcp.m','demos/redwoodfull.txt');
x=load(L);
lgcp(x,'range',[0 1 0 1],'latent_method','Laplace','gridn',20)
h=line(x(:,1),x(:,2),'marker','.','linestyle','none','color','k','markersize',10);
colorbar
axis square
set(gca,'xtick',[0:.2:1],'ytick',[0:.2:1])
title('Redwood data and intensity estimate')


ylim([0 5])
set(gcf,'units','centimeters');
set(gcf,'pos',[ 9.5742    9.5742    7.8962    5.9222])
set(gcf,'paperunits',get(gcf,'units'))
set(gcf,'paperpos',get(gcf,'pos'))

print -depsc2 /proj/bayes/jpvanhat/software/doc/GPstuffDoc/pics/demo_lgcp_fig2.eps
