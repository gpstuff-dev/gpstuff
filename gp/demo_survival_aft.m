%DEMO_SURVIVAL_AFT  Survival model using accelerated failure time (AFT) models
%
%  Description 
%    Survival model using accelerated failure time (AFT) models
%    including Weibull, log-logistic and log-normal. Hazard rate for
%    AFT models is
%   
%       h(t) = h_0(t)*exp(f),
%
%    where the baseline hazard is assumed to have paramteric form such as 
%    Weibull distribution
%
%       h_0(t) = r*t^(r-1), r>0
%
%    where r is the shape parameter. A zero-mean Gaussian process prior is
%    assumed for for f = [f_1, f_2,...,f_n] ~ N(0, K), where K is the
%    covariance matrix, whose elements are given as K_ij = k(x_i, x_j |
%    th). The function k(x_i, x_j | th) is covariance function and th its
%    parameters.
%
%    The latent inference is made using Laplace approximation and
%    covariance function and likelihood parameter inference is made
%    using CCD integration.
%    
%    Example data set is leukemia survival data in Northwest England
%    presented in (Henderson, R., Shimakura, S., and Gorst, D. (2002).
%    Modeling spatial variation in leukemia survival data. Journal of the
%    American Statistical Association, 97:965–972). Data set was downloaded
%    from http://www.math.ntnu.no/%7Ehrue/r-inla.org/examples/leukemia/leuk.dat
%
%  See also  DEMO_SURVIVAL_COXPH
%
% Copyright (c) 2011 Jaakko Riihimäki
% Copyright (c) 2013 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% First load data

S = which('demo_survival_aft');
L = strrep(S,'demo_survival_aft.m','demodata/leukemia.txt');
leukemiadata=load(L);

% leukemiadata consists of:
% 'time', 'cens', 'xcoord', 'ycoord', 'age', 'sex', 'wbc', 'tpi', 'district'

% survival times
y0=leukemiadata(:,1);
y=y0;
% scale survival times (so that constant term for the latent function can be small)
y=y0/geomean(y0);

ye=1-leukemiadata(:,2); % event indicator, ye = 0 for uncensored event
                        %                  ye = 1 for right censored event

% for simplicity choose 'age', 'sex', 'wbc', and 'tpi' covariates
x0=leukemiadata(:,5:8);
x=x0;
[n, m]=size(x);
% transform white blood cell count (wbc), which has highly skewed
% distribution with zeros for measurements below measurement accuracy
x(:,3)=log10(x(:,3)+0.3);
% normalize continuous covariates 
[x(:,[1 3:4]),xmean(:,[1 3:4]),xstd(:,[1 3:4])]=normdata(x(:,[1 3:4]));
% binary sex covariate is not transformed
xmean(2)=0;xstd(2)=1;

% Create the covariance functions
cfc = gpcf_constant('constSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
cfl = gpcf_linear('coeffSigma2',1,'coeffSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
cfs = gpcf_sexp('lengthScale', ones(1,m),'lengthScale_prior',prior_t('s2',1^2,'nu',1),'magnSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));

% Create the likelihood structure
% log-logistic works best for this data
lik = lik_loglogistic();
%lik = lik_loggaussian();
%lik = lik_weibull();

% Create the GP structure
gp = gp_set('lik', lik, 'cf', {cfc cfl cfs}, 'jitterSigma2', 1e-8);
% classically used linear model could be defined like this
%gp = gp_set('lik', lik, 'cf', {cfc cfl}, 'jitterSigma2', 1e-8);

% Set the approximate inference method to Laplace
% Laplace is default, so this line could be skipped
gp = gp_set(gp, 'latent_method', 'Laplace');
%gp = gp_set(gp, 'latent_method', 'EP');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
% Optimize with the BFGS quasi-Newton method
gp=gp_optim(gp,x,y,'z',ye,'opt',opt,'optimf',@fminlbfgs);
% CCD integration over parameters
gpia=gp_ia(gp,x,y,'z',ye,'opt_optim',opt,'optimf',[]);
% Leave-one-out cross-validation
[~,~,lploo]=gp_loopred(gpia,x,y,'z',ye);sum(lploo)
% -1629 for linear + squared exponential
% -1662 for linear only

figure
set(gcf,'units','centimeters');
set(gcf,'pos',[29 6 24 6])
subplot('position',[0.07 0.21 0.20 0.77]);
i1=2;i2=1;
[pmu,~,xtc]=gp_cpred(gpia, x, y, x, [i1 i2], 'z', ye, 'target','mu');
xtc{1}=denormdata(xtc{1},xmean(i2),xstd(i2));
xtc{2}=denormdata(xtc{2},xmean(i2),xstd(i2));
h1=semilogy(xtc{1},pmu{1}*geomean(y0),'k--');
set(h1(2),'LineWidth',2)
hold on
h2=semilogy(xtc{2},pmu{2}*geomean(y0),'k-');
set(h2(2),'LineWidth',2)
hold off
set(gca,'ytick',[10 30 100 300 1000 3000],'yticklabel',{'10' '30' '100' '300' '1000' '3000'},'ylim',[20 2000])
xlabel('Age (years)')
ylabel('Expected lifetime (days)')
[hl,hlo]=legend([h1(2), h2(2)],'Female','Male');set(hl,'box','off')

subplot('position',[0.31 0.21 0.20 0.77]);
i1=2;i2=3;
[pmu,~,xtc]=gp_cpred(gpia, x, y, x, [i1 i2], 'z', ye, 'target','mu');
xtc{1}=denormdata(xtc{1},xmean(i2),xstd(i2));
xtc{2}=denormdata(xtc{2},xmean(i2),xstd(i2));
h1=semilogy(xtc{1},pmu{1}*geomean(y0),'k--');
set(h1(2),'LineWidth',2)
hold on
h2=semilogy(xtc{2},pmu{2}*geomean(y0),'k-');
set(h2(2),'LineWidth',2)
hold off
set(gca,'ytick',[10 30 100 300 1000 3000],'yticklabel',{'10' '30' '100' '300' '1000' '3000'},'ylim',[20 2000])
xlim([-1 3])
xlabel('WBC (log_{10}(50\times10^9/L))')
%ylabel('Expected lifetime (days)')
[hl,hlo]=legend([h1(2), h2(2)],'Female','Male');set(hl,'box','off')

subplot('position',[0.55 0.21 0.20 0.77]);
i1=2;i2=4;
[pmu,~,xtc]=gp_cpred(gpia, x, y, x, [i1 i2], 'z', ye, 'target','mu');
xtc{1}=denormdata(xtc{1},xmean(i2),xstd(i2));
xtc{2}=denormdata(xtc{2},xmean(i2),xstd(i2));
h1=semilogy(xtc{1},pmu{1}*geomean(y0),'k--');
set(h1(2),'LineWidth',2)
hold on
h2=semilogy(xtc{2},pmu{2}*geomean(y0),'k-');
set(h2(2),'LineWidth',2)
hold off
set(gca,'ytick',[10 30 100 300 1000 3000],'yticklabel',{'10' '30' '100' '300' '1000' '3000'},'ylim',[20 2000])
xlim([-7 10])
xlabel('Townsend deprivation index (TDI)')
%ylabel('Expected lifetime (days)')
[hl,hlo]=legend([h1(2), h2(2)],'Female','Male');set(hl,'box','off')

subplot('position',[0.79 0.21 0.20 0.77]);
i2=3;cla
[pmu,~,xtc]=gp_cpred(gpia, x, y, x, i2, 'z', ye,'var',[NaN -1 NaN -.368],'target','mu');
xtc=denormdata(xtc,xmean(i2),xstd(i2));
h1=semilogy(xtc,pmu*geomean(y0),'k--');
set(h1(2),'LineWidth',2)
[pmu,~,xtc]=gp_cpred(gpia, x, y, x, i2, 'z', ye,'var',[NaN -1 NaN 1.56],'target','mu');
xtc=denormdata(xtc,xmean(i2),xstd(i2));
hold on
h2=semilogy(xtc,pmu*geomean(y0),'k-');
set(h2(2),'LineWidth',2)
hold off
set(gca,'ytick',[10 30 100 300 1000 3000],'yticklabel',{'10' '30' '100' '300' '1000' '3000'},'ylim',[20 2000])
xlim([-1 3])
xlabel('WBC (log_{10}(50\times10^9/L))')
%ylabel('Expected lifetime (days)')
[hl,hlo]=legend([h1(2), h2(2)],'Female and TDI = -1','Female and TDI = 6');set(hl,'box','off')
