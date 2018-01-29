%DEMO_SURVIVAL_COXPH  Survival model using Cox proportional hazard model 
%
%  Description 
%    Survival model using Cox proportional model with a piecewise
%    log-constant baseline hazard. The hazard rate is 
%   
%       h(t) = h_0(t)*exp(f),
%
%    where the baseline hazard is assumed to piecewise log-constant. 
%
%    The inference is conducted via Laplace, where we find
%    Gaussian approximation for p(f| th, data), where th is the
%    maximum a posterior (MAP) estimate for the parameters.
%    
%    The censoring indicator ye is
%    
%      ye = 0 for uncensored event
%      ye = 1 for right censored event.
%
%    If survival times y for n observation are given as nx2 matrix with
%    entry times into follow-up in the first column and exit times from
%    follow-up in the second column, left truncated right censored
%    modelling is possible, for instance, in cases where age is wanted to
%    be set as a baseline hazard.  
%
%    Example data set is leukemia survival data in Northwest England
%    presented in (Henderson, R., Shimakura, S., and Gorst, D. (2002).
%    Modeling spatial variation in leukemia survival data. Journal of the
%    American Statistical Association, 97:965–972). Data set was downloaded
%    from http://www.math.ntnu.no/%7Ehrue/r-inla.org/examples/leukemia/leuk.dat
%
%    Note that the log-logistic model in DEMO_SURVIVAL_AFT works
%    better for this data.
%
%  Reference
%
%    Heikki Joensuu, Aki Vehtari, Jaakko Riihimäki, Toshirou Nishida,
%    Sonja E Steigen, Peter Brabec, Lukas Plank, Bengt Nilsson,
%    Claudia Cirilli, Chiara Braconi, Andrea Bordoni, Magnus K
%    Magnusson, Zdenek Linke, Jozef Sufliarsky, Federico Massimo, Jon
%    G Jonasson, Angelo Paolo Dei Tos and Piotr Rutkowski (2012). Risk
%    of gastrointestinal stromal tumour recurrence after surgery: an
%    analysis of pooled population-based cohorts. In The Lancet
%    Oncology, 13(3):265-274.
%
%  See also  DEMO_SURVIVAL_AFT
%
% Copyright (c) 2011 Jaakko Riihimäki
% Copyright (c) 2013 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% First load data
S = which('demo_survival_coxph');
L = strrep(S,'demo_survival_coxph.m','demodata/leukemia.txt');
leukemiadata=load(L);

% leukemiadata consists of:
% 'time', 'cens', 'xcoord', 'ycoord', 'age', 'sex', 'wbc', 'tpi', 'district'

% survival times
y0=leukemiadata(:,1);
% scale survival times 
y=y0/500;

ye=1-leukemiadata(:,2); % event indicator, ye = 0 for uncensored event
                  %                        ye = 1 for right censored event

% for simplicity choose 'age', 'sex', 'wbc', and 'tpi' covariates
x0=leukemiadata(:,5:8);
x=x0;
[n, m]=size(x);
% transform white blood cell count (wbc), which highly skewed
% distribution with zeros for measurements below measurement accuracy
x(:,3)=log10(x(:,3)+0.3);
% normalize continuous covariates 
[x(:,[1 3:4]),xmean(:,[1 3:4]),xstd(:,[1 3:4])]=normdata(x(:,[1 3:4]));
% binary sex covariate is not transformed
xmean(2)=0;xstd(2)=1;

% number of time intervals
ntime=50;
% create finite partition of time axis
S=linspace(0,max(y)+0.001,ntime+1);

% Create the covariance functions
plh = prior_invt('s2',1, 'nu', 4);
pl = prior_t('s2',1, 'nu', 4);
pm = prior_t('s2',1, 'nu', 4); 

% covariance for hazard function
cfhc = gpcf_constant('constSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
cfhl = gpcf_linear('coeffSigma2',1,'coeffSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
cfhs = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1.1, 'lengthScale_prior', plh, 'magnSigma2_prior', pm);
% covariance for proportional part
cfc = gpcf_constant('constSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
cfl = gpcf_linear('coeffSigma2',1,'coeffSigma2_prior',prior_sinvchi2('s2',1^2,'nu',1));
cfs = gpcf_sexp('lengthScale', ones(1,m),'lengthScale_prior',pl,'magnSigma2_prior',pm);

% Create the likelihood structure
lik = lik_coxph('S', S);

% NOTE! if multiple covariance functions per latent is used, define
% gp.comp_cf as follows:
% gp = gp_set(..., 'comp_cf' {[1 2 3] [4 5]};
% where [1 2 3] are for hazard function, and [4 5] for proportional part
gp = gp_set('lik', lik, 'cf', {cfhc cfhl cfhs cfl cfs}, 'jitterSigma2', 1e-6, 'comp_cf', {[1 2 3] [4 5]});

% Set the approximate inference method to Laplace
gp = gp_set(gp, 'latent_method', 'Laplace');

opt=optimset('TolFun',1e-2,'TolX',1e-2,'Display','iter');
gp=gp_optim(gp,x,y,'z',ye,'opt',opt);
[~,~,lp]=gp_pred(gp,x,y,'z',ye);sum(lp)-log(500)*sum(~ye)
% -5.9202e+03
[crit,cvpreds]=gp_kfcv(gp,x,y,'z',ye,'inf_method','MAP','display','iter');crit.mlpd_cv*n

figure
set(gcf,'units','centimeters');
set(gcf,'pos',[29 6 24 6])
subplot('position',[0.07 0.21 0.20 0.77]);
i1=2;i2=1;
[Ef,Varf,xtc]=gp_cpred(gp, x, y, x, [i1 i2], 'z', ye, 'target', 'f');
xtc{1}=denormdata(xtc{1},xmean(i2),xstd(i2));
xtc{2}=denormdata(xtc{2},xmean(i2),xstd(i2));
h1=plot(xtc{1},Ef{1},'k--',xtc{1},Ef{1}-1.64*sqrt(Varf{1}),'k--',xtc{1},Ef{1}+1.64*sqrt(Varf{1}),'k--');
set(h1(1),'LineWidth',2)
hold on
h2=plot(xtc{2},Ef{2},'k-',xtc{2},Ef{2}-1.64*sqrt(Varf{2}),'k--',xtc{2},Ef{2}+1.64*sqrt(Varf{2}),'k-');
set(h2(1),'LineWidth',2)
hold off
ylim([-4 2])
xlabel('Age (years)')
ylabel('Log-hazard')
[hl,hlo]=legend([h1(1), h2(1)],{'Female','Male'},'location','northwest');set(hl,'box','off')

subplot('position',[0.31 0.21 0.20 0.77]);
i1=2;i2=3;
[Ef,Varf,xtc]=gp_cpred(gp, x, y, x, [i1 i2], 'z', ye, 'target', 'f');
xtc{1}=denormdata(xtc{1},xmean(i2),xstd(i2));
xtc{2}=denormdata(xtc{2},xmean(i2),xstd(i2));
h1=plot(xtc{1},Ef{1},'k--',xtc{1},Ef{1}-1.64*sqrt(Varf{1}),'k--',xtc{1},Ef{1}+1.64*sqrt(Varf{1}),'k--');
set(h1(1),'LineWidth',2)
hold on
h2=plot(xtc{2},Ef{2},'k-',xtc{2},Ef{2}-1.64*sqrt(Varf{2}),'k-',xtc{2},Ef{2}+1.64*sqrt(Varf{2}),'k-');
set(h2(1),'LineWidth',2)
hold off
%ylim([-5 0])
ylim([-4 2])
xlim([-1 3])
xlabel('WBC (log_{10}(50\times10^9/L))')
%ylabel('Expected lifetime (days)')
[hl,hlo]=legend([h1(1), h2(1)],{'Female','Male'},'location','northwest');set(hl,'box','off')

subplot('position',[0.55 0.21 0.20 0.77]);
i1=2;i2=4;
[Ef,Varf,xtc]=gp_cpred(gp, x, y, x, [i1 i2], 'z', ye, 'target', 'f');
xtc{1}=denormdata(xtc{1},xmean(i2),xstd(i2));
xtc{2}=denormdata(xtc{2},xmean(i2),xstd(i2));
h1=plot(xtc{1},Ef{1},'k--',xtc{1},Ef{1}-1.64*sqrt(Varf{1}),'k--',xtc{1},Ef{1}+1.64*sqrt(Varf{1}),'k--');
set(h1(1),'LineWidth',2)
hold on
h2=plot(xtc{2},Ef{2},'k-',xtc{2},Ef{2}-1.64*sqrt(Varf{2}),'k-',xtc{2},Ef{2}+1.64*sqrt(Varf{2}),'k-');
set(h2(1),'LineWidth',2)
hold off
%ylim([-5 0])
ylim([-4 2])
xlim([-7 10])
xlabel('Townsend deprivation index')
%ylabel('Expected lifetime (days)')
[hl,hlo]=legend([h1(1), h2(1)],{'Female','Male'},'location','northwest');set(hl,'box','off')

subplot('position',[0.79 0.21 0.20 0.77]);
i2=0;cla
[Ef,Varf,xtc]=gp_cpred(gp, x, y, x, i2, 'z', ye, 'target', 'f');
%gp_cpred(gp, x, y, x, i2, 'z', ye);
xtc=xtc*500/365;
h1=plot(xtc,Ef,'k-',xtc,Ef-1.64*sqrt(Varf),'k-',xtc,Ef+1.64*sqrt(Varf),'k-');
set(h1(1),'LineWidth',2)
ylim([-5 3])
%xlim([-1 3])
xlabel('Time (years)')
%ylabel('Expected lifetime (days)')
hl=legend(h1(1),'Baseline','location','northwest');set(hl,'box','off')
ylim([-4 2])

