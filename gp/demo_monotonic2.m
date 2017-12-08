%DEMO_MONOTONIC2  Demonstration of monotonicity information in GP Poisson regression
%
%  Description
%    This demonstration shows the benefit of adding monotonicity
%    information in GP Poisson regression of mortality data.
%
%  References
%
%    Riihimäki and Vehtari (2010). Gaussian processes with
%    monotonicity information.  Journal of Machine Learning Research:
%    Workshop and Conference Proceedings, 9:645-652.
%
%    Broffitt, J. D. (1988). Increasing and increasing convex Bayesian
%    graduation. Transactions of the Society of Actuaries, 40(1), 115-48.
%
%  See also
%    DEMO_*, GP_MONOTONIC, DEMO_MONOTONIC1
%
% Copyright (c) 2014 Ville Tolvanen, Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Note that as number of the observations is quite small, it would
% better to sample hyperparameters with MCMC, but for faster demo CCD
% integration with gp_ia is used below

% Mortality rate data from Broffitt (1988)
% N is the number of people insured under a certain policy
% y is the number of insured who died
S = which('demo_monotonic2');
L = strrep(S,'demo_monotonic2.m','demodata/broffit.txt');
d=readtable(L,'Delimiter',';');
x=d.age;
y=d.y;
z=d.N.*(sum(d.y)./sum(d.N));
zt=repmat(1000,size(y)).*(sum(d.y)./sum(d.N));
[xn,nd.xmean,nd.xstd]=normdata(x);

% Basic GP
pl=prior_t();
pm=prior_sqrtt();
cfl=gpcf_linear('coeffSigma2',10^2,'coeffSigma2_prior',[]);
cfs=gpcf_sexp('lengthScale',.4,'lengthScale_prior',pl,'magnSigma2_prior', pm);
lik=lik_poisson();
% Alternative GP models a) sexp b) lin + sexp
gpa = gp_set('lik', lik, 'cf', {cfs}, 'jitterSigma2', 1e-6, 'latent_method', 'EP');
gpb = gp_set('lik', lik, 'cf', {cfl cfs}, 'jitterSigma2', 1e-6, 'latent_method', 'EP');

% 1) sexp
subplot(2,2,1)
gpiaa=gp_ia(gpa,xn,y,'z',z);
% Do predictions for testing purposes
[Eft,Varft]=gp_pred(gpiaa, xn,y,xn,'z',z);
gp_plot(gpiaa,xn,y,'z',z,'zt',zt,'target','mu','normdata',nd)
hold on
plot(d.age,d.y./d.N*1000,'r*')
axis([34 65 0 23])
xlabel('Age')
ylabel('Deaths/1000')
title('sexp')

% 2) sexp + monotonicity
subplot(2,2,3)
opt=optimset('TolX',1e-1,'TolFun',1e-1,'Display','iter');
gpam=gpa;gpam.xv=xn(2:2:end);
gpam=gp_monotonic(gpam,xn,y,'z',z,'nvd', 1, 'optimize', 'on', ...
                  'opt', opt, 'optimf', @fminlbfgs);
gpiaam=gp_ia(gpam,xn,y,'z',z);
% Do predictions for testing purposes
[Eftm,Varftm]=gp_pred(gpiaam, xn,y,xn,'z',z);
gp_plot(gpiaam,xn,y,'z',z,'zt',zt,'target','mu','normdata',nd)
hold on
plot(d.age,d.y./d.N*1000,'r*')
axis([34 65 0 23])
xlabel('Age')
ylabel('Deaths/1000')
title('sexp + monot.')

% 3) lin + sexp
subplot(2,2,2)
gpiab=gp_ia(gpb,xn,y,'z',z);
gp_plot(gpiab,xn,y,'z',z,'zt',zt,'target','mu','normdata',nd)
hold on
plot(d.age,d.y./d.N*1000,'r*')
axis([34 65 0 23])
xlabel('Age')
ylabel('Deaths/1000')
title('lin + sexp')

% 4) lin + sexp + monotonicity
subplot(2,2,4)
opt=optimset('TolX',1e-1,'TolFun',1e-1,'Display','iter');
gpbm=gpb;gpbm.xv=xn(2:2:end);
gpbm=gp_monotonic(gpb,xn,y,'z',z,'nvd', 1, 'optimize', 'on', ...
                  'opt', opt, 'optimf', @fminlbfgs);
gpiabm=gp_ia(gpbm,xn,y,'z',z);
gp_plot(gpiabm,xn,y,'z',z,'zt',zt,'target','mu','normdata',nd)
hold on
plot(d.age,d.y./d.N*1000,'r*')
axis([34 65 0 23])
xlabel('Age')
ylabel('Deaths/1000')
title('lin + sexp + monot.')
