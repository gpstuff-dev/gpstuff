%DEMO_REGRESSION_HIER  Hierarchical regression demonstration using
%                      Rats data
%
%  Description
%    The example data is taken from section 6 of Gelfand et al
%    (1990) (also used in WinBUGS/OpenBUGS), and concerns 30 young
%    rats whose weights were measured weekly for five week. This
%    demo demosntrates how to make hierarchical linear and
%    non-linear models.
%
%  Reference
%    Gelfand, A. E., Hills, S. E., Racine-Poon, A. and Smith, A. F. 
%    M. (1990) Illustration of Bayesian Inference in Normal Data
%    Models Using Gibbs Sampling. Journal of the American
%    Statistical Association 85(412):972-985.
%

% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

S = which('demo_regression_hier');
L = strrep(S,'demo_regression_hier.m','demos/rats.mat');
data=load(L);
xx = data.x;
yy = data.y;
% Show data : 5 weight measurements per rat for 30 rats
figure
plot(xx,yy,'o-')
axis([7 37 100 400])
title('Data')

% Reshape data
ntime = size(xx,2);
nrats = size(yy,1);
% All y's to one vector
y=yy(:);
% Repeat x for each rat
x=reshape(repmat(xx,nrats,1),ntime*nrats,1);
% Add ratid
x=[x repmat([1:nrats]',ntime,1)];
% Now 'x' consist of the inputs (ratid,time) and 'y' of the output (weight). 
% Normalize x and y
[xn,xmean,xstd]=normdata(x);
[yn,ymean,ystd]=normdata(y);

% optmization options
opt=optimset('TolFun',1e-5,'TolX',1e-5);

% common noise term with weakly informative prior
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
% common categorical covariance term
cc=gpcf_cat('selectedVariables',2);

% 1) Linear model with intercept and slope wrt time
cfc=gpcf_constant('constSigma2',1);
cfl=gpcf_linear('coeffSigma2',1,'selectedVariables',1);
% construct GP
gp=gp_set('cf',{cfc cfl},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn,'opt',opt);
% predict and plot
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,1)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Linear model')

% 2) Linear model with hierarchical intercept
cfc=gpcf_constant('constSigma2',1);
cfl=gpcf_linear('coeffSigma2',1,'selectedVariables',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% construct GP
gp=gp_set('cf',{cfc cfci cfl},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn,'opt',opt);
% predict and plot
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,2)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Linear model with random intercept')

% 3) Linear model with hierarchical intercept and slope
cfc=gpcf_constant('constSigma2',1);
cfl=gpcf_linear('coeffSigma2',1,'selectedVariables',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% linear covariance term for each rat
cfli=gpcf_prod('cf',{cfl cc});
% construct GP
gp=gp_set('cf',{cfc cfci cfl cfli},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn,'opt',opt);
% predict and plot
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,3)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Linear model with random intercept and slope')

% 4) Nonlinear model with hierarchical intercept
% include linear part, too
cfc=gpcf_constant('constSigma2',1);
cfl=gpcf_linear('coeffSigma2',1,'selectedVariables',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% nonlinear part
cfs=gpcf_sexp('selectedVariables',[1],'lengthScale_prior',prior_t());
% construct GP
gp=gp_set('cf',{cfc cfci cfl cfs},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn,'opt',opt);
% predict and plot
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,4)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear model with random intercept')

% 5) Nonlinear model with hierarchical intercept and curve
% include linear part, too
cfc=gpcf_constant('constSigma2',1);
cfl=gpcf_linear('coeffSigma2',1,'selectedVariables',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% linear covariance term for each rat
cfli=gpcf_prod('cf',{cfl cc});
% nonlinear part
cfs=gpcf_sexp('selectedVariables',1,'lengthScale_prior',prior_t());
% nonlinear covariance term for each rat
cfsi=gpcf_prod('cf',{cfs cc});
% construct GP
gp=gp_set('cf',{cfc cfci cfl cfli cfs cfsi},'noisef',{cfn},...
          'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn,'opt',opt);
% predict and plot
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,5)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 1 with MAP')

% 6) With increasing flexibility of the modeling function
%    we need to integrate over the hyperparameteres
% integrate over hyperparameters
gps=gp_ia(gp,xn,yn);
% predict and plot
Ef=gp_pred(gps,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,6)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 1 with IA')

% 7) Nonlinear model with hierarchical intercept and curve
%    Same as 5, but with no linear and product covariances
cfc=gpcf_constant('constSigma2',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% nonlinear part with delta distance for ratid
cfs=gpcf_sexp('metric',metric_euclidean('components',{[1] [2]},...
                                        'deltadist', [0 1], ...
                                        'lengthScale_prior',prior_t()));
% construct GP
gp=gp_set('cf',{cfc cfci cfs},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn,'opt',opt);
% predict and plot
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,7)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 2 with MAP')

% 8) With increasing flexibility of the modeling function
%    we need to integrate over the hyperparameteres
% integrate over hyperparameters
gps=gp_ia(gp,xn,yn);
% predict and plot
Ef=gp_pred(gps,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,8)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 2 with IA')

% 9) With neuralnetwork covariance and integration over the hyperparameters
cfc=gpcf_constant('constSigma2',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% nonlinear part with neuralnetwork covariance
cfnn=gpcf_neuralnetwork('selectedVariables',1);
% nonlinear covariance term for each rat
cfnni=gpcf_prod('cf',{cfnn cc});
% construct GP
gp=gp_set('cf',{cfc cfci cfnn cfnni},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn);
% integrate over hyperparameters
gps=gp_ia(gp,xn,yn);
% predict and plot
Ef=gp_pred(gps,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,9)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 3 with IA')

%*** Missing Data Example ***
% In the original paper (Gelfand et al, 1990) data was aslo to
% demonstrate misiing data handling, by removing 1-4 weeks of data
% for part of the rats. Handling missing data in this case is trivial
% for GPs, too

S = which('demo_regression_hier');
L = strrep(S,'demo_regression_hier.m','demos/rats.mat');
data=load(L);
xx = data.x;
yy = data.y;
yymiss = data.ymiss;
% Show data : 5 weight measurements per rat for 30 rats
figure
plot(xx,yymiss,'o-')
axis([7 37 100 400])
title('Data')
% Reshape data
ntime = size(xx,2);
nrats = size(yy,1);
% All y's to one vector
y=yymiss(:);
% Repeat x for each rat
x=reshape(repmat(xx,nrats,1),ntime*nrats,1);
% Add ratid
x=[x repmat([1:nrats]',ntime,1)];
% Now 'x' consist of the inputs (ratid,time) and 'y' of the output (weight). 
% Normalize x and y
[xn,xmean,xstd]=normdata(x);
[yn,ymean,ystd]=normdata(y);
% test x is the complete x
xnt=xn;
% remove missing data from the training data
missi=isnan(y);
yn(missi,:)=[];
xn(missi,:)=[];

% 10) neuralnetwork covariance, IA and missing data
cfc=gpcf_constant('constSigma2',1);
% own constant term for each rat
cfci=gpcf_prod('cf',{cfc cc});
% nonlinear part with neuralnetwork covariance
cfnn=gpcf_neuralnetwork('selectedVariables',1);
% nonlinear covariance term for each rat
cfnni=gpcf_prod('cf',{cfnn cc});
% construct GP
gp=gp_set('cf',{cfc cfci cfnn cfnni},'noisef',{cfn},'jitterSigma2',1e-6);
% optimize
gp=gp_optim(gp,xn,yn);
% integrate over hyperparameters
gps=gp_ia(gp,xn,yn);
% predict and plot
Ef=gp_pred(gps,xn,yn,xnt);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
Effc=Eff;Effc(isnan(y))=NaN;
plot(xx,Effc,'bo-',xx,Eff,'bo--')
axis([7 37 100 400])
title('Non-linear hierarchical model 3 with IA and missing data')
