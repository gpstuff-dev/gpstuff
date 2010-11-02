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

% 1) Linear model with intercept and slope wrt time
cfc=gpcf_constant('constSigma2',1,'constSigma2_prior',[]);
cfl=gpcf_linear('coeffSigma2',10,'coeffSigma2_prior',[], ...
                'selectedVariables',1);
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
gp=gp_set('cf',{cfc cfl},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn,'opt',opt);
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,1)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Linear model')

%mfc=gpmf_constant();
%mfl=gpmf_linear('selectedVariables',1)
%gpm=gp_set('meanf',{mfc mfl},'noisef',{cfn},'jitterSigma2',1e-6);
%gpm=gp_optim(gpm,xn,yn);
%Ef=gp_pred(gpm,xn,yn,xn);
%Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
%plot(xx,Eff,'o-')

% 2) Linear model with hierarchical intercept
cfl=gpcf_linear('coeffSigma2',10,'coeffSigma2_prior',[], ...
                'selectedVariables',1);
% gpcf_exp with delta distance, produces hierarchial constant
% covariance, ie. hierarchical intercept
cfh=gpcf_exp('metric',metric_euclidean('components',{[2]}, ...
                                       'deltaflag',1, ...
                                       'lengthScale_prior',prior_t()));
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
gp=gp_set('cf',{cfh cfl},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn,'opt',opt);
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,2)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Linear model with random intercept')

% 3) Linear model with hierarchical intercept and slope
cfl=gpcf_linear('coeffSigma2',10,'coeffSigma2_prior',[], ...
                'selectedVariables',1);
% gpcf_exp with delta distance, produces hierarchial constant
% covariance, ie. hierarchical intercept
cfh=gpcf_exp('metric',metric_euclidean('components',{[2]}, ...
                                       'deltaflag',1, ...
                                       'lengthScale_prior',prior_t()));
% gpcf_exp with delta distance times linear covariance produces
% hierarchial linear covariance
cfhh=gpcf_exp(cfh,'magnSigma2',1,'magnSigma2_prior',[]);
cflh=gpcf_prod('functions',{cfl cfhh});
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
gp=gp_set('cf',{cfh cflh},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn,'opt',opt);
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,3)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Linear model with random intercept and slope')

% 4) Nonlinear model with hierarchical intercept
% include linear part, too
cfl=gpcf_linear('coeffSigma2',10,'coeffSigma2_prior',[], ...
                'selectedVariables',1);
% gpcf_exp with delta distance, produces hierarchial constant
% covariance, ie. hierarchical intercept
cfh=gpcf_exp('metric',metric_euclidean('components',{[2]}, ...
                                       'deltaflag',1, ...
                                       'lengthScale_prior',prior_t()));
% nonlinear part
cfs=gpcf_sexp('metric',metric_euclidean('components',{[1]},...
                                        'lengthScale_prior',prior_t()));
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
gp=gp_set('cf',{cfh cfl cfs},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn,'opt',opt);
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,4)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear model with random intercept')

% 5) Nonlinear model with hierarchical intercept and curve
% include linear part, too
cfl=gpcf_linear('coeffSigma2',10,'coeffSigma2_prior',[], ...
                'selectedVariables',1);
% nonlinear part
cfs=gpcf_sexp('metric',metric_euclidean('components',{[1]},...
                                        'lengthScale_prior',prior_t()));
% gpcf_exp with delta distance, produces hierarchial constant
% covariance, ie. hierarchical intercept
cfh=gpcf_exp('metric',metric_euclidean('components',{[2]}, ...
                                       'deltaflag',1, ...
                                       'lengthScale_prior',prior_t()));
% gpcf_exp with delta distance times linear covariance produces
% hierarchial linear covariance
cfhh=gpcf_exp(cfh,'magnSigma2',1,'magnSigma2_prior',[]);
cflh=gpcf_prod('functions',{cfl cfhh});
% gpcf_exp with delta distance times sexp covariance produces
% hierarchial sexp covariance
cfsh=gpcf_prod('functions',{cfs cfhh});
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
% combine hierarchical constant, linear and sexp covariances
gp=gp_set('cf',{cfh cflh cfsh},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn,'opt',opt);
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,5)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 1 with MAP')

% 6) With increasing flexibility of the modeling function
%    we need to integrate over the hyperparameteres
gps=gp_ia(gp,xn,yn);
Ef=gp_pred(gps,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,6)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 1 with IA')

% 7) Nonlinear model with hierarchical intercept and curve
%    Same as 5, but with no linear and product covariances
% nonlinear part with delta distance for ratid
cfs=gpcf_sexp('metric',metric_euclidean('components',{[1] [2]},...
                                        'deltaflag', [0 1], ...
                                        'lengthScale_prior',prior_t()));
% gpcf_exp with delta distance, produces hierarchial constant
% covariance, ie. hierarchical intercept
cfh=gpcf_exp('metric',metric_euclidean('components',{[2]}, ...
                                       'deltaflag',1, ...
                                       'lengthScale_prior',prior_t()));
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
% combine hierarchical constant and sexp covariances
gp=gp_set('cf',{cfh cfs},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn,'opt',opt);
Ef=gp_pred(gp,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,7)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 2 with MAP')

% 8) With increasing flexibility of the modeling function
%    we need to integrate over the hyperparameteres
gps=gp_ia(gp,xn,yn);
Ef=gp_pred(gps,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,8)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 2 with IA')

% 9) With neuralnetwork covariance and integration over the hyperparameters
cfnn=gpcf_neuralnetwork('selectedVariables',1);
% gpcf_exp with delta distance, produces hierarchial constant
% covariance, ie. hierarchical intercept
cfh=gpcf_exp('metric',metric_euclidean('components',{[2]}, ...
                                       'deltaflag',1, ...
                                       'lengthScale_prior',prior_t()));
% gpcf_exp with delta distance times linear covariance produces
% hierarchial linear covariance
cfhh=gpcf_exp(cfh,'magnSigma2',1,'magnSigma2_prior',[]);
% gpcf_exp with delta distance times sexp covariance produces
% hierarchial sexp covariance
cfnnh=gpcf_prod('functions',{cfnn cfhh});
cfn=gpcf_noise('noiseSigma2',.1,...
               'noiseSigma2_prior',prior_sinvchi2('s2',0.01,'nu',1));
gp=gp_set('cf',{cfh cfnnh},'noisef',{cfn},'jitterSigma2',1e-6);
gp=gp_optim(gp,xn,yn);
gps=gp_ia(gp,xn,yn);
Ef=gp_pred(gps,xn,yn,xn);
Eff=reshape(denormdata(Ef,ymean,ystd),nrats,ntime);
subplot(3,3,9)
plot(xx,Eff,'o-')
axis([7 37 100 400])
title('Non-linear hierarchical model 3 with IA')
