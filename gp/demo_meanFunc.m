%DEMO_MEANFUNC        Regression problem demonstration for GP model with a
%                     mean function
%
%    Description
%    The regression problem consist of a data with one input variable and
%    one output variable with Gaussian noise. The problem is modelled with
%    Full GP model with gaussian likelihood and a specified mean function.
%    The mean function m is a weighted sum of some basis functions h, where
%
%                   m=h'*Beta
%
%    and we have set a gaussian prior for the weights Beta
%
%                   Beta ~ N(b,B)
%
%    Inference is done according to Rasmussen and Williams (2006) p. 27-29.
%
%    In this demonstration the data is from an arbitrary function:
%
%         y = x + x^2 + 4*cos(x)*sin(x) + epsilon
%
%    and we define the basis functions to be:
%
%       h = [x;x²],    where h is a 2 x n row matrix,            
%
%    References:
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.

 % Create the data

 t=-2:0.6:2;
 res=4*cos(t).*sin(t)+0.4*randn(size(sin(t)));
 y2=t' + t'.^2 +res';
 x=t';
 
gpcf1 = gpcf_sexp('init', 'lengthScale', [0.5], 'magnSigma2', .5);
gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.1^2);

pl = prior_logunif('init');               % a prior structure
pm = prior_logunif('init');               % a prior structure
%pm = prior_t('init', 's2', 0.3);               % a prior structure
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigma2_prior', pm);

% Set the prior values for mean functions
b=[0.5;0.5];
B=diag([1;1]);
% Define which mean functions are used
s={@gpmf_linear,@gpmf_squared};

gp = gp_init('init', 'FULL', 'gaussian', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.00001,'meanFuncs',s,'mean_p',{b,B});



w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y2);
% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);
p=-3:0.1:3;
p=p';
[Ef, Varx] = gp_pred(gp, x, y2, p);

% PLOT THE DATA

figure
m=plot(p,Ef,'k','lineWidth',2);
hold on
plot(p,Ef(1:size(p))+2*sqrt(Varx(1:size(p))),'k--')
hold on
m95=plot(p,Ef(1:size(p))-2*sqrt(Varx(1:size(p))),'k--');
hold on
hav=plot(x,y2,'r*');
hold on
h=plot(p,p+p.^2+4*cos(p).*sin(p),'b--','lineWidth',2);
hold on
mean=plot(p,p+p.^2,'r--','lineWidth',1);
legend([m m95 h mean hav],'prediction','95%','f(x)','mean function','observations');
xlabel('input x')
ylabel('output y')
title('GP regression with a mean function')


