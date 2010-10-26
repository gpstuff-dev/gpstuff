%DEMO_DERIVATIVEOBS    Regression problem demonstration with derivative 
%                      observations
%
%  Description
%    The regression problem consist of a data with one input variable,
%    two output variables with Gaussian noise; observations and 
%    derivative observations. The constructed model is full GP with
%    Gaussian likelihood.
%
%    The covariance matrix K includes now also covariances between
%    derivative observations and between derivative and latent
%    observations. With derivative observations, the K matrix is a
%    block matrix with following blocks:
%
%        K = [K_ll K_Dl'; K_Dl K_DD]
%
%    Where D refers to derivative and l to latent observation and
%       K_ll = k(x_i, x_j | th)
%       K_Dl = d k(x_i, x_j | th) / dx_i
%       K_DD = d^2 k(x_i, x_j | th) / dx_i dx_j
%
%
%    To include derivative observations in the inference:
%
%       - provide partial derivative observations in the observation vector after
%         output observations y=[y;dy_1;...;dy_n]; 
%            for ex. if size(x)=[10 2] -> size(y)=[30 1] 
%
%       - after gp_init(...), type: gp.grad_obs=1;
%
%   The demo is organised in two parts:
%     1) data analysis without derivative observations
%     2) data analysis with derivative observations
%
%  See also  DEMO_REGRESSION1
%

% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

 % Create the data
 tp=9;                                  %number of training points -1
 x=-2:4/tp:2;
 y=sin(x).*cos(x).^2;                   % The underlying process
 dy=cos(x).^3 - 2*sin(x).^2.*cos(x);    % Derivative of the process
 ns=0.06;                              % noise standard deviation
 
 % Add noise
 y=y + ns*randn(size(y));
 dy=dy + ns*randn(size(dy));           % derivative obs are also noisy
 x=x';           
 dy=dy';
 
 y=y';          % observation vector without derivative observations
 y2=[y;dy];     % observation vector with derivative observations

 % test points
 p=-3:0.05:3;
 p=p';
 

%========================================================
% PART 1 data analysis with full GP model without derivative obs
%========================================================
 
gpcf1 = gpcf_sexp('lengthScale', 0.5, 'magnSigma2', .5);
gpcf2 = gpcf_noise('noiseSigma2', ns^2);

pl = prior_logunif();               % a prior structure
pm = prior_logunif();               % a prior structure
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_noise(gpcf2, 'noiseSigma2_prior', pm);

gp = gp_set('cf', {gpcf1},'noisef', {gpcf2}, 'jitterSigma2', 0.00001);

w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=@gp_e;     % create a function handle to negative log posterior
fg=@gp_g;     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);
% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);
% do the prediction
[Ef, Varx] = gp_pred(gp, x, y, p);

% PLOT THE DATA

figure
%m=shadedErrorBar(p,Ef(1:size(p)),2*sqrt(Varx(1:size(p))),{'k','lineWidth',2});
subplot(2,1,1)
m=plot(p,Ef(1:size(p)),'k','lineWidth',2);
hold on
plot(p,Ef(1:size(p))+2*sqrt(Varx(1:size(p))),'k--')
hold on
m95=plot(p,Ef(1:size(p))-2*sqrt(Varx(1:size(p))),'k--');
hold on
hav=plot(x, y(1:length(x)), 'ro','markerSize',7,'MarkerFaceColor','r');
hold on
h=plot(p,sin(p).*cos(p).^2,'b--','lineWidth',2);
%legend([m.mainLine m.patch h hav],'prediction','95%','f(x)','observations');
legend([m m95 h hav],'prediction','95%','f(x)','observations');
title('GP without derivative observations')
xlabel('input x')
ylabel('output y')

%========================================================
% PART 2 data analysis with full GP model with derivative obs
%========================================================

gpcf1 = gpcf_sexp('lengthScale', 0.5, 'magnSigma2', .5);
gpcf2 = gpcf_noise('noiseSigma2', ns^2);

pl = prior_logunif();               % a prior structure
pm = prior_logunif();               % a prior structure
gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_noise(gpcf2, 'noiseSigma2_prior', pm);

% Field grad_obs added to gp_init so that the derivatives are in use
gp = gp_set('cf', {gpcf1},'noisef', {gpcf2}, 'jitterSigma2', 0.00001,'grad_obs',1);

w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=@gp_e;     % create a function handle to negative log posterior
fg=@gp_g;     % create a function handle to gradient of negative log posterior


% Check gradients
gradcheck(w, fe, fg, gp, x, y2);

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y2);
% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);
% do the prediction
[Ef2, Varx2] = gp_pred(gp, x, y2, p);

% PLOT THE DATA
% plot lines indicating the derivative

%m=shadedErrorBar(p,Ef2(1:size(p)),2*sqrt(Varx2(1:size(p))),{'k','lineWidth',2});
subplot(2,1,2)
m=plot(p,Ef2(1:size(p)),'k','lineWidth',2);
hold on
plot(p,Ef2(1:size(p))+2*sqrt(Varx2(1:size(p))),'k--')
hold on
m95=plot(p,Ef(1:size(p))-2*sqrt(Varx2(1:size(p))),'k--');
hold on
hav=plot(x, y(1:length(x)), 'ro','markerSize',7,'MarkerFaceColor','r');
hold on
h=plot(p,sin(p).*cos(p).^2,'b--','lineWidth',2);

xlabel('input x')
ylabel('output y')
title('GP with derivative observations')

i1=0;
a=0.1;
ddx=zeros(2*length(x),1);
ddy=zeros(2*length(x),1);
for i=1:length(x)
    i1=i1+1;
    ddx(i1)=x(i)-a;
    ddy(i1)=y(i)-a*dy(i);
    i1=i1+1;
    ddx(i1)=x(i)+a;
    ddy(i1)=y(i)+a*dy(i);
end

for i=1:2:length(ddx)
hold on
dhav=plot(ddx(i:i+1), ddy(i:i+1),'r','lineWidth',2);
end
%legend([m.mainLine m.patch h hav dhav],'prediction','95%','f(x)','observations','der. obs.');
legend([m m95 h hav dhav],'prediction','95%','f(x)','observations','der. obs.');

