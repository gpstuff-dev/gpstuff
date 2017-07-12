%DEMO_DERIVATIVES   Demonstration on use of observations on derivatives,
%                   monotonicity infomration and predictions for derivative
%                   of the process  
%
%  This demo demonstrates how to use derivative observations in various
%  settings. 
%
%  See also  DEMO_REGRESSION1, DEMO_SPATIAL1
%

% Copyright (c) 2017 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.



%% One dimensional example with direct derivative observations

% Construct data
% ======================================================
rng(90)
ftarget    = @(x) sin(x)+0.1*(x-2).^2-0.005.*x.^3;
ftargetDer = @(x) cos(x)+0.2*(x-2)-0.015.*x.^2;
%gradcheck(5,ftarget,ftargetDer);

% Regular observations
x = linspace(1,10,5)';
y = ftarget(x) + 0.05*randn(size(x));
% Derivative obseravtions
xd = linspace(2,8,5)';
yd = ftargetDer(xd) + 0.05*randn(size(xd));
% All observations
xt = [x zeros(size(x)) ; xd ones(size(x)) ];
yt = [y;yd];
% prediction points
xpred = linspace(0,10,100)';
xpredt = [xpred zeros(size(xpred)) ; xpred ones(size(xpred)) ];

% Inference without derivative observations
% ======================================================
lik = lik_gaussian;
cf = gpcf_sexp('lengthScale', 1.5);
gp = gp_set('lik', lik, 'cf', {cf}, 'deriv', 2);
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt(xt(:,2)==0,:),y,'opt',opt);
[Ef_p1,Varf_p1] = gp_pred(gp,xt(xt(:,end)==0,:),y,xpredt);

figure, 
subplot(1,2,1)
plot(xpred,ftarget(xpred),'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpred,ftargetDer(xpred), 'r')
plot(xpredt(xpredt(:,2)==0,1),Ef_p1(xpredt(:,2)==0), 'b--')
plot(xpredt(xpredt(:,2)==0,1),Ef_p1(xpredt(:,2)==0)+2*sqrt(Varf_p1(xpredt(:,2)==0)), 'b:')
plot(xpredt(xpredt(:,2)==0,1),Ef_p1(xpredt(:,2)==0)-2*sqrt(Varf_p1(xpredt(:,2)==0)), 'b:')

plot(xpredt(xpredt(:,2)==1,1),Ef_p1(xpredt(:,2)==1), 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef_p1(xpredt(:,2)==1)+2*sqrt(Varf_p1(xpredt(:,2)==1)), 'r:')
plot(xpredt(xpredt(:,2)==1,1),Ef_p1(xpredt(:,2)==1)-2*sqrt(Varf_p1(xpredt(:,2)==1)), 'r:')
title('Only observations from the function')

% Inference with derivative observations
% ======================================================
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
% gradcheck(gp_pak(gp), @gp_e, @gp_g, gp, xt, yt);
gp=gp_optim(gp,xt,yt,'opt',opt);

[Ef_p1,Varf_p1] = gp_pred(gp,xt,yt,xpredt);
subplot(1,2,2) 
plot(xpred,ftarget(xpred),'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==0,1),Ef_p1(xpredt(:,2)==0), 'b--')
plot(xpredt(xpredt(:,2)==0,1),Ef_p1(xpredt(:,2)==0)+2*sqrt(Varf_p1(xpredt(:,2)==0)), 'b:')
plot(xpred,ftargetDer(xpred), 'r')
plot(xd,yd,'r.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==1,1),Ef_p1(xpredt(:,2)==1), 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef_p1(xpredt(:,2)==1)+2*sqrt(Varf_p1(xpredt(:,2)==1)), 'r:')
plot(xpredt(xpredt(:,2)==0,1),Ef_p1(xpredt(:,2)==0)-2*sqrt(Varf_p1(xpredt(:,2)==0)), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef_p1(xpredt(:,2)==1)-2*sqrt(Varf_p1(xpredt(:,2)==1)), 'r:')
legend('true function','observation', 'posterior mean of function', '95% credible int of function', 'derivative function', 'deriv obs', 'posterior mean of derivative', '95% credible int of derivative')
title('function + derivative observations')

%% 1D example with monotonicity information and gaussian likelihood 
% for function observations

% simulate data
% =============================
% Regular observations
x = linspace(1,10,5)';
y = ftarget(x) + 0.05*randn(size(x));
% monotonicity observations
xd = linspace(2,8,7)';
yd = 2*double(ftargetDer(xd) > 0)-1;
% All data
xt = [x zeros(size(x)) ; xd ones(size(xd)) ];
yt = [y;yd];
xpredt = [xpred zeros(size(xpred)) ; xpred ones(size(xpred)) ];

% The "likelihood covariate" which tells which likelihood to use for which
% row of data
z = [ones(size(y)) ; 2*ones(size(yd))];

% covariance function
cf = gpcf_sexp('lengthScale',1.5);

% -----
% without monotonicity information
gp = gp_set('lik', lik_gaussian, 'cf', {cf}, 'deriv', 2);
opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,xt(xt(:,2)==0,:),y,'opt',opt);
[Ef1_p2,Varf1_p2] = gp_pred(gp,xt(xt(:,end)==0,:),y,xpredt); %, 'zt', zt


figure, 
subplot(1,3,1)
plot(xpred,ftarget(xpred),'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpred,ftargetDer(xpred), 'r')
plot(xpredt(xpredt(:,2)==0,1),Ef1_p2(xpredt(:,2)==0), 'b--')
plot(xpredt(xpredt(:,2)==0,1),Ef1_p2(xpredt(:,2)==0)+2*sqrt(Varf1_p2(xpredt(:,2)==0)), 'b:')
plot(xpredt(xpredt(:,2)==0,1),Ef1_p2(xpredt(:,2)==0)-2*sqrt(Varf1_p2(xpredt(:,2)==0)), 'b:')

plot(xpredt(xpredt(:,2)==1,1),Ef1_p2(xpredt(:,2)==1), 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef1_p2(xpredt(:,2)==1)+2*sqrt(Varf1_p2(xpredt(:,2)==1)), 'r:')
plot(xpredt(xpredt(:,2)==1,1),Ef1_p2(xpredt(:,2)==1)-2*sqrt(Varf1_p2(xpredt(:,2)==1)), 'r:')
title('Only observations from the function')



% With monotonicity information
%===================
likpr = lik_probit('nu', 1e-6);
lik = lik_liks('likelihoods', {lik_gaussian,likpr},'classVariables', 1) ;


% EP implementation
gp = gp_set('lik', lik, 'cf', {cf}, 'latent_method', 'EP', 'deriv', 2);
%gradcheck(gp_pak(gp), @gpep_e, @gpep_g, gp, xt, yt, 'z', z);

% Optimize with the scaled conjugate gradient method
opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,xt,yt,'opt',opt, 'z', z);
[Ef_p2,Varf_p2] = gp_pred(gp,xt,yt,xpredt, 'z', z); %, 'zt', zt


subplot(1,3,2)
plot(xpred,ftarget(xpred) ,'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpred, ftargetDer(xpred), 'r')
plot(xd,yd,'r.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==0,1),  Ef_p2(xpredt(:,2)==0) , 'b--')
plot(xpredt(xpredt(:,2)==0,1), Ef_p2(xpredt(:,2)==0)+2*sqrt(Varf_p2(xpredt(:,2)==0) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),  Ef_p2(xpredt(:,2)==1) , 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef_p2(xpredt(:,2)==1)+2*sqrt(Varf_p2(xpredt(:,2)==1) ), 'r:')
%legend('true function','observation','posterio mean of function','95% credible int of function', 'true derivative','monotonicity observation','posterior mean of derivative','95% credible int of derivative')
plot(xpredt(xpredt(:,2)==0,1),Ef_p2(xpredt(:,2)==0)-2*sqrt(Varf_p2(xpredt(:,2)==0) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef_p2(xpredt(:,2)==1)-2*sqrt(Varf_p2(xpredt(:,2)==1) ), 'r:')
title('EP approximation, with monotonicity information')

% Laplace implementation
% -----
gp = gp_set('lik', lik, 'cf', {cf}, 'latent_method', 'Laplace', 'deriv', 2);
%gradcheck(gp_pak(gp), @gpla_e, @gpla_g, gp, xt, yt, 'z', z);

opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt,yt,'opt',opt, 'z', z);

[Ef_l_p2,Varf_l_p2] = gp_pred(gp,xt,yt,xpredt, 'z', z); %, 'zt', zt
%figure, 
subplot(1,3,3)
plot(xpred,ftarget(xpred) ,'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==0,1),  Ef_l_p2(xpredt(:,2)==0) , 'b--')
plot(xpredt(xpredt(:,2)==0,1), Ef_l_p2(xpredt(:,2)==0)+2*sqrt(Varf_l_p2(xpredt(:,2)==0) ), 'b:')
plot(xpred, ftargetDer(xpred), 'r')
plot(xd,yd,'r.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==1,1),  Ef_l_p2(xpredt(:,2)==1) , 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef_l_p2(xpredt(:,2)==1)+2*sqrt(Varf_l_p2(xpredt(:,2)==1) ), 'r:')
plot(xpredt(xpredt(:,2)==0,1),Ef_l_p2(xpredt(:,2)==0)-2*sqrt(Varf_l_p2(xpredt(:,2)==0) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef_l_p2(xpredt(:,2)==1)-2*sqrt(Varf_l_p2(xpredt(:,2)==1) ), 'r:')
legend('true function','observation','posterio mean of function','95% credible int of function', 'true derivative','monotonicity observation','posterior mean of derivative','95% credible int of derivative')
title('Laplace approximation, monotonicity information')


%% 1D example with monotonicity information and non-gaussian likelihoods

% Simulate data
% =============================
% Poisson observations
x = linspace(1,10,10)';
y = poissrnd( exp(ftarget(x)) );
% monotonicity observations
xd = linspace(2,8,5)';
yd = 2*double(ftargetDer(xd) > 0)-1;
% All observations
xt = [x zeros(size(x)) ; xd ones(size(xd)) ];
yt = [y;yd];
xpredt = [xpred zeros(size(xpred)) ; xpred ones(size(xpred)) ];

% introduce the "likelihood covariate" which tells which likelihood to use
z = [ones(size(y)) ; 2*ones(size(yd))];
%zt = [ones(size(xpredt,1),1) ; 2*ones(size(xpredt,1),1)];

likpr = lik_probit('nu', 1e-6); 
lik = lik_liks('likelihoods', {lik_poisson,likpr},'classVariables', 1) ;
cf = gpcf_sexp('selectedVariables', 1, 'lengthScale', 1.5);

% EP implementation
% -----
% without monotonicity information
gp = gp_set('lik', lik_poisson, 'cf', {cf}, 'latent_method', 'EP', 'deriv', 2);
opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,xt(xt(:,2)==0,:),y,'opt',opt);
%gp=gp_optim(gp,x,y,'opt',opt);
%[Ef1,Varf1] = gp_pred(gp,x,y,xpredt(:,1)); %, 'zt', zt
[Ef1_p3,Varf1_p3] = gp_pred(gp,xt(xt(:,2)==0,:),y,xpredt); %, 'zt', zt


% With monotonicity information
% ------------------------------
gp = gp_set('lik', lik, 'cf', {cf}, 'latent_method', 'EP', 'deriv', 2);
% gradcheck(gp_pak(gp), @gpep_e, @gpep_g, gp, xt, yt, 'z', z);

% Optimize with the scaled conjugate gradient method
opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,xt,yt,'opt',opt, 'z', z);
[Ef_p3,Varf_p3] = gp_pred(gp,xt,yt,xpredt, 'z', z); %, 'zt', zt


figure, 
subplot(1,3,1)
plot(xpred,exp( ftarget(xpred) ),'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpred, ftargetDer(xpred), 'r')
plot(xpredt(xpredt(:,2)==0,1), exp( Ef1_p3(xpredt(:,2)==0) ), 'b--')
plot(xpredt(xpredt(:,2)==0,1),exp( Ef1_p3(xpredt(:,2)==0)+2*sqrt(Varf1_p3(xpredt(:,2)==0)) ), 'b:')
plot(xpredt(xpredt(:,2)==0,1),exp( Ef1_p3(xpredt(:,2)==0)-2*sqrt(Varf1_p3(xpredt(:,2)==0)) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef1_p3(xpredt(:,2)==1), 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef1_p3(xpredt(:,2)==1)+2*sqrt(Varf1_p3(xpredt(:,2)==1)), 'r:')
plot(xpredt(xpredt(:,2)==1,1),Ef1_p3(xpredt(:,2)==1)-2*sqrt(Varf1_p3(xpredt(:,2)==1)), 'r:')
title('EP approximation, NO monotonicity information')


subplot(1,3,2)
plot(xpred,exp( ftarget(xpred) ),'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpred, ftargetDer(xpred), 'r')
plot(xd,yd,'r.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==0,1), exp( Ef_p3(xpredt(:,2)==0) ), 'b--')
plot(xpredt(xpredt(:,2)==0,1),exp( Ef_p3(xpredt(:,2)==0)+2*sqrt(Varf_p3(xpredt(:,2)==0)) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef_p3(xpredt(:,2)==1), 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef_p3(xpredt(:,2)==1)+2*sqrt(Varf_p3(xpredt(:,2)==1)), 'r:')
%legend('true intensity','observation', 'posterior mean of intensity', '95% credible int of intensity','derivative of log intensity', 'deriv obs', 'posterior mean of deriv', '95% credible int of deriv')
plot(xpredt(xpredt(:,2)==0,1),exp( Ef_p3(xpredt(:,2)==0)-2*sqrt(Varf_p3(xpredt(:,2)==0)) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef_p3(xpredt(:,2)==1)-2*sqrt(Varf_p3(xpredt(:,2)==1)), 'r:')
title('EP approximation, monotonicity information')


% Laplace implementation
% ----------------------
gp = gp_set('lik', lik, 'cf', {cf}, 'latent_method', 'Laplace', 'deriv', 2);
%gradcheck(gp_pak(gp), @gpla_e, @gpla_g, gp, xt, yt, 'z', z);

opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt,yt,'opt',opt, 'z', z);

[Ef_l_p3,Varf_l_p3] = gp_pred(gp,xt,yt,xpredt, 'z', z); %, 'zt', zt
%figure, 
subplot(1,3,3)
plot(xpred,exp( ftarget(xpred) ),'b')
hold on
plot(x,y,'b.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==0,1), exp( Ef_l_p3(xpredt(:,2)==0) ), 'b--')
plot(xpredt(xpredt(:,2)==0,1),exp( Ef_l_p3(xpredt(:,2)==0)+2*sqrt(Varf_l_p3(xpredt(:,2)==0)) ), 'b:')
plot(xpred, ftargetDer(xpred), 'r')
plot(xd,yd,'r.','MarkerSize', 10)
plot(xpredt(xpredt(:,2)==1,1),Ef_l_p3(xpredt(:,2)==1), 'r--')
plot(xpredt(xpredt(:,2)==1,1),Ef_l_p3(xpredt(:,2)==1)+2*sqrt(Varf_l_p3(xpredt(:,2)==1)), 'r:')
plot(xpredt(xpredt(:,2)==0,1),exp( Ef_l_p3(xpredt(:,2)==0)-2*sqrt(Varf_l_p3(xpredt(:,2)==0)) ), 'b:')
plot(xpredt(xpredt(:,2)==1,1),Ef_l_p3(xpredt(:,2)==1)-2*sqrt(Varf_l_p3(xpredt(:,2)==1)), 'r:')
legend('true intensity','observation', 'posterior mean of intensity', '95% credible int of intensity','derivative of log intensity', 'deriv obs', 'posterior mean of deriv', '95% credible int of deriv')
title('Laplace approximation, monotonicity information')



%% =======================================================================
% 2D examples
% =======================================================================

%% 2D Derivative observations
% =======================================================================

% Construct data
% ======================================================
ftarget    = @(x) sin(x(:,1))+0.1*(x(:,2)-2).^2-0.005.*x(:,2).^3;
ftargetDer = @(x) [cos(x(:,1)) 0.2*(x(:,2)-2)-0.015.*x(:,2).^2];
%gradcheck([5 5],ftarget,ftargetDer);


%x = 10*rand(5,1);
% Regular observations
x = 10*rand(15,2);
y = ftarget(x) + 0.05*randn(size(x,1),1);
% Derivative observations
xd = 10*rand(15,2);                                 % locations of derivative observations
ydtmp = ftargetDer(xd) + 0.05*randn(size(xd));      % derivative observations for both dimensions
derivInd = ceil(2*rand(size(ydtmp,1),1));
yd=zeros(size(xd,1),1);
for i1 = 1:length(derivInd)
    yd(i1) = ydtmp(i1,derivInd(i1));               % However, use randomly only one derivative direction per location
end
% All observations
xt = [x zeros(size(x,1),1) ; xd derivInd ];
yt = [y;yd];
% prediction points
[X1,X2]=meshgrid(linspace(0,10,50),linspace(0,10,50));
xpred = [X1(:) X2(:)];
xpredt = [xpred zeros(size(xpred,1),1) ; xpred ones(size(xpred,1),1) ; xpred 2*ones(size(xpred,1),1) ];


% Inference without derivative observations
% ======================================================
lik = lik_gaussian;
cf = gpcf_sexp('lengthScale', 1.5, 'selectedVariables', 1:2);
%cf = gpcf_sexp;
gp = gp_set('lik', lik, 'cf', {cf}, 'deriv', 3);
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt(xt(:,end)==0,:),y,'opt',opt);
[Ef1_p4,Varf1_p4] = gp_pred(gp,xt(xt(:,end)==0,:),y,xpredt);

figure, 
subplot(3,2,1)
mesh(X1,X2,reshape(ftarget([X1(:),X2(:)]),size(X1)))
title('true function')
subplot(3,2,2)
mesh(X1,X2,reshape(Ef1_p4(xpredt(:,end)==0),size(X1)))
title('posterior mean')
subplot(3,2,3)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,1),size(X1)))
title('true derivative with respect to x_1')
subplot(3,2,4)
mesh(X1,X2,reshape(Ef1_p4(xpredt(:,end)==1),size(X1)))
title('posterior mean of derivative with respect to x_1')
subplot(3,2,5)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,2),size(X1)))
title('true derivative with respect to x_2')
subplot(3,2,6)
mesh(X1,X2,reshape(Ef1_p4(xpredt(:,end)==2),size(X1)))
title('posterior mean of derivative with respect to x_2')


% Inference with derivative observations
% ======================================================

lik = lik_gaussian;
%cf = gpcf_sexp('lengthScale', [1.5 1.5]);
gp = gp_set('lik', lik, 'cf', {cf}, 'deriv', 3);
% gradcheck(gp_pak(gp), @gp_e, @gp_g, gp, xt, yt);

opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt,yt,'opt',opt);

[Ef_p4,Varf_p4] = gp_pred(gp,xt,yt,xpredt);

figure
subplot(3,2,1)
mesh(X1,X2,reshape(ftarget([X1(:),X2(:)]),size(X1)))
title('true function')
subplot(3,2,2)
mesh(X1,X2,reshape(Ef_p4(xpredt(:,end)==0),size(X1)))
title('posterior mean')
subplot(3,2,3)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,1),size(X1)))
title('true derivative with respect to x_1')
subplot(3,2,4)
mesh(X1,X2,reshape(Ef_p4(xpredt(:,end)==1),size(X1)))
title('posterior mean of derivative with respect to x_1')
subplot(3,2,5)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,2),size(X1)))
title('true derivative with respect to x_2')
subplot(3,2,6)
mesh(X1,X2,reshape(Ef_p4(xpredt(:,end)==2),size(X1)))
title('posterior mean of derivative with respect to x_2')


%% 2D Derivative observations with additive GP
% =======================================================================

% Construct data
% ======================================================
ftarget    = @(x) sin(x(:,1))+0.1*(x(:,2)-2).^2-0.005.*x(:,2).^3;
ftargetDer = @(x) [cos(x(:,1)) 0.2*(x(:,2)-2)-0.015.*x(:,2).^2];
%gradcheck([5 5],ftarget,ftargetDer);


%x = 10*rand(5,1);
% Regular observations
x = 10*rand(15,2);
y = ftarget(x) + 0.05*randn(size(x,1),1);
% Derivative observations
xd = 10*rand(15,2);                                 % locations of derivative observations
ydtmp = ftargetDer(xd) + 0.05*randn(size(xd));      % derivative observations for both dimensions
derivInd = ceil(2*rand(size(ydtmp,1),1));
yd=zeros(size(xd,1),1);
for i1 = 1:length(derivInd)
    yd(i1) = ydtmp(i1,derivInd(i1));               % However, use randomly only one derivative direction per location
end
% All observations
xt = [x zeros(size(x,1),1) ; xd derivInd ];
yt = [y;yd];
% prediction points
[X1,X2]=meshgrid(linspace(0,10,50),linspace(0,10,50));
xpred = [X1(:) X2(:)];
xpredt = [xpred zeros(size(xpred,1),1) ; xpred ones(size(xpred,1),1) ; xpred 2*ones(size(xpred,1),1) ];

% Inference without derivative observations
% ======================================================
lik = lik_gaussian;
cf = gpcf_sexp('selectedVariables',1);
cf2 = gpcf_sexp('selectedVariables',2);
gp = gp_set('lik', lik, 'cf', {cf cf2}, 'deriv', 3);
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt(xt(:,end)==0,:),y,'opt',opt);
[Ef1_p5,Varf1_p5] = gp_pred(gp,xt(xt(:,end)==0,:),y,xpredt);

figure, 
subplot(3,2,1)
mesh(X1,X2,reshape(ftarget([X1(:),X2(:)]),size(X1)))
title('true function')
subplot(3,2,2)
mesh(X1,X2,reshape(Ef1_p5(xpredt(:,end)==0),size(X1)))
title('posterior mean')
subplot(3,2,3)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,1),size(X1)))
title('true derivative with respect to x_1')
subplot(3,2,4)
mesh(X1,X2,reshape(Ef1_p5(xpredt(:,end)==1),size(X1)))
title('posterior mean of derivative with respect to x_1')
subplot(3,2,5)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,2),size(X1)))
title('true derivative with respect to x_2')
subplot(3,2,6)
mesh(X1,X2,reshape(Ef1_p5(xpredt(:,end)==2),size(X1)))
title('posterior mean of derivative with respect to x_2')


% Inference with derivative observations
% ======================================================

lik = lik_gaussian;
gp = gp_set('lik', lik, 'cf', {cf cf2}, 'deriv', 3);
% gradcheck(gp_pak(gp), @gp_e, @gp_g, gp, xt, yt);

opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt,yt,'opt',opt);

[Ef_p5,Varf_p5] = gp_pred(gp,xt,yt,xpredt);

figure
subplot(3,2,1)
mesh(X1,X2,reshape(ftarget([X1(:),X2(:)]),size(X1)))
title('true function')
subplot(3,2,2)
mesh(X1,X2,reshape(Ef_p5(xpredt(:,end)==0),size(X1)))
title('posterior mean')
subplot(3,2,3)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,1),size(X1)))
title('true derivative with respect to x_1')
subplot(3,2,4)
mesh(X1,X2,reshape(Ef_p5(xpredt(:,end)==1),size(X1)))
title('posterior mean of derivative with respect to x_1')
subplot(3,2,5)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,2),size(X1)))
title('true derivative with respect to x_2')
subplot(3,2,6)
mesh(X1,X2,reshape(Ef_p5(xpredt(:,end)==2),size(X1)))
title('posterior mean of derivative with respect to x_2')



%% 2D Derivative observations with additive GP
% Monotonicity information
% =======================================================================

% Construct data
% ======================================================
ftarget    = @(x) sin(x(:,1))+0.1*(x(:,2)-2).^2-0.005.*x(:,2).^3;
ftargetDer = @(x) [cos(x(:,1)) 0.2*(x(:,2)-2)-0.015.*x(:,2).^2];
%gradcheck([5 5],ftarget,ftargetDer);


%x = 10*rand(5,1);
% Regular observations
x = 10*rand(15,2);
y = ftarget(x) + 0.05*randn(size(x,1),1);
% Derivative observations
xd = 10*rand(15,2);                                 % locations of derivative observations
ydtmp = ftargetDer(xd) + 0.05*randn(size(xd));      % derivative observations for both dimensions
derivInd = ceil(2*rand(size(ydtmp,1),1));
yd=zeros(size(xd,1),1);
for i1 = 1:length(derivInd)
    yd(i1) = ydtmp(i1,derivInd(i1));               % However, use randomly only one derivative direction per location
end
% All observations
xt = [x zeros(size(x,1),1) ; xd derivInd ];
yt = [y;yd./abs(yd)];
% prediction points
[X1,X2]=meshgrid(linspace(0,10,50),linspace(0,10,50));
xpred = [X1(:) X2(:)];
xpredt = [xpred zeros(size(xpred,1),1) ; xpred ones(size(xpred,1),1) ; xpred 2*ones(size(xpred,1),1) ];

% Inference without derivative observations
% ======================================================
lik = lik_gaussian;
cf = gpcf_sexp('selectedVariables',1);
cf2 = gpcf_sexp('selectedVariables',2);
gp = gp_set('lik', lik, 'cf', {cf cf2}, 'deriv', 3);
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt(xt(:,end)==0,:),y,'opt',opt);
[Ef1_p6,Varf1_p6] = gp_pred(gp,xt(xt(:,end)==0,:),y,xpredt);

figure, 
subplot(3,2,1)
mesh(X1,X2,reshape(ftarget([X1(:),X2(:)]),size(X1)))
title('true function')
subplot(3,2,2)
mesh(X1,X2,reshape(Ef1_p6(xpredt(:,end)==0),size(X1)))
title('posterior mean')
subplot(3,2,3)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,1),size(X1)))
title('true derivative with respect to x_1')
subplot(3,2,4)
mesh(X1,X2,reshape(Ef1_p6(xpredt(:,end)==1),size(X1)))
title('posterior mean of derivative with respect to x_1')
subplot(3,2,5)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,2),size(X1)))
title('true derivative with respect to x_2')
subplot(3,2,6)
mesh(X1,X2,reshape(Ef1_p6(xpredt(:,end)==2),size(X1)))
title('posterior mean of derivative with respect to x_2')


% Inference with derivative observations
% ======================================================

lik = lik_liks('likelihoods', {lik_gaussian, lik_probit('nu',1e-6)}, 'classVariables', 1);
gp = gp_set('lik', lik, 'cf', {cf cf2}, 'deriv', 3, 'latent_method', 'EP');
z = [ones(size(y)) ; 2*ones(size(yd))];
% gradcheck(gp_pak(gp), @gp_e, @gp_g, gp, xt, yt, 'z', z);

opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,xt,yt,'opt',opt, 'z', z);

[Ef_p6,Varf_p6] = gp_pred(gp,xt,yt,xpredt, 'z', z);

figure
subplot(3,2,1)
mesh(X1,X2,reshape(ftarget([X1(:),X2(:)]),size(X1)))
title('true function')
subplot(3,2,2)
mesh(X1,X2,reshape(Ef_p6(xpredt(:,end)==0),size(X1)))
title('posterior mean')
subplot(3,2,3)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,1),size(X1)))
title('true derivative with respect to x_1')
subplot(3,2,4)
mesh(X1,X2,reshape(Ef_p6(xpredt(:,end)==1),size(X1)))
title('posterior mean of derivative with respect to x_1')
subplot(3,2,5)
dertemp=ftargetDer([X1(:),X2(:)]);
mesh(X1,X2,reshape(dertemp(:,2),size(X1)))
title('true derivative with respect to x_2')
subplot(3,2,6)
mesh(X1,X2,reshape(Ef_p6(xpredt(:,end)==2),size(X1)))
title('posterior mean of derivative with respect to x_2')


