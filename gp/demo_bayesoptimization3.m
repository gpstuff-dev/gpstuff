%DEMO_BAYESIANOPTIMIZATION3  A demonstration program for Bayesian
%                            optimization with constraints
%
% The set of BO demos
%  Part 1: see demo_bayesoptimization1
%  One dimensional example 
%
%  Part 2: see demo_bayesoptimization3
%  Two dimensional example 
%
%  Part 3: this file
%  Two dimensional example with constraints 
%  * The implementation of constraints follows Gelbart et al. (2014)
% 
%  References:
%    Jones, D., Schonlau, M., & Welch, W. (1998). Efficient global
%    optimization of expensive black-box functions. Journal of Global
%    Optimization, 13(4), 455-492. doi:10.1023/a:1008306431147  
%
%    Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams
%    (2014). Bayesian Optimization with Unknown Constraints.
%    http://arxiv.org/pdf/1403.5607v1.pdf
%
%    Snoek, J., Larochelle, H, Adams, R. P. (2012). Practical Bayesian
%    Optimization of Machine Learning Algorithms. NIPS 25 
%
%  Copyright (c) 2015 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%%  Part 3:
%  Two dimensional example with constraints 
% For testing purposes:
stack = dbstack;
if (~isempty(stack) && (strcmp(stack(end).name, 'runtestset') || strcmp(stack(end).name, 'runtests'))) test = 1; else test = 0; end;

rng(3)
% Construct function handles to objective function (fx) and two constraint
% functions (fxc, fxc2)
fx = @(x) -log( (mvnpdf([x(:,1) x(:,2)],[3.5 2.5], [1 0.3; 0.3 1]) + 0.3*mvnpdf([x(:,1) x(:,2)],[7 8], [3 0.5; 0.5 4])).*...
    mvnpdf([x(:,1) x(:,2)],[5 5], [100 0; 0 100])) ./15 -1;
fxc = @(x) ((x(:,1)-5) .^2 + (x(:,2)-5).^2 -1)/30;
fxc2 = @(x) ( (x(:,1)-5) .^2)./30 - 0.5;

% The upper and lower limits for the constraints
const = [0 0.8 ; -10 0.1];

% Help variables for visualization
lb=0;
ub=10;
[X,Y] = meshgrid(linspace(lb,ub,100),linspace(lb,ub,100));
xl = [X(:) Y(:)];
Z = reshape(fx(xl),100,100);
Zc1 = fxc(xl); Zc1(Zc1<const(1,1) | Zc1>const(1,2)) = nan; 
Zc1(~isnan(Zc1))=1; Zc1 = reshape(Zc1,100,100);
Zc2 = fxc2(xl); Zc2(Zc2<const(2,1) | Zc2>const(2,2)) = nan;
Zc2(~isnan(Zc2))=1; Zc2 = reshape(Zc2,100,100);

% ----- conduct Bayesian optimization -----

% construct GP models for the objective function and constraint functions
cfc = gpcf_constant('constSigma2',10,'constSigma2_prior', prior_fixed);
cfse = gpcf_sexp('lengthScale',[1 1]);
cfl = gpcf_linear('coeffSigma2', 10); 
cfl2 = gpcf_squared('coeffSigma2', 10, 'interactions', 'on');
lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_fixed);
% GP model for objective function
%gp1 = gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik);cfl, cfl2, 
gp1 = gp_set('cf', {cfc, cfse}, 'lik', lik);
% GP models for constraint functions
gpc1 = {gp_set('cf', {cfc, cfse}, 'lik', lik, 'jitterSigma2', 1e-6),...
    gp_set('cf', {cfc, cfse}, 'lik', lik, 'jitterSigma2', 1e-6)};

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','on','Algorithm','interior-point','TolFun',1e-9,'TolX',1e-6, 'Display', 'iter');
opt=optimset(optdefault);
lb=[0 0];     % lower bound of the input space
ub=[10 10];   % upper bound of the input space

% draw initial points
% we assume that at first we don't have any observation from the objective
% function but only observations from constraint functions. Hence x and y
% are initialized to zero
x = [];
y = [];
xc1 = 10*rand(2,2);
yc1 = fxc(xc1);
xc2 = 10*rand(5,2);
yc2 = fxc2(xc2);

figure, % figure for visualization
i1 = 1;
maxiter = 25;
improv = inf;   % improvement between two successive query points
while i1 < maxiter && improv>1e-6
%while i1 < maxiter
    % Train the GP models and calculate variables that are needed when
    %   calculating the Expected improvement (Acquisition function) 
    % Objective function
    if ~isempty(x)
        gp = gp_optim(gp1,x,y);
        [K, C] = gp_trcov(gp,x);
        invC = inv(C);
        a = C\y;
        fmin = min( fx(x) );
    else
        a=[];
        x=[];
        invC=[];
        fmin=[];
        gp=gp1;
    end
    % constrain function 1
    gpct = gp_optim(gpc1{1},xc1,yc1);
    [~, Cct] = gp_trcov(gpct,xc1);
    const1.gpc = gpct;
    const1.invCc = inv(Cct);
    const1.ac = Cct\yc1;
    const1.const = const(1,:);
    const1.xc = xc1;
    % constrain function 2
    gpct = gp_optim(gpc1{2},xc2,yc2);
    [~, Cct] = gp_trcov(gpct,xc2);
    const2.gpc = gpct;
    const2.invCc = inv(Cct);
    const2.ac = Cct\yc2;
    const2.const = const(2,:);
    const2.xc = xc2;
       
    % Calculate EI and the posterior of the functions for visualization
    if ~isempty(x)
        [Ef,Varf] = gp_pred(gp, x, y, xl);
        EI = expectedimprovement_eg(xl, gp, x, a, invC, fmin, const1, const2);
    else
        Ef = zeros(size(xl,1),1);
        Varf = zeros(size(xl,1),1);
        EI = zeros(size(xl,1),1);
    end
    [Efc1] = gp_pred(const1.gpc, xc1, yc1, xl);
    [Efc2] = gp_pred(const2.gpc, xc2, yc2, xl);

    % optimize acquisition function
    %  * Note! Opposite to the standard notation we minimize negative Expected
    %    Improvement since Matlab optimizers seek for functions minimum
    %  * Note! We alternate the acquisition function between Expected
    %    Improvement and expected variance. The latter helps the
    %    optimization so that it does not get stuck in local mode
    % Here we use multiple starting points for the optimization so that we
    % don't crash into suboptimal mode of acquisition function
%     if mod(i1,5)==0  %Do just exploration by finding the maimum variance location        
%         fh_eg = @(x_new) expectedvariance_eg(x_new, gp, x, [], invC);
%     else
        fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin, const1, const2);
%     end
    nstarts = 20;
    xstart = [repmat(lb,nstarts,1) + repmat(ub-lb,nstarts,1).*rand(nstarts,2) ]; %; repmat(x(indbest,:),2,1)+0.1*randn(2,size(x,2))
    for s1=1:nstarts
        x_new(s1,:) = optimf(fh_eg, xstart(s1,:), [], [], [], [], lb, ub, [], opt);
    end
    xnews = x_new;
    EIs = fh_eg(x_new);
    x_new = x_new( find(EIs==min(EIs),1), : );
        
    % New sample point
    x(end+1,:) = x_new;
    y(end+1,:) = fx(x(end,:));
    xc1(end+1,:) = x_new;
    yc1(end+1,:) = fxc(x(end,:));
    xc2(end+1,:) = x_new;
    yc2(end+1,:) = fxc2(x(end,:));

    % visualize
    clf
    % Plot the objective function
    subplot(2,4,1),hold on, title('Objective, query points')
    pcolor(X,Y,Z),shading flat
    clim = caxis;
    plot(x(1:end-1,1),x(1:end-1,2), 'rx', 'MarkerSize', 10),
    plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3)
    % Plot the posterior mean of the GP model
    subplot(2,4,2),hold on, title(sprintf('GP prediction, mean, iter: %d',i1))
    pcolor(X,Y,reshape(Ef,100,100)),shading flat
    caxis(clim)
    % Plot the posterior variance of GP model
    subplot(2,4,6),hold on, title('GP prediction, variance')
    pcolor(X,Y,reshape(Varf,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10)
    plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3)
    plot(x(1:end-1,1),x(1:end-1,2), 'rx', 'MarkerSize', 10),
    
    % The expected information    
    subplot(2,4,5), hold on, title(sprintf('Expected improvement %.2e', min(EIs)))
    pcolor(X,Y,reshape(EI,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10)
    plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3)
    plot(x(1:end-1,1),x(1:end-1,2), 'rx'),
    
    % constraint 1
    subplot(2,4,3), hold on, title(sprintf('constraint 1'))
    pcolor(X,Y,Zc1),shading flat    
    plot(xc1(1:end-1,1),xc1(1:end-1,2), 'rx', 'MarkerSize', 10);
    plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    plot(xc1(end,1),xc1(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    % constraint 2
    subplot(2,4,4), hold on, title(sprintf('constraint 2'))
    pcolor(X,Y,Zc2),shading flat    
    l1= plot(xc2(1:end-1,1),xc2(1:end-1,2), 'rx', 'MarkerSize', 10);
    l2=plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    l3=plot(xc2(end,1),xc2(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    legend([l1,l2,l3], {'function evaluation points','local modes of acquisition function','The next query point'})
    
    % prediction of constraint 1
    subplot(2,4,7), hold on, title(sprintf('prediction for const 1'))
    Efc1(Efc1<const(1,1) | Efc1>const(1,2)) = nan; 
    Efc1(~isnan(Efc1))=1; Efc1 = reshape(Efc1,100,100);
    pcolor(X,Y,reshape(Efc1,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10)
    plot(xc1(end,1),xc1(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3)
    plot(xc1(1:end-1,1),xc1(1:end-1,2), 'rx'),    
    % prediction of constraint 2
    subplot(2,4,8), hold on, title(sprintf('prediction for const 2'))
    Efc2(Efc2<const(2,1) | Efc2>const(2,2)) = nan; 
    Efc2(~isnan(Efc2))=1; Efc2 = reshape(Efc2,100,100);
    pcolor(X,Y,reshape(Efc2,100,100)),shading flat
    plot(xc2(1:end-1,1),xc2(1:end-1,2), 'rx', 'MarkerSize', 10),
    plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10)
    plot(xc2(end,1),xc2(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3)
    
      
    if length(y)>1
        improv = abs(y(end) - y(end-1));
    end
    i1=i1+1;
    
    if test == 0
        pause
    end
end

