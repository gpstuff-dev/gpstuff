%DEMO_BAYESIANOPTIMIZATIO  A demonstration program for Bayesian
%                          optimization
%
%  Part 1:
%  One dimensional example 
%
%  Part 2:
%  Two dimensional example 
%
%  Part 3:
%  Two dimensional example with constraints 
%  * The implementation of constraints follows Gelbart et al. 
% 
%  References:
%   Jones, D., Schonlau, M., & Welch, W. (1998). Efficient global
%   optimization of expensive black-box functions. Journal of Global
%   Optimization, 13(4), 455–492. doi:10.1023/a:1008306431147  
%
%   Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams. Bayesian
%   Optimization with Unknown Constraints.
%   http://arxiv.org/pdf/1403.5607v1.pdf 
%
%   Snoek, J., Larochelle, H, Adams, R. P. (2012). Practical Bayesian
%   Optimization of Machine Learning Algorithms. NIPS 25 
%
% Copyright (c) 2015 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%%  Part 1:
%  One dimensional example 
clear

% Construct a function to be optimized
xl = linspace(0,10,100)';
fx = @(x) 0.6*x -0.1*x.^2 + sin(2*x);

% construct GP
cfse = gpcf_sexp('lengthScale',1);
lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_fixed);
gp = gp_set('cf', {cfse}, 'lik', lik);

% ----- conduct Bayesian optimization -----
% draw initial point

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','off','Algorithm','SQP','TolFun',1e-6,'TolX',1e-3);
opt=optimset(optdefault);
lb=0;     % lower bound of the input space
ub=10;    % upper bound of the input space

% draw initial point
x = 10*rand;
y(1) = fx(x);

figure, % figure for visualization
i1 = 1;
maxiter = 15;
improv = inf;   % improvement between two successive query points
while i1 < maxiter && improv>1e-6
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    gp = gp_optim(gp,x,y);
    [K, C] = gp_trcov(gp,x);
    invC = inv(C);
    a = C\y;
    fmin = min( fx(x) );
    
    % Calculate EI and posterior of the function for visualization purposes
    EI = expectedimprovement_eg(xl, gp, x, a, invC, fmin);
    [Ef,Varf] = gp_pred(gp, x, y, xl); 

    % optimize acquisition function
    %    Note! Opposite to the standard notation we minimize negative Expected
    %    Improvement since Matlab optimizers seek for functions minimum
    % Here we use multiple starting points for the optimization so that we
    % don't chrash into suboptimal mode
    fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin); % The function handle to the Expected Improvement function
    indbest = find(y == fmin);
    xstart = [linspace(0.5,9.5,5) x(indbest)+0.1*randn(1,2)];
    for s1=1:length(xstart)
        x_new(s1) = optimf(fh_eg, xstart(s1), [], [], [], [], lb, ub, [], opt);
    end
    EIs = expectedimprovement_eg(x_new(:), gp, x, a, invC, fmin);    
    x_new = x_new( find(EIs==min(EIs),1) ); % pick up the point where Expected Improvement is maximized
        
    % put new sample point to the list of evaluation points
    x(end+1) = x_new;
    y(end+1) = fx(x(end));  % calculate the function value at query point
    x=x(:);y=y(:);

    % visualize
    clf
    subplot(2,1,1),hold on, title('function to be optimized and GP fit')
    plot(xl,fx(xl))
    plot(x(1:end-1),y(1:end-1), 'k*')
    % the posterior of the function
    plot(xl,Ef, 'k')
    plot(xl,Ef + 2*sqrt(Varf), 'k--')
    plot(xl,Ef - 2*sqrt(Varf), 'k--')
    % The new sample location
    plot(x(end),y(end), 'r*')
    % The expected information    
    subplot(2,1,2)
    plot(xl,EI, 'r'), hold on
    plot(x(end),0, 'r*')
    title('acquisition function')

       
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    pause
end

%%  Part 2:
%  Two dimensional example 
clear

fx = @(x) -log( (mvnpdf([x(:,1) x(:,2)],[3.5 2.5], [1 0.3; 0.3 1]) + 0.3*mvnpdf([x(:,1) x(:,2)],[7 8], [3 0.5; 0.5 4])).*...
    mvnpdf([x(:,1) x(:,2)],[5 5], [100 0; 0 100])) ./15 -1;

% Help variables for visualization
lb=0;
ub=10;
[X,Y] = meshgrid(linspace(lb,ub,100),linspace(lb,ub,100));
xl = [X(:) Y(:)];
Z = reshape(fx(xl),100,100);

% construct GP to model the function
cfc = gpcf_constant('constSigma2',10,'constSigma2_prior', prior_fixed);
cfse = gpcf_sexp('lengthScale',[1 1]);
cfl = gpcf_linear('coeffSigma2', 10); 
cfl2 = gpcf_squared('coeffSigma2', 10);
lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_fixed);
gp = gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik);

% ----- conduct Bayesian optimization -----

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','off','Algorithm','trust-region-reflective','TolFun',1e-8,'TolX',1e-5);
opt=optimset(optdefault);
lb=[0 0];     % lower bound of the input space
ub=[10 10];   % upper bound of the input space

% draw initial points
x = 10*rand(5,2);
y = fx(x);

figure, % figure for visualization
i1 = 1;
maxiter = 20;
improv = inf;   % improvement between two successive query points
while i1 < maxiter && improv>1e-6
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    gp = gp_optim(gp,x,y);
    [K, C] = gp_trcov(gp,x);
    invC = inv(C);
    a = C\y;
    fmin = min( fx(x) );
    
    % Calculate EI and the posterior of the function for visualization
    [Ef,Varf] = gp_pred(gp, x, y, xl);
    EI = expectedimprovement_eg(xl, gp, x, a, invC, fmin);

    % optimize acquisition function
    %  * Note! Opposite to the standard notation we minimize negative Expected
    %    Improvement since Matlab optimizers seek for functions minimum
    %  * Note! We alternate the acquisition function between Expected
    %    Improvement and expected variance. The latter helps the
    %    optimization so that it does not get stuck in local mode
    % Here we use multiple starting points for the optimization so that we
    % don't chrash into suboptimal mode of acquisition function
    if mod(i1,5)==0  % Do just exploration by finding the maimum variance location
        fh_eg = @(x_new) expectedvariance_eg(x_new, gp, x, [], invC);
    else
        fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin);
    end
    indbest = find(y == fmin);
    nstarts = 10;
    xstart = [repmat(lb,nstarts,1) + repmat(ub-lb,nstarts,1).*rand(nstarts,2) ]; 
    for s1=1:length(xstart)
        x_new(s1,:) = optimf(fh_eg, xstart(s1,:), [], [], [], [], lb, ub, [], opt);
    end
    xnews = x_new;
    EIs = expectedimprovement_eg(x_new, gp, x, a, invC, fmin);
    x_new = x_new( find(EIs==min(EIs),1), : ); % pick up the point where Expected Improvement is maximized
        
    % put new sample point to the list of evaluation points
    x(end+1,:) = x_new;
    y(end+1,:) = fx(x(end,:));     % calculate the function value at query point

    % visualize
    clf
    subplot(2,2,1),hold on, title('Objective, query points')
    pcolor(X,Y,Z),shading flat
    clim = caxis;
    plot(x(1:end-1,1),x(1:end-1,2), 'k*'),
    plot(x(end,1),x(end,2), 'kX', 'MarkerSize', 15)
    % the posterior of the function
    subplot(2,2,2),hold on, title(sprintf('GP prediction, mean, iter: %d',i1))
    pcolor(X,Y,reshape(Ef,100,100)),shading flat
    caxis(clim)
    subplot(2,2,4),hold on, title('GP prediction, variance')
    pcolor(X,Y,reshape(Varf,100,100)),shading flat
    % The expected improvement 
    subplot(2,2,3), hold on, title(sprintf('Expected improvement %.2e', min(EIs)))
    pcolor(X,Y,reshape(EI,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'kX', 'MarkerSize', 15)
    plot(x(end,1),x(end,2), 'wX', 'MarkerSize', 15)
    
       
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    pause
end

%%  Part 3:
%  Two dimensional example with constraints 
clear

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
cfl2 = gpcf_squared('coeffSigma2', 10);
lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_fixed);
% GP model for objective function
gp1 = gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik);
% GP models for constraint functions
gpc1 = {gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik, 'jitterSigma2', 1e-6),...
    gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik, 'jitterSigma2', 1e-6)};

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','on','Algorithm','interior-point','TolFun',1e-8,'TolX',1e-5, 'Display', 'iter');
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
    % don't chrash into suboptimal mode of acquisition function
    if mod(i1,5)==0  %Do just exploration by finding the maimum variance location        
        fh_eg = @(x_new) expectedvariance_eg(x_new, gp, x, [], invC);
    else
        fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin, const1, const2);
    end
    nstarts = 10;
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
    subplot(2,4,1),hold on, title('Objective, query points')
    pcolor(X,Y,Z),shading flat
    clim = caxis;
    plot(x(1:end-1,1),x(1:end-1,2), 'k*'),
    plot(x(end,1),x(end,2), 'kX', 'MarkerSize', 15)
    % the posterior of the function
    subplot(2,4,2),hold on, title(sprintf('GP prediction, mean, iter: %d',i1))
    pcolor(X,Y,reshape(Ef,100,100)),shading flat
    caxis(clim)
    subplot(2,4,6),hold on, title('GP prediction, variance')
    pcolor(X,Y,reshape(Varf,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'kX', 'MarkerSize', 15)
    plot(x(end,1),x(end,2), 'wX', 'MarkerSize', 15)
    plot(x(1:end-1,1),x(1:end-1,2), 'k*'),
    
    % The expected information    
    subplot(2,4,5), hold on, title(sprintf('Expected improvement %.2e', min(EIs)))
    pcolor(X,Y,reshape(EI,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'kX', 'MarkerSize', 15)
    plot(x(end,1),x(end,2), 'wX', 'MarkerSize', 15)
    plot(x(1:end-1,1),x(1:end-1,2), 'k*'),
    
    % constraint 1
    subplot(2,4,3), hold on, title(sprintf('constraint 1'))
    pcolor(X,Y,Zc1),shading flat
    plot(x(1:end-1,1),x(1:end-1,2), 'k*'),
    % constraint 2
    subplot(2,4,4), hold on, title(sprintf('constraint 2'))
    pcolor(X,Y,Zc2),shading flat
    plot(x(1:end-1,1),x(1:end-1,2), 'k*'),
    
    % prediction of constraint 1
    subplot(2,4,7), hold on, title(sprintf('prediction for const 1'))
    Efc1(Efc1<const(1,1) | Efc1>const(1,2)) = nan; 
    Efc1(~isnan(Efc1))=1; Efc1 = reshape(Efc1,100,100);
    pcolor(X,Y,reshape(Efc1,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'kX', 'MarkerSize', 15)
    plot(xc1(end,1),xc1(end,2), 'wX', 'MarkerSize', 15)
    plot(xc1(1:end-1,1),xc1(1:end-1,2), 'k*'),
    
    % prediction of constraint 2
    subplot(2,4,8), hold on, title(sprintf('prediction for const 2'))
    Efc2(Efc2<const(2,1) | Efc2>const(2,2)) = nan; 
    Efc2(~isnan(Efc2))=1; Efc2 = reshape(Efc2,100,100);
    pcolor(X,Y,reshape(Efc2,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'kX', 'MarkerSize', 15)
    plot(xc2(end,1),xc2(end,2), 'wX', 'MarkerSize', 15)
    plot(xc2(1:end-1,1),xc2(1:end-1,2), 'k*'),
      
    if length(y)>1
        improv = abs(y(end) - y(end-1));
    end
    i1=i1+1;
    pause
end

