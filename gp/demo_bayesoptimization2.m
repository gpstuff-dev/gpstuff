%DEMO_BAYESIANOPTIMIZATION 2 A demonstration program for Bayesian
%                            optimization in two dimensions
%
% The set of BO demos
%  Part 1: see demo_bayesoptimization1
%  One dimensional example 
%
%  Part 2: this file
%  Two dimensional example 
%
%  Part 3: see demo_bayesoptimization3
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
%  Copyright (c) 2015-2017 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

%%  Part 2:
%  Two dimensional example 
% For testing purposes:
stack = dbstack;
if (~isempty(stack) && (strcmp(stack(end).name, 'runtestset') || strcmp(stack(end).name, 'runtests'))) test = 1; else test = 0; end;

rng(3)
% The objective function
fx = @(x) -log( (mvnpdf([x(:,1) x(:,2)],[-1.5 -2.5], [1 0.3; 0.3 1]) + 0.3*mvnpdf([x(:,1) x(:,2)],[2 3], [3 0.5; 0.5 4])).*...
    mvnpdf([x(:,1) x(:,2)],[0 0], [100 0; 0 100])) ./15 -1;

% Help variables for visualization
lb=-5;
ub=5;
[X,Y] = meshgrid(linspace(lb,ub,100),linspace(lb,ub,100));
xl = [X(:) Y(:)];
Z = reshape(fx(xl),100,100);

% construct GP to model the function
cfc = gpcf_constant('constSigma2',10,'constSigma2_prior', prior_fixed);
cfl = gpcf_linear('coeffSigma2', .01, 'coeffSigma2_prior', prior_sqrtt()); 
cfl2 = gpcf_squared('coeffSigma2', .01, 'coeffSigma2_prior', prior_sqrtt(), 'interactions', 'on');
cfse = gpcf_sexp('lengthScale',[5 5],'lengthScale_prior',prior_t('s2',4),'magnSigma2',.1,'magnSigma2_prior',prior_sqrtt('s2',10^2));
lik = lik_gaussian('sigma2', 0.001, 'sigma2_prior', prior_fixed);
gp = gp_set('cf', {cfc, cfl, cfl2, cfse}, 'lik', lik);

% ----- conduct Bayesian optimization -----

% Set the options for optimizer of the acquisition function
optimf = @fmincon;
optdefault=struct('GradObj','on','LargeScale','off','Algorithm','trust-region-reflective','TolFun',1e-9,'TolX',1e-6);
opt=optimset(optdefault);
lb=[-5 -5];     % lower bound of the input space
ub=[5 5];   % upper bound of the input space

% draw initial points
x = [-4 -4;-4 4;4 -4;4 4;0 0];
y = fx(x);

figure, % figure for visualization
i1 = 1;
maxiter = 20;
improv = inf;   % improvement between two successive query points
x_new=[];
while i1 < maxiter && improv>1e-6
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    if i1>1
        gp = gp_optim(gp,x,y);
        [gpia,pth,th]=gp_ia(gp,x,y);
        gp = gp_unpak(gp,sum(bsxfun(@times,pth,th)));
    end
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
    % don't crash into suboptimal mode of acquisition function
    if mod(i1,5)==0  % Do just exploration by finding the maimum variance location
        fh_eg = @(x_new) expectedvariance_eg(x_new, gp, x, [], invC);
    else
        fh_eg = @(x_new) expectedimprovement_eg(x_new, gp, x, a, invC, fmin);
    end
    indbest = find(y == fmin);
    nstarts = 20;
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
    % Plot the objective function
    subplot(2,2,1),hold on, title('Objective, query points')
    box on
    pcolor(X,Y,Z),shading flat
    clim = caxis;
    l1=plot(x(1:end-1,1),x(1:end-1,2), 'rx', 'MarkerSize', 10);
    %plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10)
    l2=plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    l3=plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    legend([l1,l2,l3], {'function evaluation points','local modes of acquisition function','The next query point'})
    % Plot the posterior mean of the GP model for the objective function
    subplot(2,2,2),hold on, title(sprintf('GP prediction, mean, iter: %d',i1))
    box on
    pcolor(X,Y,reshape(Ef,100,100)),shading flat
    caxis(clim)
    % Plot the posterior variance of GP model
    subplot(2,2,4),hold on, title('GP prediction, variance')
    box on
    pcolor(X,Y,reshape(Varf,100,100)),shading flat
    l2=plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    l3=plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);
    % Plot the expected improvement 
    subplot(2,2,3), hold on, title(sprintf('Expected improvement %.2e', min(EIs)))
    box on
    pcolor(X,Y,reshape(EI,100,100)),shading flat
    plot(xnews(:,1),xnews(:,2), 'ro', 'MarkerSize', 10);
    plot(x(end,1),x(end,2), 'ro', 'MarkerSize', 10, 'linewidth', 3);

       
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    
    if test == 0
        pause
    end
end

