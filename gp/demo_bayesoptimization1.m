%DEMO_BAYESIANOPTIMIZATION1  A demonstration program for Bayesian
%                            optimization in 1 dimension
%
% The set of BO demos
%  Part 1:  this file
%  One dimensional example 
%
%  Part 2:  see demo_bayesoptimization2
%  Two dimensional example 
%
%  Part 3:  see demo_bayesoptimization3
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

%%  Part 1:
%  One dimensional example 
% For testing purposes:
stack = dbstack;
if (~isempty(stack) && (strcmp(stack(end).name, 'runtestset') || strcmp(stack(end).name, 'runtests'))) test = 1; else test = 0; end;

% Construct a function to be optimized
xl = linspace(0,10,100)';
fx = @(x) 0.6*x -0.1*x.^2 + sin(2*x);

% construct GP
cfse = gpcf_sexp('lengthScale',1,'magnSigma2',1,'magnSigma2_prior',prior_sqrtt('s2',10^2));
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
rng(3)
x = 10*rand;
y = fx(x);

figure, % figure for visualization
i1 = 1;
maxiter = 15;
improv = inf;   % improvement between two successive query points
while i1 < maxiter && improv>1e-6
%while i1 < maxiter

    % Train the GP model for objective function and calculate variables
    % that are needed when calculating the Expected improvement
    % (Acquisition function) 
    if i1>1
        gp = gp_optim(gp,x,y);
    end
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
    % don't crash into suboptimal mode
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
    %plot(xl,fx(xl))
    box on
    plot(xl,fx(xl),'r')
    % The function evaluations so far
    plot(x(1:end-1),y(1:end-1), 'ko')
    % The new sample location
    plot(x(end),y(end), 'ro')
    % the posterior of the function
    plot(xl,Ef, 'k')
    plot(xl,Ef + 2*sqrt(Varf), 'k--')
    plot(xl,Ef - 2*sqrt(Varf), 'k--')
    legend('objective function', 'function evaluations', 'next query point', 'GP mean', 'GP 95% interval','location','southwest')
    % The expected information    
    subplot(2,1,2)
    plot(xl,EI, 'r'), hold on
    plot(x(end),0, 'r*')
    plot(x(end)*[1 1],ylim, 'r--')
    title('acquisition function')

       
    improv = abs(y(end) - y(end-1));
    i1=i1+1;
    
    if test == 0
        pause
    end
end
%subplot(2,1,1)
%plot(xl,fx(xl),'r')