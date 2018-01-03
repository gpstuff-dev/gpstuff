% DEMO_MGP ─ A practical example of the multivariate Gaussian process model 
%            with different probabilistic models
%
% Description :
%
%   For many biological real-case scenarios two (or more) 
%   animal populations can be strongly interwoven. The way in 
%   which species dependent on each other are manifold. For instance,
%   predator-prey relationship, competition, symbiosis, etc.
%
%   It is fundamental to develop mechanistic understanding on the
%   underlying processes that governs the natural growth of the population.
%   However this goal is sometimes difficult and often the models are too
%   complex to derive from first principles. Nowadays, there are many 
%   mechanistic models which are developed from first principles and that they 
%   are able to describe the population dynamics in a very accurate manner
%   (system of ODEs or PDEs). Nevertheless, in many practical applications
%   it is easier to acquire data from the realization of such underlying biological 
%   processes and infer their most probable form. Therefore Bayesian
%   nonparametric models become useful tools.
%
%   The example presented here uses a classical data-set from two animal
%   populations in Canada. The Lynx (Lynx canadensis) and the snowshoe hare 
%   (Lepus americanus). The lynx population rise and fall according to the
%   variations in the populations of snowshoe hares over time. That is it, when 
%   hares are abundant, lynx populations expand, and when the density of hares is 
%   low, the population of lynx shrinks (there are many reasons why).
%   This is know as the predator-prey interaction.
%
%   In this example we want exemplify the joint modelling of the two species
%   population (abundance) aforementioned when data is missing in different
%   time intervals (for both species) and how one species can inform the 
%   abundance of each another.
%
%   We assume that the number of lynx and hares follow the 
%   negative-binomial distribution given unknown latent values (function
%   values of some unspecified function form for the regression), i.e., 
%
%       N1(t)|f1(t) ~ Neg-Binomial(exp(f1(t)), r1)  (hare)
%       N2(t)|f2(t) ~ Neg-Binomial(exp(f2(t)), r2)  (lynx)
%
%   and that N1(t)|f1(t) is independent of N2(t)|f2(t). The expression exp(f1(t)) 
%   is the expected value of N1 given f1 and exp(f2(t)) is the expected value 
%   of N2 given f2. The correlation is now introduced through the multivariate
%   Gaussian process prior assuming the linear model of coregionalization, i.e.,
%
%       [f1(t), f2(t')] ~ MGP(0, K)
% 
%   where K is a full covariance matrix formed by covariance function 
%
%       k(fj(t), fj'(t')) = Sig_(c = 1, 2) u_c(j, j') k_c(t, t')
%  
%   with u_c(j, j') the (j, j') entry of the matrix U_c = Lc Lc^T, where Lc
%   is cth column of the Cholesky decomposition of the coregionalization
%   matrix (covariance matrix) {Sig}j,j' = sig_j sig_j' rho_{j, j'}. We use the
%   analytical approximation expectation-propagation to carry out the inference
%   over the latent values f1, f2 and then calcule the unconditional expected values
%   of N1 and N2, i.e, E[N1(t)|Y1 = y1, Y2 = y2] and E[N2(t)|Y1 = y1, Y2 = y2].
%
% Additional references :
%
%   Murray, J, D (2002). Mathematical biology. Third Edition. Springer
%    Series.
%
%   Gelfand et al. (2004). Nonstationary multivariate process modelling 
%     through spatially varying coregionalization.
%
%   Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
%
%   Bernardo, J and Smith, A (2008). Bayesian Theory. Wiley series in
%    probability and statistics
%
% -------------- Marcelo Hartmann

%%
% ------- data analysis with the multivariate Gaussian process model

% download the data
S = which('demo_multivariategp');
L = strrep(S, 'demo_multivariategp.m', 'demodata/predpreydata.txt');
data = importdata(L);

% colors;
col = [0 0 1; 1 0 0];

% all time points (yearly)
t = data.data(:, 1);

% full data
yy = [round(data.data(:, 2)); round(data.data(:, 3))];

% take sparse data (you can change the years);
yr1 = [1870, 1900];
% yr1 = [1846,  1935]; one observation
indt1 = ~logical((t >= yr1(1)) .* (t <= yr1(2)));
t1 = t(indt1);

yr2 = [1850, 1870];
% yr2 = [1845, 1934]; one observation
indt2 = ~logical((t >= yr2(1)) .* (t <= yr2(2)));
t2 = t(indt2);

% hare (thousands)
y1 = round(data.data(indt1, 2));

% lynx (thousands)
y2 = round(data.data(indt2, 3));

% predator-prey data;
y = [y1; y2];

% create species markers
c = [ones(size(t1)); 2*ones(size(t2))];

% data gp-format - species markers in the last column
x = [[t1; t2] c];
z = [ones(size(x, 1), 1) c];

% likelihood structure for each species (the likelihood can also be distinct)
lik1 = lik_negbin;
lik2 = lik_negbin;
likS = {lik1 lik2};
lik = lik_liks('likelihoods', likS, 'classVariables', 2);

% correlation function for each species (the correlation functions can also be distinct)
k1 = gpcf_sexp('magnSigma2', 1, 'magnSigma2_prior', prior_fixed, 'selectedVariables', 1);
k2 = gpcf_sexp('magnSigma2', 1, 'magnSigma2_prior', prior_fixed, 'selectedVariables', 1);

% the linear model of coregionalization (multivariate Gaussian process model)
k = gpcf_covar('numberClass', 2, 'classVariables', 2, ...
    'R_prior', prior_corrunif('nu', 3), 'corrFun', {k1 k2});

% set gp structure
% for the EP approximation
gp = gp_set('lik', lik, 'cf', k, 'latent_method', 'EP');

% For the Laplace approximation
%gp = gp_set('lik', lik, 'cf', k, 'latent_method', 'Laplace'); 
 
% optimzation options
opt = optimset('TolFun', 1e-3, 'TolX', 1e-5, 'Display', 'iter');   

% random initialization (this is important ...)
% wini = 2.* randn(size(gp_pak(gp))); gp = gp_unpak(gp, wini);

% map ─ type-II maximum likelihood
gp = gp_optim(gp, x, y, 'z', z, 'opt', opt);

% check the gradient ...
% gp_g(gp_pak(gp), gp, x, y, 'z', z)';

% take the parameters, hyperparameters estimates
[th ss] = gp_pak(gp);

% correlation matrix estimate
%rho = k.fh.RealToRho(th(end - 2 - 2), 1, [])
Corr = gp.cf{1}.fh.sigma(gp.cf{1}, 'corr');
Corr{1}

% prediction
np = 200;
xp = repmat(linspace(min(x(:, 1)), max(x(:, 1)), np)', 2, 1);
xp = [xp repelem((1:2), np)'];
zp = ones(size(xp, 1), 1);

% leave-one-out
[~, ~, lpyt] = gp_loopred(gp, x, y,'z', z); 

% prediction for new observations
[Ef, Varf, ~, Ey, Vary] = gp_pred(gp, x, y, xp, 'z', z, 'zt', [zp xp(:, 2)]); 

% --- visualize predictions

figure; hold on;

subplt = subplot(2, 1, 1); hold on
subplt.XLim = [min(xp(:, 1)), max(xp(:, 1))];
psubplt = get(subplt, 'pos');
psubplt(4) = psubplt(4) + 0.03;
psubplt(3) = psubplt(3) + 0.04;
psubplt(1) = psubplt(1) - 0.04;
set(subplt, 'pos', psubplt);

for j = 1:2
    indj = xp(:, 2) == j; 
    indX = xp(indj, 1);
    
    pl(j) = plot(xp(indj, 1), Ey(indj), 'color', col(j, :), 'LineWidth', 2);
 end

% visualize observed (training) predator-pray data
pl(3) = plot(t1(t1 <= yr1(1)), y1(t1 <= yr1(1)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(1, :)); 
plot(t1(t1 >= yr1(2)), y1(t1 >= yr1(1)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(1, :)); 
pl(4) = plot(t2(t2 <= yr2(1)), y2(t2 <= yr2(1)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(2, :)); 
plot(t2(t2 >= yr2(2)), y2(t2 >= yr2(2)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(2, :)); 

xlabel('years'); ylabel('Population size'); 
title('multivariate Gaussian process model');

legend(pl, 'E[N_1(t_*)|N_1 = n_1, N_2 = n_2]', 'E[N_2(t_*)|N_1 = n_1, N_2 = n_2]', ...
    'hare-data', 'lynx-data', 'Location', 'northeast'); 

%%
% -------- data analysis with independent Gaussian process model

EfI = {}; VarfI = {}; EyI = {}; VaryI = {};
gpI = {};

subplt = subplot(2, 1, 2); hold on
subplt.XLim = [min(xp(:, 1)), max(xp(:, 1))];
psubplt = get(subplt, 'pos');
psubplt(4) = psubplt(4) + 0.05;
psubplt(3) = psubplt(3) + 0.04;
psubplt(1) = psubplt(1) - 0.03;
set(subplt, 'pos', psubplt);

for j = 1:2
    inddj = x(:, 2) == j; 
    cf = gpcf_sexp('selectedVariables', 1);
       
    gpI{j} = gp_set('lik', likS{j}, 'cf', cf);
    gpI{j} = gp_optim(gpI{j}, x(inddj, :), y(inddj), 'opt', opt);
    
    indj = xp(:, 2) == j; 
    indX = xp(indj, 1);
    
    [EfI{j}, VarfI{j}, lpyt, EyI{j}, VaryI{j}] = ...
        gp_pred(gpI{j}, x(inddj, :), y(inddj), xp(indj, :));
    
    pl(j) = plot(xp(indj, 1), EyI{j}, 'color', col(j, :), 'LineWidth', 1.7); 
end

% visualize observed (training) predator-pray data
pl(3) = plot(t1(t1 <= yr1(1)), y1(t1 <= yr1(1)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(1, :)); 
plot(t1(t1 >= yr1(2)), y1(t1 >= yr1(1)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(1, :)); 
pl(4) = plot(t2(t2 <= yr2(1)), y2(t2 <= yr2(1)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(2, :)); 
plot(t2(t2 >= yr2(2)), y2(t2 >= yr2(2)), '.', ...
    'MarkerSize', 23, 'lineWidth', 2, 'color', col(2, :)); 

xlabel('years'); ylabel('Population size'); 
title('independent Gaussian process models');

legend(pl, 'E[N1(t)|Y1 = y1, Y2 = y2]', 'E[N2(t)|Y1 = y1, Y2 = y2]', ...
    'hare-data', 'lynx-data', 'Location', 'northeast'); 

%%
% ------ all measurements (compare the difference ...)

figure; hold on;

for j = 1:2
    indj = x(:, 2) == j;
    plot(x(indj, 1), y(indj), '.', 'MarkerSize', 13, 'color', col(j, :))
    pd(j) =  plot(t, yy((j - 1) * 91 + [1:91]), '.-', ... 
        'color', col(j, :), 'MarkerSize', 23, 'LineWidth', 2);
end    

xlabel('years'); ylabel('Population size'); 
title('full data');

legend(pd, 'hare-data', 'lynx-data', 'Location', 'northeast'); 

