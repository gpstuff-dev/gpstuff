%DEMO_REGRESSION_SPARSE1  Regression problem demonstration for 2-input 
%                         function with sparse Gaussian processes
%
%  Description
%    The problem is the same as in demo_regression1 but now we use
%    sparse Gaussian processes. 
% 
%    The regression problem consist of a data with two input variables
%    and output variable contaminated with Gaussian noise. The model
%    constructed is following:
%
%    The observations y are assumed to satisfy
%
%         y = f + e,    where e ~ N(0, s^2)
%
%    where f is an underlying function, which we are interested in. We
%    place a zero mean Gaussian process prior for f, which implies
%    that at the observed input locations latent values have prior
%
%         f ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is
%    covariance function and th its parameters, hyperparameters.
%
%    Since both likelihood and prior are Gaussian, we obtain a
%    Gaussian marginal likelihood
%
%        p(y|th) = N(0, K + I*s^2).
%    
%   By placing a hyperprior for hyperparameters, p(th), we can find
%   the maximum a posterior (MAP) estimate for them by maximizing
%
%       argmax   log p(y|th) + log p(th).
%         th
%   
%   After finding MAP estimate, we can make predictions for f_new:
%
%       p(f_new | y, th) = N(m, S),
%
%          m = K_nt*(K + I*s^2)^(-1)*y
%          S = K_new - K_nt*(K + I*s^2)^(-1)*K_tn
%   
%      where K_new is the covariance matrix of new f, and K_nt between
%      new f and training f.
%
%   Since we use sparse GPs the full covariance matrix K is replaced
%   either by
%     K_sp                   (a sparse covariance matrix from a compact
%                              suppport covariance function)
%     K_fu*(K_uu)\K_uf + La  (sparse approximation for K, where 
%                             size(K_fu)=[n,m] and La is either 
%                             diagonal, FIC, or block diagonal, PIC, 
%                             or left out, VAR and DTC)
%   
%
%   See (Snelson and Ghahramani, 2006) and (Quinonera-Candela and
%   Rasmussen, 2005) for FIC, PIC and DTC, (Titsias, 2009) for VAR and
%   (Vanhatalo and Vehtari, 2008) for compact suppport (CS) covariance
%   functions.
%
%   The demo is organised in five parts:
%    1) data analysis with CS covariance function
%    2) data analysis with FIC sparse approximation
%    3) data analysis with PIC sparse approximation
%    4) data analysis with VAR sparse approximation
%    5) data analysis with DTC sparse approximation
%
%
%    (We could integrate also over the hyperparameters with, for
%    example, grid integration or MCMC. This is not demonstrated
%    here but it is done exactly the similar way as in
%    demo_regression1.)
%
%  References:
%    Quiñonero-Candela, J. and Rasmussen, C. E. (2005). A unifying
%    view of sparse approximate Gaussian process regression. 
%    Journal of Machine Learning Re- search, 6(3):1939-1959.
%
%    Snelson, E. and Ghahramani, Z. (2006). Sparse Gaussian process
%    using pseudo- inputs. Advances in Neural Information
%    Processing Systems 18.
%
%    Titsias, M. K. (2009). Variational Model Selection for Sparse
%    Gaussian Process Regression. Technical Report, University of
%    Manchester.
%
%    Vanhatalo, J. and Vehtari, A. (2008). Modelling local and
%    global phenomena with sparse Gaussian processes. Proceedings
%    of the 24th Conference on Uncertainty in Artificial
%    Intelligence, pp. 571-578, AUAI Press.
%
%  See also  
%    DEMO_REGRESSION1, DEMO_REGRESSION2, DEMO_REGRESSION_SPARSE2

% Copyright (c) 2008-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

%========================================================
% PART 1 data analysis with CS covariance function gpcf_ppcs2 
%========================================================

% Load the data
S = which('demo_regression_sparse1');
L = strrep(S,'demo_regression_sparse1.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Now 'x' consist of the inputs and 'y' of the output. 
% 'n' and 'nin' are the number of data points and the 
% dimensionality of 'x' (the number of inputs).

% ---------------------------
% --- Construct the model ---
% 
% First create a piece wise polynomial covariance function with ARD and 
% Gaussian noise data structures...
gpcf = gpcf_ppcs2('nin', nin, 'lengthScale', [0.8 0.6], 'magnSigma2', 0.2^2);
lik = lik_gaussian('sigma2', 0.2^2);

pl = prior_t('s2', 0.5);               % a prior structure
pm = prior_sqrtt('s2', 0.3);               % a prior structure
pn = prior_logunif();
gpcf = gpcf_ppcs2(gpcf, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

gp = gp_set('lik', lik, 'cf', {gpcf}, 'jitterSigma2', 1e-8);

% We have now constructed a GP model with gpcf_ppcs2 covariance
% function. This is a compact support function which produces
% sparse covariance matrices if its lenght-scale is short enough. 
% The inference with CS functions is conducted exactly the same way
% as with full support functions.

% For a demo where compact suppor is actually speeding up the
% inference see demo_ppcsCov

% -----------------------------
% --- Conduct the inference ---

% MAP estimate for the hyperparameters using scaled conjugate
% gradient algorithm
% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'optimf',@fminscg,'opt',opt);

% for the last make prections of the underlying function on a dense
% grid and plot it. Below Ef_full is the predictive mean and
% Varf_full the predictive variance.
[xt1,xt2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
xt=[xt1(:) xt2(:)];
Eft_cs = gp_pred(gp, x, y, xt);

% Plot the prediction and data
figure
mesh(xt1, xt2, reshape(Eft_cs,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
axis on;
title('The predicted underlying function and the data points (MAP solution)');

%========================================================
% PART 2 data analysis with FIC sparse approximation
%========================================================

% Here we conduct the same analysis as in part 1, but this time we
% use FIC sparse approximation. The sparse approximations are
% constructed very similarly to full GP. The only difference is
% that we have to define the type of the GP structure differently
% and set the inducing inputs in it.

% First we create the GP data structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
lik = lik_gaussian('sigma2', 0.2^2);
gpcf = gpcf_sexp('lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Next we initialize the inducing inputs and set them in GP
% structure. We have to give a prior for the inducing inputs also,
% if we want to optimize them
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];
gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf', {gpcf}, ...
                'jitterSigma2', 1e-4, 'X_u', X_u)

% -----------------------------
% --- Conduct the inference ---

% Then we can conduct the inference. We can now optimize i) only
% the hyperparameters, ii) both the hyperparameters and the
% inducing inputs, or iii) only the inducing inputs. Which option
% is used is defined by a string that is given to the gp_pak,
% gp_unpak, gp_e and gp_g functions. The strings for the different
% options are: 'covariance' (i), 'covariance+inducing' (ii),
% 'inducing' (iii).
%

% Now you can choose, if you want to optimize only hyperparameters
% or optimize simultaneously hyperparameters and inducing inputs. 
% Note that the inducing inputs are not transformed through
% logarithm when packed

% optimize hyperparameters and inducing inputs
gp_fic = gp_set(gp_fic, 'infer_params', 'covariance+likelihood+inducing');  
% optimize only hyperparameters
%gp_fic = gp_set(gp_fic, 'infer_params', 'covariance+likelihood');           

% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter','Maxiter',50);
% Optimize with the scaled conjugate gradient method
gp_fic=gp_optim(gp_fic,x,y,'optimf',@fminscg,'opt',opt);

% To optimize the hyperparameters and inducing inputs sequentially uncomment the below lines
% $$$ iter = 1
% $$$ e = gp_e(w,gp_fic,x,y)
% $$$ e_old = inf;
% $$$ while iter < 100 & abs(e_old-e) > 1e-3
% $$$     e_old = e;
% $$$     
% $$$     gp_fic = gp_set(gp_fic, 'infer_params', 'covariance+likelihood');  % optimize hyperparameters and inducing inputs
% $$$     gp_fic=gp_optim(gp_fic,x,y,'optimf',@fminscg,'opt',opt);
% $$$     
% $$$     gp_fic = gp_set(gp_fic, 'infer_params', 'inducing');  % optimize hyperparameters and inducing inputs
% $$$     gp_fic=gp_optim(gp_fic,x,y,'optimf',@fminscg,'opt',opt);
% $$$     e = gp_e(w,gp_fic,x,y);
% $$$     iter = iter +1;
% $$$     [iter e]
% $$$ end

% Make the prediction
[xt1,xt2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
xt=[xt1(:) xt2(:)];
Eft_fic = gp_pred(gp_fic, x, y, xt);

% Plot the solution of CS GP and FIC
figure
mesh(xt1, xt2, reshape(Eft_fic,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
plot3(gp_fic.X_u(:,1), gp_fic.X_u(:,2), -3*ones(length(u1(:))), 'rx')
axis on;
title(['The predicted underlying function,   ';
       'data points and inducing inputs (FIC)']);
xlim([-2 2]), ylim([-2 2])

%========================================================
% PART 4 data analysis with PIC approximation
%========================================================

% Now we will use the PIC sparse approximation. The model is constructed 
% in similar way as FIC but now we have to set also the block indeces in 
% the GP structure
 
% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];

% Initialize test points
[xt1,xt2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
xt=[xt1(:) xt2(:)];

% set the data points into clusters. Here we construct two cell arrays. 
%  trindex  contains the block index vectors for training data. That is 
%           x(trindex{i},:) and y(trindex{i},:) belong to the i'th block.
%  tstindex contains the block index vectors for test data. That is test 
%           inputs p(tstindex{i},:) belong to the i'th block.
%
b1 = [-1.7 -0.8 0.1 1 1.9];
mask = zeros(size(x,1),size(x,1));
trindex={}; tstindex={}; 
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));
        trindex{4*(i1-1)+i2} = ind';
        ind2 = 1:size(xt,1);
        ind2 = ind2(: , b1(i1)<=xt(ind2',1) & xt(ind2',1) < b1(i1+1));
        ind2 = ind2(: , b1(i2)<=xt(ind2',2) & xt(ind2',2) < b1(i2+1));
        tstindex{4*(i1-1)+i2} = ind2';
    end
end

% Create the PIC GP data structure and set the inducing inputs and block indeces
gpcf = gpcf_sexp('lengthScale', [1 1], 'magnSigma2', 0.2^2);
lik = lik_gaussian('sigma2', 0.2^2);

gp_pic = gp_set('type', 'PIC', 'lik', lik, 'cf', {gpcf}, ...
                'X_u', X_u, 'tr_index', trindex, 'jitterSigma2', 1e-4);

% -----------------------------
% --- Conduct the inference ---

% MAP estimate for the hyperparameters and inducing inputs using
% scaled conjugate gradient algorithm

% Now you can choose, if you want to optimize only hyperparameters
% or optimize simultaneously hyperparameters and inducing inputs. 
% Note that the inducing inputs are not transformed through
% logarithm when packed

% optimize hyperparameters and inducing inputs
gp_pic = gp_set(gp_pic, 'infer_params', 'covariance+inducing');  
% optimize only hyperparameters
%gp_pic = gp_set(gp_pic, 'infer_params', 'covariance+likelihood');           

% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter','Maxiter',50);
% Optimize with the scaled conjugate gradient method
gp_pic=gp_optim(gp_pic,x,y,'optimf',@fminscg,'opt',opt);

% Make the prediction. 
% Here it should be noticed that since we are using PIC we have to
% give the block indeces of test cases (tstindex) for gp_pred.
Eft_pic = gp_pred(gp_pic, x, y, xt, 'tstind', tstindex);

% Plot the solution of CS GP, FIC, and PIC
figure
mesh(xt1, xt2, reshape(Eft_pic,37,37));
hold on
% plot the data points in each block with different colors and marks
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot3(x(trindex{i},1),x(trindex{i},2), y(trindex{i}),col{i})
end
for i=1:16
    plot3(xt(tstindex{i}(1:2:end),1),xt(tstindex{i}(1:2:end),2),-3.5*ones(length(tstindex{i}(1:2:end))),col{i})
end
plot3(gp_pic.X_u(:,1), gp_pic.X_u(:,2), -3*ones(length(u1(:))), 'rx')
axis on;
title(['The predicted underlying function, data points (colors ';
       'distinguish the blocks) and inducing inputs (PIC)      ']);
xlim([-2 2]), ylim([-2 2])


%========================================================
% PART 5 data analysis with VAR sparse approximation
%========================================================

% Now we will use the variational sparse approximation.

% First we create the GP data structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
lik = lik_gaussian('sigma2', 0.2^2);
gpcf = gpcf_sexp('lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Next we initialize the inducing inputs and set them in GP
% structure. We have to give a prior for the inducing inputs also,
% if we want to optimize them
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', {gpcf}, ...
                'jitterSigma2', 1e-4, 'X_u', X_u);

% -----------------------------
% --- Conduct the inference ---

% Then we can conduct the inference. We can now optimize i) only
% the hyperparameters, ii) both the hyperparameters and the
% inducing inputs, or iii) only the inducing inputs. Which option
% is used is defined by a string that is given to the gp_pak,
% gp_unpak, gp_e and gp_g functions. The strings for the different
% options are: 'covariance+likelihood' (i), 'covariance+inducing' (ii),
% 'inducing' (iii).
%

% Now you can choose, if you want to optimize only hyperparameters
% or optimize simultaneously hyperparameters and inducing inputs. 
% Note that the inducing inputs are not transformed through
% logarithm when packed

% optimize hyperparameters and inducing inputs
gp_var = gp_set(gp_var, 'infer_params', 'covariance+inducing');  
% optimize only hyperparameters
%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');           

% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter','Maxiter',50);
% Optimize with the scaled conjugate gradient method
gp_var=gp_optim(gp_var,x,y,'optimf',@fminscg,'opt',opt);

% To optimize the hyperparameters and inducing inputs sequentially uncomment the below lines
% $$$ iter = 1
% $$$ e = gp_e(w,gp_var,x,y)
% $$$ e_old = inf;
% $$$ while iter < 100 & abs(e_old-e) > 1e-3
% $$$     e_old = e;
% $$$     
% $$$     gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');  % optimize hyperparameters and inducing inputs
% $$$     gp_var=gp_optim(gp_var,x,y,'optimf',@fminscg,'opt',opt);
% $$$     gp_var = gp_set(gp_var, 'infer_params', 'inducing');  % optimize hyperparameters and inducing inputs
% $$$     gp_var=gp_optim(gp_var,x,y,'optimf',@fminscg,'opt',opt);
% $$$     e = gp_e(w,gp_var,x,y);
% $$$     iter = iter +1;
% $$$     [iter e]
% $$$ end


% Make the prediction
[xt1,xt2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
xt=[xt1(:) xt2(:)];
Eft_var = gp_pred(gp_var, x, y, xt);

% Plot the solution of CS GP and VAR
figure
mesh(xt1, xt2, reshape(Eft_var,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
plot3(gp_var.X_u(:,1), gp_var.X_u(:,2), -3*ones(length(u1(:))), 'rx')
axis on;
title(['The predicted underlying function,   ';
       'data points and inducing inputs (VAR)']);
xlim([-2 2]), ylim([-2 2])



%========================================================
% PART 6 data analysis with DTC sparse approximation
%========================================================

% Now we will use the DTC sparse approximation.

% First we create the GP data structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
lik = lik_gaussian('sigma2', 0.2^2);
gpcf = gpcf_sexp('lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Next we initialize the inducing inputs and set them in GP structure. 
% We have to give a prior for the inducing inputs also, if we want to optimize 
% them
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];
gp_dtc = gp_set('type', 'DTC', 'lik', lik, 'cf', {gpcf}, ...
                'jitterSigma2', 0.001, 'X_u', X_u);

% -----------------------------
% --- Conduct the inference ---

% Then we can conduct the inference. We can now optimize i) only
% the hyperparameters, ii) both the hyperparameters and the
% inducing inputs, or iii) only the inducing inputs. Which option
% is used is defined by a string that is given to the gp_pak,
% gp_unpak, gp_e and gp_g functions. The strings for the different
% options are: 'covariance+likelihood' (i), 'covariance+inducing' (ii),
% 'inducing' (iii).
%

% Now you can choose, if you want to optimize only hyperparameters
% or optimize simultaneously hyperparameters and inducing inputs. 
% Note that the inducing inputs are not transformed through
% logarithm when packed

% optimize hyperparameters and inducing inputs
gp_dtc = gp_set(gp_dtc, 'infer_params', 'covariance+inducing');  
% optimize only hyperparameters
%gp_dtc = gp_set(gp_dtc, 'infer_params', 'covariance+likelihood');           

% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter','Maxiter',50);
% Optimize with the scaled conjugate gradient method
gp_dtc=gp_optim(gp_dtc,x,y,'optimf',@fminscg,'opt',opt);

% To optimize the hyperparameters and inducing inputs sequentially uncomment the below lines
% $$$ iter = 1
% $$$ e = gp_e(w,gp_dtc,x,y)
% $$$ e_old = inf;
% $$$ while iter < 100 & abs(e_old-e) > 1e-3
% $$$     e_old = e;
% $$$     
% $$$     gp_dtc = gp_set(gp_dtc, 'infer_params', 'covariance+likelihood');  % optimize hyperparameters and inducing inputs
% $$$     gp_dtc=gp_optim(gp_dtc,x,y,'optimf',@fminscg,'opt',opt);
% $$$     gp_dtc = gp_set(gp_dtc, 'infer_params', 'inducing');  % optimize hyperparameters and inducing inputs
% $$$     gp_dtc=gp_optim(gp_dtc,x,y,'optimf',@fminscg,'opt',opt);
% $$$     e = gp_e(w,gp_dtc,x,y);
% $$$     iter = iter +1;
% $$$     [iter e]
% $$$ end


% Make the prediction
[xt1,xt2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
xt=[xt1(:) xt2(:)];
Eft_dtc = gp_pred(gp_dtc, x, y, xt);

% Plot the solution of CS GP and DTC
figure
mesh(xt1, xt2, reshape(Eft_dtc,37,37));
hold on
plot3(x(:,1), x(:,2), y, '*')
plot3(gp_dtc.X_u(:,1), gp_dtc.X_u(:,2), -3*ones(length(u1(:))), 'rx')
axis on;
title(['The predicted underlying function,   ';
       'data points and inducing inputs (DTC)']);
xlim([-2 2]), ylim([-2 2])
