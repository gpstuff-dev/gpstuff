addpath ../../matlab/GPstuff-4.1

startup

load minimal_demo.mat

[ntp,d] = size(x_train);

likfunc       = lik_probit();
covfunc       = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1);
model         = gp_set('lik', likfunc, 'cf', covfunc, 'jitterSigma2', 1e-9);
model         = gp_set(model, 'latent_method', 'EP');
opt           = optimset('TolFun',  1e-6, ...
                         'TolX',    1e-6, ...
                         'MaxIter', 100, ...
                         'Display', 'iter');
[model,logev] = gp_optim(model, x_train, t_train, 'opt', opt);
model.logev   = logev;

