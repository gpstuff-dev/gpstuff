% TODO!
% - tulosten oikeellisuuden tarkistus (nyt tarkistetaan vain, että funktioissa ei bugeja) 
% - metriikat
% - IA:n testaus regressiossa ja muilla likelihoodeilla

% Initialize the statistics parameters:
warnings = '';
numwarn = 0;

% =========================== 
% Regression models
% ===========================

S = which('test_package');
L = strrep(S,'test_package.m','demos/dat.1');
data=load(L);
data = data(1:2:end,:);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

%----------------------------
% check covariance functions
%----------------------------

fprintf(' \n ================================= \n \n Checking the covariance functions \n \n ================================= \n ')

covfunc = {'gpcf_sexp' 'gpcf_exp' 'gpcf_matern32' 'gpcf_matern52' ...
           'gpcf_ppcs0' 'gpcf_ppcs1' 'gpcf_ppcs2' 'gpcf_ppcs3' 'gpcf_neuralnetwork'...
           'gpcf_dotproduct' 'gpcf_prod'};

gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.2^2);
pl = prior_logunif('init');
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigma2_prior', pl);
        
for i = 1:length(covfunc)
    
    switch covfunc{i}
      case 'gpcf_neuralnetwork'
        gpcf1 = feval(covfunc{i}, 'init', nin);
        
        % Set prior
        pl = prior_logunif('init');
        gpcf1 = gpcf_neuralnetwork('set', gpcf1, 'weightSigma2_prior', pl);
        gpcf1 = gpcf_neuralnetwork('set', gpcf1, 'biasSigma2_prior', pl);
      case 'gpcf_dotproduct'
        gpcf1 = feval(covfunc{i}, 'init', nin);

        % Set prior
        pl = prior_logunif('init');
        gpcf1 = gpcf_dotproduct('set', gpcf1, 'coeffSigma2_prior', pl);
        gpcf1 = gpcf_dotproduct('set', gpcf1, 'constSigma2_prior', pl);
      case 'gpcf_prod'
        gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', 1+0.01*randn(1,nin), 'magnSigma2', 0.5);
        gpcf4 = gpcf_exp('init', 'lengthScale', 1+0.01*randn(1,nin), 'magnSigma2', 0.2^2);

        % Set prior
        pl = prior_logunif('init');
        gpcf3 = gpcf_ppcs2('set', gpcf3, 'lengthScale_prior', pl);
        gpcf3 = gpcf_ppcs2('set', gpcf3, 'magnSigma2_prior', pl);
        gpcf4 = gpcf_exp('set', gpcf4, 'lengthScale_prior', pl);
        gpcf4 = gpcf_exp('set', gpcf4, 'magnSigma2_prior', pl);

        gpcf1 = gpcf_prod('init', 'functions', {gpcf3, gpcf4});
      otherwise
        gpcf1 = feval(covfunc{i}, 'init', 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
        
        % Set prior
        pl = prior_logunif('init');
        gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl);
        gpcf1 = gpcf_sexp('set', gpcf1, 'magnSigma2_prior', pl);
    end
           
    gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
    
    w=gp_pak(gp);  % pack the hyperparameters into one vector
    fe=str2fun('gp_e');     % create a function handle to negative log posterior
    fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
    
    % set the options for scg2
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    
    % do the optimization
    w=scg2(fe, w, opt, fg, gp, x, y);
    
    % Set the optimized hyperparameter values back to the gp structure
    gp=gp_unpak(gp,w);

    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 30;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.07;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    hmc2('state', sum(100*clock));
    
    % Do the sampling (this takes approximately 5-10 minutes)    
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);

    % After sampling we delete the burn-in and thin the sample chain
    rfull = thin(rfull, 10, 2);
    
    if sqrt(mean((y - gp_pred(gp, x, y, x)).^2)) > 0.186*2
        warnings = sprintf([warnings '\n * Check the optimization of hyper-parameters of ' covfunc{i} ' with GP regression']);       
        numwarn = numwarn + 1;
    end
        
    if sqrt(mean((y - mean(mc_pred(rfull, x, y, x),2) ).^2)) > 0.186*2
        warnings = sprintf([warnings '\n * Check the MCMC sampling of hyper-parameters of ' covfunc{i} ' with GP regression']);
        numwarn = numwarn + 1;
    end

    delta = gradcheck(w, @gp_e, @gp_g, gp, x, y);
    
        
    switch covfunc{i}
      case 'gpcf_neuralnetwork'
        gpcf1 = gpcf_neuralnetwork('set', gpcf1, 'weightSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf1 = gpcf_neuralnetwork('set', gpcf1, 'weightSigma2_prior', pl, 'biasSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf1 = gpcf_neuralnetwork('set', gpcf1, 'weightSigma2_prior', [], 'biasSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
      case 'gpcf_dotproduct'
        gpcf1 = gpcf_dotproduct('set', gpcf1, 'constSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf1 = gpcf_dotproduct('set', gpcf1, 'constSigma2_prior', pl, 'coeffSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf1 = gpcf_dotproduct('set', gpcf1, 'constSigma2_prior', [], 'coeffSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y, 'hyper')];
      case 'gpcf_prod'
        gpcf4 = gpcf_exp('set', gpcf1, 'lengthScale_prior', []);
        gpcf1 = gpcf_prod('init', 'functions', {gpcf3, gpcf4});
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf4 = gpcf_exp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', []);
        gpcf1 = gpcf_prod('init', 'functions', {gpcf3, gpcf4});
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf4 = gpcf_exp('set', gpcf1, 'lengthScale_prior', [], 'magnSigma2_prior', []);
        gpcf1 = gpcf_prod('init', 'functions', {gpcf3, gpcf4});
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
      otherwise
        gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
        
        gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', [], 'magnSigma2_prior', []);
        gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
        w=gp_pak(gp);
        delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y)];
    end
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters of ' covfunc{i}]);
        warning([' Check the gradients of ' covfunc{i}]);
        numwarn = numwarn + 1;
    end

end

gpcf1 = gpcf_sexp('init', 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigmas2_prior', []);
gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001);
w=gp_pak(gp);
delta = gradcheck(w, @gp_e, @gp_g, gp, x, y);


%----------------------------
% check sparse approximations
%----------------------------

fprintf(' \n ================================= \n \n Checking the sparse approximations \n \n ================================= \n ')

sparse = {'FIC' 'PIC' 'CS+FIC'};

gpcf1 = gpcf_sexp('init', 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', 'noiseSigmas2', 0.2^2);
gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Set prior
pl = prior_logunif('init');
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigmas2_prior', pl);
gpcf3 = gpcf_ppcs2('set', gpcf3, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);

% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.8,1.8,7),linspace(-1.8,1.8,7));
X_u = [u1(:) u2(:)];

% set the data points into clusters
[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];
b1 = [-1.7 -0.8 0.1 1 1.9];
mask = zeros(size(x,1),size(x,1));
trindex={}; 
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));
        trindex{4*(i1-1)+i2} = ind';
    end
end

for i = 1:length(sparse)
    
    switch sparse{i}
      case 'FIC'
        gp = gp_init('init', 'FIC', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u, 'Xu_prior', prior_unif('init'));
        tstindex = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex}, 'Xu_prior', prior_unif('init'));
        tstindex = trindex;
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u, 'Xu_prior', prior_unif('init'))
        tstindex = 1:n;
    end
    
    param = 'covariance+inducing';
    w=gp_pak(gp, param);  % pack the hyperparameters into one vector
    fe=str2fun('gp_e');     % create a function handle to negative log posterior
    fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
    
    % set the options for scg2
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    
    % do the optimization
    w=scg2(fe, w, opt, fg, gp, x, y, param);
    
    % Set the optimized hyperparameter values back to the gp structure
    gp=gp_unpak(gp,w, param);

    delta = gradcheck(w, @gp_e, @gp_g, gp, x, y, param);
    
    % check that gradients are OK
    if delta>0.01
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters with ' sparse{i} ' sparse approximation']);
        numwarn = numwarn + 1;
    end

    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 30;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.03;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    hmc2('state', sum(100*clock));
    
    % Do the sampling (this takes approximately 5-10 minutes)    
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);

    % After sampling we delete the burn-in and thin the sample chain
    switch sparse{i}
      case 'PIC'
        rfull = rmfield(rfull, 'tr_index');
        rfull = thin(rfull, 10, 2);
        rfull.tr_index = trindex;
      otherwise
        rfull = thin(rfull, 10, 2);
    end

    if sqrt(mean((y - gp_pred(gp, x, y, x, [], tstindex)).^2)) > 0.186*2
        warnings = sprintf([warnings '\n * Check the optimization of hyper-parameters of ' covfunc{i} ' with GP regression']);
        numwarn = numwarn + 1;
    end
        
    if sqrt(mean((y - mean(mc_pred(rfull, x, y, x, [], tstindex),2) ).^2)) > 0.186*2
        warnings = sprintf([warnings '\n * Check the MCMC sampling of hyper-parameters of ' covfunc{i} ' with GP regression']);
        numwarn = numwarn + 1;
    end
    
end

%----------------------------
% Check the additive model and 
% gp_pred and mc_pred
%----------------------------

fprintf(' \n ================================= \n \n Checking the additive model and gp_pred \n \n =================================\n ')

gpcf1 = gpcf_sexp('init', 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.2^2);
gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Set prior
pl = prior_logunif('init');
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigma2_prior', pl);
gpcf3 = gpcf_ppcs2('set', gpcf3, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);

% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.8,1.8,7),linspace(-1.8,1.8,7));
X_u = [u1(:) u2(:)];

% set the data points into clusters
[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];
b1 = [-1.8 -0.8 0.1 1 1.9];
mask = zeros(size(x,1),size(x,1));
trindex={}; testindex={};
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));
        trindex{4*(i1-1)+i2} = ind';
        ind2 = 1:size(p,1);
        ind2 = ind2(: , b1(i1)<=p(ind2',1) & p(ind2',1) < b1(i1+1));
        ind2 = ind2(: , b1(i2)<=p(ind2',2) & p(ind2',2) < b1(i2+1));
        testindex{4*(i1-1)+i2} = ind2';
    end
end
p2 = [x ; p(1:10,:)];
models = {'FULL' 'FIC' 'PIC' 'CS+FIC'};

for i=1:length(models)
    switch models{i}
      case 'FULL' 
        gp = gp_init('init', 'FULL', 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        gp = gp_init('init', 'FIC', 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end
    
    w=gp_pak(gp, 'covariance');  % pack the hyperparameters into one vector
    fe=str2fun('gp_e');     % create a function handle to negative log posterior
    fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
    
    % set the options for scg2
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    
    % do the optimization
    w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');

    % --- Check predictions ---
    [Efa, Varfa] = gp_pred(gp, x, y, p, [], tstindex);
    [Ef1, Varf1] = gp_pred(gp, x, y, p, [1], tstindex);
    [Ef2, Varf2] = gp_pred(gp, x, y, p, [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-12 
            warnings = sprintf([warnings '\n * Check gp_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check gp_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = gp_pred(gp, x, y, p2, [], tstindex2);
    [Ef1, Varf1] = gp_pred(gp, x, y, p2, [1], tstindex2);
    [Ef2, Varf2] = gp_pred(gp, x, y, p2, [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-12 
            warnings = sprintf([warnings '\n * Check gp_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check gp_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = gp_pred(gp, x, y, p2, [], tstindex2);
    
    if max( abs(Efaa - Ey) ) > 1e-12 
        warnings = sprintf([warnings '\n * Check gp_pred for ' models{i} ' model. The predictive mean for f and y do not match.']);
        numwarn = numwarn + 1;
    end

    if max( abs(Varfaa + gp.noise{1}.noiseSigma2  - Vary) ) > 1e-12 
        warnings = sprintf([warnings '\n * Check gp_pred for ' models{i} ' model. The predictive variance for f and y do not match.']);
        numwarn = numwarn + 1;
    end
    
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = gp_pred(gp, x, y, x, [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = gp_pred(gp, x, y, x, [], tstindex2, y);
    end

    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 30;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.07;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    hmc2('state', sum(100*clock));
    
    % Do the sampling (this takes approximately 5-10 minutes)    
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);
    
    % --- Check predictions ---
    [Efa, Varfa] = mc_pred(rfull, x, y, p, [], tstindex);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p, [1], tstindex);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p, [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-12 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = mc_pred(rfull, x, y, p2, [], tstindex2);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p2, [1], tstindex2);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p2, [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-12 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = mc_pred(rfull, x, y, p2, [], tstindex2);
    
    if max( abs(Efaa - Ey) ) > 1e-12 
        warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictive mean for f and y do not match.']);
        numwarn = numwarn + 1;
    end

    if max( abs(Varfaa + repmat(rfull.noise{1}.noiseSigma2',length(p2),1)  - Vary) ) > 1e-12 
        warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictive variance for f and y do not match.']);
        numwarn = numwarn + 1;
    end
    
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], tstindex2, y);
    end    
end


% =========================== 
% check metrics
% ===========================


S = which('test_package');
L = strrep(S,'test_package.m','demos/dat.1');
data=load(L);
data = data(1:2:end,:);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

fprintf(' \n ================================= \n \n Checking the metrics \n \n ================================= \n ')

% priors 
pm = prior_logunif('init');
pl = prior_logunif('init');

% First input
gpcf1 = gpcf_sexp('init', 'magnSigma2', 0.2, 'magnSigma2_prior', pm);
metric1 = metric_euclidean('init', {[1]},'lengthScales',[0.8], 'lengthScales_prior', pl);
% Lastly, plug the metric to the covariance function structure.
gpcf1 = gpcf_sexp('set', gpcf1, 'metric', metric1);

% Do the same for the second input
gpcf2 = gpcf_sexp('init', 'magnSigma2', 0.2);
metric2 = metric_euclidean('init', {[2]},'lengthScales',[1.2], 'lengthScales_prior', pl);
gpcf2 = gpcf_sexp('set', gpcf2, 'metric', metric2);

% We also need the noise component
gpcfn = gpcf_noise('init', 'noiseSigma2', 0.2);

% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', 'regr', {gpcf1,gpcf2}, {gpcfn}, 'jitterSigma2', 0.001)

param = 'covariance';
gradcheck(gp_pak(gp,param), @gp_e, @gp_g, gp, x, y, param);

w=gp_pak(gp, 'covariance');  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, 'covariance');

% --- MCMC approach ---
opt=gp_mcopt;
opt.nsamples= 30;
opt.repeat=2;
opt.hmc_opt = hmc2_opt;
opt.hmc_opt.steps=2;
opt.hmc_opt.stepadj=0.07;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));
    
[rfull,g,rstate1] = gp_mc(opt, gp, x, y);

% After sampling we delete the burn-in and thin the sample chain
rfull = thin(rfull, 10, 2);

if sqrt(mean((y - gp_pred(gp, x, y, x)).^2)) > 0.186*2
    warnings = sprintf([warnings '\n * Check the optimization of hyper-parameters of metric_euclidean with GP regression']);       
    numwarn = numwarn + 1;
end

if sqrt(mean((y - mean(mc_pred(rfull, x, y, x),2) ).^2)) > 0.186*2
    warnings = sprintf([warnings '\n * Check the MCMC sampling of hyper-parameters of metric_euclidean with GP regression']);
    numwarn = numwarn + 1;
end

delta = gradcheck(w, @gp_e, @gp_g, gp, x, y, 'covariance');
    
metric1 = metric_euclidean('init', {[1]},'lengthScales',[0.8], 'lengthScales_prior', pl);
% Lastly, plug the metric to the covariance function structure.
gpcf1 = gpcf_sexp('set', gpcf1, 'metric', metric1);
gp = gp_init('init', 'FULL', 'regr', {gpcf1, gpcf2}, {gpcfn}, 'jitterSigma2', 0.0001);
w=gp_pak(gp, 'covariance');
delta = [delta ; gradcheck(w, @gp_e, @gp_g, gp, x, y, 'covariance')];

% check that gradients are OK
if delta>0.0001
    warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters of metric_euclidean']);
    numwarn = numwarn + 1;
end




% ===========================
% check priors
% ===========================

S = which('test_package');
L = strrep(S,'test_package.m','demos/dat.1');
data=load(L);
data = data(1:2:end,:);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

fprintf(' \n ================================= \n \n Checking the prior structures \n \n ================================= \n ')

priorfunc = {'prior_t' 'prior_unif' 'prior_logunif', 'prior_gamma', 'prior_laplace', 'prior_sinvchi2', 'prior_normal', 'prior_invgam'};

gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.2^2);
ps = prior_logunif('init');
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigma2_prior', ps);
        
for i = 1:length(priorfunc)
    
    gpcf1 = gpcf_sexp('init', 'lengthScale', [1.3 1.4], 'magnSigma2', 0.2^2);
    
    % Set prior
    pl = feval(priorfunc{i}, 'init');
    gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl);
    gpcf1 = gpcf_sexp('set', gpcf1, 'magnSigma2_prior', ps);
           
    gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001)
    
    w=gp_pak(gp, 'covariance');  % pack the hyperparameters into one vector
    fe=str2fun('gp_e');     % create a function handle to negative log posterior
    fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
    
    % set the options for scg2
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    
    % do the optimization
    w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    
    % Set the optimized hyperparameter values back to the gp structure
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gp_e, @gp_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters of ' covfunc{i}]);
        warning([' Check the gradients of ' covfunc{i}]);
        numwarn = numwarn + 1;
    end

    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 20;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.07;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    hmc2('state', sum(100*clock));
    
    % Do the sampling
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);

    % After sampling we delete the burn-in and thin the sample chain
    rfull = thin(rfull, 5, 2);            
end

% ==========================================
% ==========================================
% Other models than Gaussian regression
% ==========================================
% ==========================================

% =========================== 
% Probit model 
% ===========================

fprintf(' \n ================================= \n \n Check the probit model \n \n =================================\n')

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

% Set prior
pl = prior_logunif('init');

gpcf1 = gpcf_sexp('init', 'lengthScale', [0.5 0.7], 'magnSigma2', 0.2^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcf2 = gpcf_ppcs1('init', nin, 'lengthScale', [1.1 1.3], 'magnSigma2', 0.2^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
gpcf3 = gpcf_ppcs3('init', nin, 'lengthScale', [1.1 1.3], 'magnSigma2', 0.5^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);

% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
X_u=[u1(:) u2(:)];
X_u = X_u([3 4 7:18 20:24 26:30 33:36],:);

% set the data points into clusters
[p1,p2]=meshgrid(-1.25:0.1:0.9, -0.2:0.1:1.1);
p=[p1(:) p2(:)];
b1 = linspace(-1.25, 0.9, 5);
b2 = linspace(-0.2,  1.1, 5);
trindex={}; testindex={};
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b2(i2)<=x(ind',2) & x(ind',2) < b2(i2+1));
        trindex{4*(i1-1)+i2} = ind';
        ind2 = 1:size(p,1);
        ind2 = ind2(: , b1(i1)<=p(ind2',1) & p(ind2',1) < b1(i1+1));
        ind2 = ind2(: , b2(i2)<=p(ind2',2) & p(ind2',2) < b2(i2+1));
        testindex{4*(i1-1)+i2} = ind2';
    end
end
trindex = {trindex{[1:3 5:16]}};
p2 = [x ; p(1:10,:)];
models = {'FULL' 'CS-FULL' 'FIC' 'PIC' 'CS+FIC'};

likelih = likelih_probit('init', y);

for i=1:length(models)
    switch models{i}
      case 'FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'CS-FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf2, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end
    
    gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, 'covariance'});
   
    fe=str2fun('gpla_e');
    fg=str2fun('gpla_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp, 'covariance');
    w = scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gpla_e, @gpla_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters for Laplace approximation for probit model and ' models{i} ' GP']);
        numwarn = numwarn + 1;
    end

    % --- Check predictions ---
    [Efa, Varfa] = la_pred(gp, x, y, p, 'covariance', [], tstindex);
    [Ef1, Varf1] = la_pred(gp, x, y, p, 'covariance', [1], tstindex);
    [Ef2, Varf2] = la_pred(gp, x, y, p, 'covariance', [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = la_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    [Ef1, Varf1] = la_pred(gp, x, y, p2, 'covariance', [1], tstindex2);
    [Ef2, Varf2] = la_pred(gp, x, y, p2, 'covariance', [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = la_pred(gp, x, y, p2, 'covariance', [], tstindex2);
        
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = la_pred(gp, x, y, x, 'covariance', [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = la_pred(gp, x, y, x, 'covariance', [], tstindex2, y);
    end


    % -------------------
    % --- EP approach ---
    % -------------------
    switch models{i}
      case 'FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'CS-FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf2, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.1, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.1, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.5, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end

    gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'covariance'});
   
    fe=str2fun('gpep_e');
    fg=str2fun('gpep_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp, 'covariance');
    w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gpep_e, @gpep_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters for Laplace approximation for probit model and ' models{i} ' GP']);
        warning([' Check the gradients of ' covfunc{i}]);
        numwarn = numwarn + 1;
    end

    % --- Check predictions ---
    [Efa, Varfa] = ep_pred(gp, x, y, p, 'covariance', [], tstindex);
    [Ef1, Varf1] = ep_pred(gp, x, y, p, 'covariance', [1], tstindex);
    [Ef2, Varf2] = ep_pred(gp, x, y, p, 'covariance', [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = ep_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    [Ef1, Varf1] = ep_pred(gp, x, y, p2, 'covariance', [1], tstindex2);
    [Ef2, Varf2] = ep_pred(gp, x, y, p2, 'covariance', [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = ep_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = ep_pred(gp, x, y, x, 'covariance', [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = ep_pred(gp, x, y, x, 'covariance', [], tstindex2, y);
    end
    
    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 50;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.02;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    opt.latent_opt.display=0;
    opt.latent_opt.repeat = 20;
    opt.latent_opt.sample_latent_scale = 0.5;
    hmc2('state', sum(100*clock));

% $$$     gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
    gp=gp_unpak(gp,w, 'covariance');
    gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});
    
    % Do the sampling (this takes approximately 5-10 minutes)    
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);
    
    % --- Check predictions ---
    [Efa, Varfa] = mc_pred(rfull, x, y, p, [], tstindex);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p, [1], tstindex);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p, [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(mean(Efa) - (mean(Ef1)+mean(Ef2))) ) > 1e-11
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = mc_pred(rfull, x, y, p2, [], tstindex2);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p2, [1], tstindex2);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p2, [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(mean(Efa) - (mean(Ef1)+mean(Ef2))) ) > 1e-11
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = mc_pred(rfull, x, y, p2, [], tstindex2);
        
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], tstindex2, y);
    end
end


% =========================== 
% Logit model 
% ===========================

likelih = likelih_logit('init', y);

for i=1:length(models)
    switch models{i}
      case 'FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'CS-FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf2, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end
    
    gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, 'covariance'});
   
    fe=str2fun('gpla_e');
    fg=str2fun('gpla_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp, 'covariance');
    w = scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gpla_e, @gpla_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters for Laplace approximation for logit model and ' models{i} ' GP']);
        numwarn = numwarn + 1;
    end

    % --- Check predictions ---
    [Efa, Varfa] = la_pred(gp, x, y, p, 'covariance', [], tstindex);
    [Ef1, Varf1] = la_pred(gp, x, y, p, 'covariance', [1], tstindex);
    [Ef2, Varf2] = la_pred(gp, x, y, p, 'covariance', [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = la_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    [Ef1, Varf1] = la_pred(gp, x, y, p2, 'covariance', [1], tstindex2);
    [Ef2, Varf2] = la_pred(gp, x, y, p2, 'covariance', [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = la_pred(gp, x, y, p2, 'covariance', [], tstindex2);
        
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = la_pred(gp, x, y, x, 'covariance', [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = la_pred(gp, x, y, x, 'covariance', [], tstindex2, y);
    end


    % -------------------
    % --- EP approach ---
    % -------------------
    switch models{i}
      case 'FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'CS-FULL' 
        gp = gp_init('init', 'FULL', likelih, {gpcf2, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.1, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.1, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.5, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end

    gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'covariance'});
   
    fe=str2fun('gpep_e');
    fg=str2fun('gpep_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp, 'covariance');
    w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gpep_e, @gpep_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters for Laplace approximation for logit model and ' models{i} ' GP']);
        warning([' Check the gradients of ' covfunc{i}]);
        numwarn = numwarn + 1;
    end

    % --- Check predictions ---
    [Efa, Varfa] = ep_pred(gp, x, y, p, 'covariance', [], tstindex);
    [Ef1, Varf1] = ep_pred(gp, x, y, p, 'covariance', [1], tstindex);
    [Ef2, Varf2] = ep_pred(gp, x, y, p, 'covariance', [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = ep_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    [Ef1, Varf1] = ep_pred(gp, x, y, p2, 'covariance', [1], tstindex2);
    [Ef2, Varf2] = ep_pred(gp, x, y, p2, 'covariance', [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with logit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = ep_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = ep_pred(gp, x, y, x, 'covariance', [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = ep_pred(gp, x, y, x, 'covariance', [], tstindex2, y);
    end
    
    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 50;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.02;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    opt.latent_opt.display=0;
    opt.latent_opt.repeat = 20;
    opt.latent_opt.sample_latent_scale = 0.5;
    hmc2('state', sum(100*clock));

% $$$     gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
    gp=gp_unpak(gp,w, 'covariance');
    gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});
    
    % Do the sampling (this takes approximately 5-10 minutes)    
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);
    
    % --- Check predictions ---
    [Efa, Varfa] = mc_pred(rfull, x, y, p, [], tstindex);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p, [1], tstindex);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p, [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(mean(Efa) - (mean(Ef1)+mean(Ef2))) ) > 1e-11
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = mc_pred(rfull, x, y, p2, [], tstindex2);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p2, [1], tstindex2);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p2, [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(mean(Efa) - (mean(Ef1)+mean(Ef2))) ) > 1e-11
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = mc_pred(rfull, x, y, p2, [], tstindex2);
        
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], tstindex2, y);
    end
end


% =========================== 
% Poisson model
% ===========================

fprintf(' \n ================================= \n \n Check the poisson model \n \n =================================\n')


S = which('demo_spatial1');
L = strrep(S,'demo_spatial1.m','demos/spatial.mat');
load(L)

x = xx;

% Create the covariance function
gpcf1 = gpcf_sexp('init', 'lengthScale', [0.5 0.7], 'magnSigma2', 0.2^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_ppcs1('init', nin, 'lengthScale', [1.1 1.3], 'magnSigma2', 0.2^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf3 = gpcf_ppcs3('init', nin, 'lengthScale', [1.1 1.3], 'magnSigma2', 0.5^2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

p = x+0.5;
dims = [1    60     1    35];
[trindex, Xu, tstindex] = set_PIC(xx, dims, 5, 'corners+1xside', 1, p);
p2 = [x ; p(1:10,:)];
models = {'FULL' 'CS-FULL' 'FIC' 'PIC' 'CS+FIC'};

for i=1:length(models)
    switch models{i}
      case 'FULL' 
        % reduce the data in order to make the demo faster
        ind = find(xx(:,2)<25);
        x = xx(ind,:);
        y = yy(ind,:);
        yee = ye(ind,:);
        [n,nin] = size(xx);
        
        % Create the likelihood structure
        likelih = likelih_poisson('init', y, yee);

        gp = gp_init('init', 'FULL', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'CS-FULL' 
        x = xx;
        y = yy;
        yee = ye;
        [n,nin] = size(xx);
        
        % Create the likelihood structure
        likelih = likelih_poisson('init', y, yee);
        
        gp = gp_init('init', 'FULL', likelih, {gpcf2, gpcf3}, {}, 'jitterSigma2', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        x = xx;
        y = yy;
        yee = ye;
        [n,nin] = size(xx);
        
        % Create the likelihood structure
        likelih = likelih_poisson('init', y, yee);
        
        gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        x = xx;
        y = yy;
        yee = ye;
        [n,nin] = size(xx);
        
        % Create the likelihood structure
        likelih = likelih_poisson('init', y, yee);
        
        gp = gp_init('init', 'PIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        x = xx;
        y = yy;
        yee = ye;
        [n,nin] = size(xx);
        
        % Create the likelihood structure
        likelih = likelih_poisson('init', y, yee);
        
        gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end
    
    gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, 'covariance'});
   
    fe=str2fun('gpla_e');
    fg=str2fun('gpla_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp, 'covariance');
    w = scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gpla_e, @gpla_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters for Laplace approximation for poisson model and ' models{i} ' GP']);
        numwarn = numwarn + 1;
    end

    % --- Check predictions ---
    [Efa, Varfa] = la_pred(gp, x, y, p, 'covariance', [], tstindex);
    [Ef1, Varf1] = la_pred(gp, x, y, p, 'covariance', [1], tstindex);
    [Ef2, Varf2] = la_pred(gp, x, y, p, 'covariance', [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with poisson model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with poisson model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = la_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    [Ef1, Varf1] = la_pred(gp, x, y, p2, 'covariance', [1], tstindex2);
    [Ef2, Varf2] = la_pred(gp, x, y, p2, 'covariance', [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with poisson model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check la_pred for ' models{i} ' GP with poisson model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = la_pred(gp, x, y, p2, 'covariance', [], tstindex2);
        
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = la_pred(gp, x, y, x, 'covariance', [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = la_pred(gp, x, y, x, 'covariance', [], tstindex2, y);
    end


    % -------------------
    % --- EP approach ---
    % -------------------
% $$$     switch models{i}
% $$$       case 'FULL' 
% $$$         gp = gp_init('init', 'FULL', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
% $$$         tstindex = [];
% $$$         tstindex2 = 1:n;
% $$$       case 'CS-FULL' 
% $$$         gp = gp_init('init', 'FULL', likelih, {gpcf2, gpcf3}, {}, 'jitterSigma2', 0.0001)
% $$$         tstindex = [];
% $$$         tstindex2 = 1:n;
% $$$       case 'FIC' 
% $$$         gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.1, 'X_u', X_u);
% $$$         tstindex = [];
% $$$         tstindex2 = 1:n;
% $$$       case 'PIC'
% $$$         gp = gp_init('init', 'PIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.1, 'X_u', X_u);
% $$$         gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
% $$$         tstindex = testindex;
% $$$         tstindex2 = trindex;
% $$$         tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
% $$$       case 'CS+FIC'
% $$$         gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.5, 'X_u', X_u);
% $$$         tstindex = [];
% $$$         tstindex2 = 1:n;
% $$$     end

    gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'covariance'});
   
    fe=str2fun('gpep_e');
    fg=str2fun('gpep_g');
    n=length(y);
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    opt.maxiter = 30;

    % do scaled conjugate gradient optimization 
    w = gp_pak(gp, 'covariance');
    w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');
    gp=gp_unpak(gp,w, 'covariance');

    delta = gradcheck(w, @gpep_e, @gpep_g, gp, x, y, 'covariance');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters for Laplace approximation for probit model and ' models{i} ' GP']);
        warning([' Check the gradients of ' covfunc{i}]);
        numwarn = numwarn + 1;
    end

    % --- Check predictions ---
    [Efa, Varfa] = ep_pred(gp, x, y, p, 'covariance', [], tstindex);
    [Ef1, Varf1] = ep_pred(gp, x, y, p, 'covariance', [1], tstindex);
    [Ef2, Varf2] = ep_pred(gp, x, y, p, 'covariance', [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = ep_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    [Ef1, Varf1] = ep_pred(gp, x, y, p2, 'covariance', [1], tstindex2);
    [Ef2, Varf2] = ep_pred(gp, x, y, p2, 'covariance', [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-11 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check ep_pred for ' models{i} ' GP with probit model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = ep_pred(gp, x, y, p2, 'covariance', [], tstindex2);
    
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = ep_pred(gp, x, y, x, 'covariance', [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = ep_pred(gp, x, y, x, 'covariance', [], tstindex2, y);
    end
    
    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 50;
    opt.repeat=2;
    opt.hmc_opt = hmc2_opt;
    opt.hmc_opt.steps=2;
    opt.hmc_opt.stepadj=0.02;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.6;
    opt.hmc_opt.nsamples=1;
    
    opt.latent_opt.nsamples=1;
    opt.latent_opt.nomit=0;
    opt.latent_opt.persistence=0;    
    opt.latent_opt.repeat=1;
    opt.latent_opt.steps=7;
    opt.latent_opt.window=1;
    opt.latent_opt.stepadj=0.15;

    opt.latent_opt.display=0;
    hmc2('state', sum(100*clock));

% $$$     gp = gp_init('init', 'FIC', likelih, {gpcf1, gpcf3}, {}, 'jitterSigma2', 0.0001)
    gp=gp_unpak(gp,w, 'covariance');
    gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_hmc});
    
    % Do the sampling (this takes approximately 5-10 minutes)    
    [rfull,g,rstate1] = gp_mc(opt, gp, x, y);
    
    % --- Check predictions ---
    [Efa, Varfa] = mc_pred(rfull, x, y, p, [], tstindex);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p, [1], tstindex);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p, [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(mean(Efa) - (mean(Ef1)+mean(Ef2))) ) > 1e-11
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = mc_pred(rfull, x, y, p2, [], tstindex2);
    [Ef1, Varf1] = mc_pred(rfull, x, y, p2, [1], tstindex2);
    [Ef2, Varf2] = mc_pred(rfull, x, y, p2, [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(mean(Efa) - (mean(Ef1)+mean(Ef2))) ) > 1e-11
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check mc_pred for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = mc_pred(rfull, x, y, p2, [], tstindex2);
        
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = mc_pred(rfull, x, y, x, [], tstindex2, y);
    end
end




% =========================== 
% Neg-Bin model
% ===========================





% =========================== 
% Student-t regression
% ===========================



%----------------------------
% check priors
%----------------------------

%----------------------------
% check sparse approximations
%----------------------------
