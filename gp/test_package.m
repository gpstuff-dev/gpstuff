% TODO!
% - tulosten oikeellisuuden tarkistus (nyt tarkistetaan vain, että funktioissa ei bugeja) 


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

gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);
gpcf2.p.noiseSigmas2 = logunif_p;

for i = 1:length(covfunc)
    
    switch covfunc{i}
      case 'gpcf_neuralnetwork'
        gpcf1 = feval(covfunc{i}, 'init', nin);
        
        % Set prior
        gpcf1.p.weightSigma2 = logunif_p;
        gpcf1.p.biasSigma2 = logunif_p;
      case 'gpcf_dotproduct'
        gpcf1 = feval(covfunc{i}, 'init', nin);

        % Set prior
        gpcf1.p.coeffSigma2 = logunif_p;
        gpcf1.p.constSigma2 = logunif_p;
      case 'gpcf_prod'
        gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', 1+0.01*randn(1,nin), 'magnSigma2', 0.5);
        gpcf4 = gpcf_exp('init', nin, 'lengthScale', 1+0.01*randn(1,nin), 'magnSigma2', 0.2^2);

        % Set prior
        gpcf3.p.lengthScale = logunif_p;
        gpcf3.p.magnSigma2 = logunif_p;
        gpcf4.p.lengthScale = logunif_p;
        gpcf4.p.magnSigma2 = logunif_p;

        gpcf1 = gpcf_prod('init', nin, 'functions', {gpcf3, gpcf4});
      otherwise
        gpcf1 = feval(covfunc{i}, 'init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
        
        % Set prior
        gpcf1.p.lengthScale = logunif_p;
        gpcf1.p.magnSigma2 = logunif_p;
    end
           
    gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.0001)
    
    w=gp_pak(gp, 'hyper');  % pack the hyperparameters into one vector
    fe=str2fun('gp_e');     % create a function handle to negative log posterior
    fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
    
    % set the options for scg2
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    
    % do the optimization
    w=scg2(fe, w, opt, fg, gp, x, y, 'hyper');
    
    % Set the optimized hyperparameter values back to the gp structure
    gp=gp_unpak(gp,w, 'hyper');

    delta = gradcheck(w, @gp_e, @gp_g, gp, x, y, 'hyper');
    
    % check that gradients are OK
    if delta>0.0001
        warnings = sprintf([warnings '\n * Check the gradients of hyper-parameters of ' covfunc{i}]);
        warning([' Check the gradients of ' covfunc{i}]);
        numwarn = numwarn + 1;
    end

    % --- MCMC approach ---
    opt=gp_mcopt;
    opt.nsamples= 150;
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
        
    if sqrt(mean((y - mean(gp_preds(rfull, x, y, x),2) ).^2)) > 0.186*2
        warnings = sprintf([warnings '\n * Check the MCMC sampling of hyper-parameters of ' covfunc{i} ' with GP regression']);
        numwarn = numwarn + 1;
    end
        
end


%----------------------------
% check sparse approximations
%----------------------------

fprintf(' \n ================================= \n \n Checking the sparse approximations \n \n ================================= \n ')

sparse = {'FIC' 'PIC' 'CS+FIC'};

gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);
gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Set prior
gpcf1.p.lengthScale = logunif_p;
gpcf1.p.magnSigma2 = logunif_p;
gpcf2.p.noiseSigmas2 = logunif_p;
gpcf3.p.lengthScale = logunif_p;
gpcf3.p.magnSigma2 = logunif_p;

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
        gp = gp_init('init', 'FIC', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.0001, 'X_u', X_u);
        tstindex = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = trindex;
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', nin, 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigmas', 0.0001, 'X_u', X_u)
        tstindex = 1:n;
    end
    
    param = 'hyper+inducing';
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
    opt.nsamples= 150;
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
        
    if sqrt(mean((y - mean(gp_preds(rfull, x, y, x, [], tstindex),2) ).^2)) > 0.186*2
        warnings = sprintf([warnings '\n * Check the MCMC sampling of hyper-parameters of ' covfunc{i} ' with GP regression']);
        numwarn = numwarn + 1;
    end
    
end

%----------------------------
% Check the additive model and 
% gp_pred and gp_preds
%----------------------------

fprintf(' \n ================================= \n \n Checking the additive model and gp_pred \n \n =================================\n ')

gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);
gpcf3 = gpcf_ppcs2('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Set prior
gpcf1.p.lengthScale = logunif_p;
gpcf1.p.magnSigma2 = logunif_p;
gpcf2.p.noiseSigmas2 = logunif_p;
gpcf3.p.lengthScale = logunif_p;
gpcf3.p.magnSigma2 = logunif_p;

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
        gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigmas', 0.0001)
        tstindex = [];
        tstindex2 = 1:n;
      case 'FIC' 
        gp = gp_init('init', 'FIC', nin, 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigmas', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
      case 'PIC'
        gp = gp_init('init', 'PIC', nin, 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigmas', 0.0001, 'X_u', X_u);
        gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});
        tstindex = testindex;
        tstindex2 = trindex;
        tstindex2{1} = [tstindex2{1} ; [n+1:length(p2)]'];
      case 'CS+FIC'
        gp = gp_init('init', 'CS+FIC', nin, 'regr', {gpcf1, gpcf3}, {gpcf2}, 'jitterSigmas', 0.0001, 'X_u', X_u);
        tstindex = [];
        tstindex2 = 1:n;
    end
    
    w=gp_pak(gp, 'hyper');  % pack the hyperparameters into one vector
    fe=str2fun('gp_e');     % create a function handle to negative log posterior
    fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
    
    % set the options for scg2
    opt = scg2_opt;
    opt.tolfun = 1e-3;
    opt.tolx = 1e-3;
    opt.display = 1;
    
    % do the optimization
    w=scg2(fe, w, opt, fg, gp, x, y, 'hyper');

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

    if max( abs(Varfaa + gp.noise{1}.noiseSigmas2  - Vary) ) > 1e-12 
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
    opt.nsamples= 50;
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
    [Efa, Varfa] = gp_preds(rfull, x, y, p, [], tstindex);
    [Ef1, Varf1] = gp_preds(rfull, x, y, p, [1], tstindex);
    [Ef2, Varf2] = gp_preds(rfull, x, y, p, [2], tstindex);
    
    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-12 
            warnings = sprintf([warnings '\n * Check gp_preds for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check gp_preds for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efa, Varfa] = gp_preds(rfull, x, y, p2, [], tstindex2);
    [Ef1, Varf1] = gp_preds(rfull, x, y, p2, [1], tstindex2);
    [Ef2, Varf2] = gp_preds(rfull, x, y, p2, [2], tstindex2);

    switch models{i}
      case {'FULL' 'CS+FIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 1e-12 
            warnings = sprintf([warnings '\n * Check gp_preds for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
        % In FIC and PIC the additive phenomenon is only approximate
      case {'FIC' 'PIC'}
        if max( abs(Efa - (Ef1+Ef2)) ) > 0.1 
            warnings = sprintf([warnings '\n * Check gp_preds for ' models{i} ' model. The predictions with only one covariance function do not match the full prediction.']);
            numwarn = numwarn + 1;
        end
    end
    
    [Efaa, Varfaa, Ey, Vary] = gp_preds(rfull, x, y, p2, [], tstindex2);
    
    if max( abs(Efaa - Ey) ) > 1e-12 
        warnings = sprintf([warnings '\n * Check gp_preds for ' models{i} ' model. The predictive mean for f and y do not match.']);
        numwarn = numwarn + 1;
    end

    if max( abs(Varfaa + repmat(rfull.noise{1}.noiseSigmas2',length(p2),1)  - Vary) ) > 1e-12 
        warnings = sprintf([warnings '\n * Check gp_preds for ' models{i} ' model. The predictive variance for f and y do not match.']);
        numwarn = numwarn + 1;
    end
    
    switch models{i}
      case {'PIC'}
        [Ef, Varf, Ey, Vary, py] = gp_preds(rfull, x, y, x, [], trindex, y);
      otherwise
        [Ef, Varf, Ey, Vary, py] = gp_preds(rfull, x, y, x, [], tstindex2, y);
    end
   
    
end




%----------------------------
% check metrics
%----------------------------

%----------------------------
% check priors
%----------------------------


















% =========================== 
% Student-t regression
% ===========================

%----------------------------
% check priors
%----------------------------

%----------------------------
% check sparse approximations
%----------------------------


% =========================== 
% Classification
% ===========================

% probit
% logit

%----------------------------
% check priors
%----------------------------

%----------------------------
% check sparse approximations
%----------------------------


% =========================== 
% spatial models
% ===========================

% Poisson
% Neg-Bin