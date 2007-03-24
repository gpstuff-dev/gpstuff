S = which('demo_2ingp');
L = strrep(S,'demo_2ingp.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

gpcf1 = gpcf_sexp('init', nin);       % LISÄÄ INITiin (...'noiseSigmas', 0.2...)  muuta se -> sexp
gpcf2 = gpcf_noise('init', nin);
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale', repmat(1,1,nin), 'magnSigma2', 0.2^2);    %10
gpcf2 = gpcf_noise('set', gpcf2, 'noiseSigmas2', 0.2^2);

% Set the prior
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});         % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = sinvchi2_p({0.1^2 0.5 0.05^2 1});  % gamma_p({xx xx xx xx});    invgam_p({0.1 0.5 0.05 1});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 1)
%gp = gp_init('set', gp, 'jitterSigmas', 1)    
 
% Set the sampling options
opt=gp2_mcopt;
opt.repeat=20;
opt.nsamples=1;
opt.hmc_opt.steps=20;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Sample 
[r,g,rstate1]=gp_mc(opt, gp, x, y);




% Testing gp_pak and gp_unpak functions
w = gp_pak(gp)
gp = gp_unpak(gp, w)

% Test the covariance function
C1 = feval(gpcf.fh_trcov, gpcf, x);
C2 = gp_cov(gp,x,x);



