function [samples, energies, diagn] = latent_mh(x, opt, varargin)
%LATENT_MH   Metropolis Hastings algorithm by Neal
%
%       Description
%       SAMPLES = LATENT_MH(F, X, OPTIONS, GRADF) uses a Metropolis Hastings with 
%       Neals modification
%
%       See also
%       HMC2_OPT, METROP
%

%       Copyright (c) 1998-2000 Aki Vehtari
%       Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% AT THIS MOMENT WORKS ONLY FOR FULL GP

z = x;
gp = varargin{1};
p = varargin{2};
t = varargin{3};

maxcut = -log(eps);
mincut = -log(1/realmin - 1);
lvs=opt.sample_latent_scale;
a = max(min(z, maxcut),mincut);
[K,C]=gp_trcov(gp, p);
L=chol(C)';
n=length(t);
likelih_e = @logistic;
e = feval(likelih_e, gp, z, t);

% Adaptive control algorithm to find such a value for lvs 
% that the rejection rate of Metropolis is optimal. 
slrej = 0;
for li=1:100
    zt=sqrt(1-lvs.^2).*z+lvs.*L*randn(n,1);
    at = max(min(zt, maxcut),mincut);
    ed = feval(likelih_e, gp, zt, t);
    a=e-ed;
    if exp(a) > rand(1)
        z=zt;
        e=ed;
        lvs=min(1,lvs*1.1);
    else
        lvs=max(1e-8,lvs/1.05);
    end
end
opt.sample_latent_scale=lvs;
% Do the actual sampling 
for li=1:(opt.repeat)
    zt=sqrt(1-lvs.^2).*z+lvs.*L*randn(n,1);
    at = max(min(zt, maxcut),mincut);
    ed = feval(likelih_e, gp, zt, t);
    a=e-ed;
    if exp(a) > rand(1)
        z=zt;
        e=ed;
    else
        slrej=slrej+1;
    end
end
diagn.rej = slrej/opt.repeat;
diagn.lvs = lvs;
diagn.opt=opt;
energies=[];
samples = z';


    function e = logistic(gp, z, t)
    % LH_2CLASS     Minus log likelihood function for 2 class classification.
    %               A logistic likelihod
    %
    %       E = H_LOGIT(GP, P, T, Z) takes.... and returns minus log from 
    
    % If class prior is defined use it
    if isfield(gp,'classprior');
        cp=gp.classprior;     % TÄMÄM VAATII VIELÄ TOTETUTUKSEN
    else
        cp=1;
    end
    
    y = 1./(1 + exp(-z));
    e = -sum(sum(cp.*t.*log(y) + (1 - t).*log(1 - y)));
    end

end