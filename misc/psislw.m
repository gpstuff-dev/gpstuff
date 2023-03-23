function [lw,kss] = psislw(lw,Reff)
%PSIS Pareto smoothed importance sampling
%   
%  Description
%    [LW,K] = PSISLW(LW,Reff) returns log weights LW
%    and Pareto tail indeces K, given log weights and optional arguments:
%      Reff - relative MCMC efficiency N_eff/N
%    
%  Reference
%    Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Pareto
%    smoothed importance sampling. https://arxiv.org/abs/1507.02646v5
%
% Copyright (c) 2015-2017 Aki Vehtari, Tuomas Sivula

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
if size(lw,1)<=1
    error('psislw: more than one log-weight needed');
end
if nargin<2
    Reff=1;
end

for i1=1:size(lw,2)
    % Loop over sets of log weights
    x=lw(:,i1);
    % improve numerical accuracy
    x=x-max(x);
    % Divide log weights into body and right tail
    n=numel(x);
    xs=sort(x);
    xcutoff=xs(end-ceil(min(0.2*n,3*sqrt(n/Reff))));
    if xcutoff<log(realmin)
        % need to stay above realmin
        xcutoff=-700;
    end
    x1=x(x<=xcutoff);
    x2=x(x>xcutoff);
    n2=numel(x2);
    if n2<=4
        % not enough tail samples for gpdfitnew
        qx=x;
        k=Inf;
    else
        % store order of tail samples
        [~,x2si]=sort(x2);
        % fit generalized Pareto distribution to the right tail samples
        [k,sigma]=gpdfitnew(exp(x2)-exp(xcutoff));
    end
    if k<1/3 || isinf(k)
        % no smoothing if short tail or GPD fit failed
        qx=x;
    else
        x1=x(x<=xcutoff);
        x2=x(x>xcutoff);
        n2=numel(x2);
        [~,x2si]=sort(x2);
        % compute ordered statistic for the fit
        qq=gpinv(([1:n2]-0.5)/n2,k,sigma)+exp(xcutoff);
        % remap back to the original order
        slq=zeros(n2,1);slq(x2si)=log(qq);
        % join body and GPD smoothed tail
        qx=x;qx(x<=xcutoff)=x1;qx(x>xcutoff)=slq;
        % truncate smoothed values to the largest raw weight
        lwtrunc=max(x);
        qx(qx>lwtrunc)=lwtrunc;
    end
    % renormalize weights
    lwx=bsxfun(@minus,qx,sumlogs(qx));
    % return log weights and tail index k
    lw(:,i1)=lwx;
    kss(1,i1)=k;
end
end

function x = gpinv(p,k,sigma)
% Octave compatibility by Tuomas Sivula
    x = NaN(size(p));
    if sigma <= 0
        return
    end
    ok = (p>0) & (p<1);
    if abs(k) < eps
        x(ok) = -log1p(-p(ok));
    else
    x(ok) = expm1(-k * log1p(-p(ok))) ./ k;
    end
    x = sigma*x;
    if ~all(ok)
        x(p==0) = 0;
        x(p==1 & k>=0) = Inf;
        x(p==1 & k<0) = -sigma/k;
    end
end
