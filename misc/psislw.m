function [lw,kss] = psislw(lw,wcpp,wtrunc)
%VGIS Pareto smoothed importance sampling
%   
%  Description
%    [LW,K] = PSISLW(LW,WCPP,WCUTOFF) returns log weights LW
%    and Pareto tail indeces K, given log weights and optional arguments:
%      WCPP    - percentage of samples used for generalise Pareto
%                distribution (GPD) fit estimate (default = 20)
%      WTRUNC  - parameter for truncating very large weights to N^WTRUNC,
%                with no truncation if 0 (default = 3/4)
%    
%  Reference:
%    Aki Vehtari and Andrew Gelman (2015). Pareto smoothed importance
%    sampling. arXiv preprint arXiv:1507.02646.
%
% Copyright (c) 2015 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
if size(lw,1)<=1
    error('vgislw: more than one log-weight needed');
end
if nargin<2
    wcpp=20;
end
if nargin<3
    wtrunc=3/4;
end

for i1=1:size(lw,2)
    % Loop over sets of log weights
    x=lw(:,i1);
    % improve numerical accuracy
    x=x-max(x);
    % Divide log weights into body and right tail
    n=numel(x);
    xcutoff=prctile(x,100-wcpp);
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
        % compute ordered statistic for the fit
        qq=gpinv(([1:n2]-0.5)/n2,k,sigma)+exp(xcutoff);
        % remap back to the original order
        slq=zeros(n2,1);slq(x2si)=log(qq);
        % join body and GPD smoothed tail
        qx=x;qx(x<=xcutoff)=x1;qx(x>xcutoff)=slq;
    end
    if wtrunc>0
        % truncate too large weights
        lwtrunc=wtrunc*log(n)-log(n)+sumlogs(qx);
        qx(qx>lwtrunc)=lwtrunc;
    end
    % renormalize weights
    lwx=bsxfun(@minus,qx,sumlogs(qx));
    % return log weights and tail index k
    lw(:,i1)=lwx;
    kss(1,i1)=k;
end
%ksi=find(kss>=0.5&kss<1);
% if ~isempty(ksi)
%     warning('Following indeces have estimated tail index 1/2>k>1');
%     disp(ksi)
% end
% ksi=find(kss>=1);
% if ~isempty(ksi)
%     warning('Following indeces have estimated tail index k>1');
%     disp(ksi)
% end
