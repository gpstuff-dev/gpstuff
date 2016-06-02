function [e, g] = expectedvariance_eg(x_new, gp, x, ~, invC, varargin)
% expectedVariance_eg    Calculate the negative expected variance and its
%                        gradient  
%
%expectedVariance_eg(x_new, gp, x, [], invC)
% Arguments to the function are:
%   x_new  - query point 
%   gp     - GP model for the objective function
%   x      - previous query points for which we have calculated objective
%            function value y 
%   invC   - [~, C] = gp_trcov(gp,x); invC = inv(C);
%
% Copyright (c) 2015 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.


K = gp_cov(gp,x_new,x);
e = - ( gp_trvar(gp,x_new)-diag((K*invC*K')) );

if nargout > 1
    ncf = length(gp.cf);
    tmpK=zeros(1,size(x_new,2));
    tmpC= cell(size(x_new,2),1);
    for j1=1:length(tmpC)
        tmpC{j1} = zeros(1,size(x,1));
    end
    for i1=1:ncf
        gpcft = gp.cf{i1};
        tmpK = tmpK + cell2mat(gpcft.fh.ginput(gpcft,x_new));
        apu = gpcft.fh.ginput(gpcft,x_new,x);
        for j1= 1:length(apu)
            tmpC{j1} = tmpC{j1} + apu{j1};
        end
    end
    g = tmpK;
    for j1=1:length(tmpC)
        g(j1) = g(j1) - tmpC{j1}*invC*K' - K*invC*tmpC{j1}';
    end
    g=-g;
end