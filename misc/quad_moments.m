function [m_0, m_1, m_2] = quad_moments(fun, a, b, rtol, atol)
% QUAD_MOMENTS  Calculate the 0th, 1st and 2nd moment of a given (unnormalized) probability distribution
%
%   [m_0, m_1, m_2] = quad_moments(fun, a, b, varargin) 
%   Inputs:
%      fun  = Function handle to the unnormalized probability distribution
%      a,b  = integration limits [a,b]
%      rtol = relative tolerance for the integration (optional, default 1e-6)
%      atol = absolute tolerance for the integration (optional, default 1e-10)
%               
%   Returns the first three moments:
%      m0  = int_a^b fun(x) dx
%      m1  = int_a^b x*fun(x) dx / m0
%      m2  = int_a^b x^2*fun(x) dx / m0
%
%   The function uses an adaptive Gaus Kronrod quadrature. The same set of 
%   integration points and intervals are used for each moment. This speeds up 
%   the evaluations by factor 3, since the function evaluations are done only 
%   once.
% 
%   The quadrature method is described by:
%   L.F. Shampine, "Vectorized Adaptive Quadrature in Matlab",
%   Journal of Computational and Applied Mathematics, to appear.

%   Copyright (c) 2010 Jarno Vanhatalo, Jouni Hartikainen
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% Set integration parameters.
    
    maxint = 650;
    
    if nargin < 3
        rtol = 1.e-6;
    end
    if nargin < 4
        atol = 1.e-10;
    end

    % nodes and weights
    pnodes = [ ...
        0.2077849550078985; 0.4058451513773972; 0.5860872354676911; ...
        0.7415311855993944; 0.8648644233597691; 0.9491079123427585; ...
        0.9914553711208126];
    pwt = [ ...
        0.2044329400752989, 0.1903505780647854, 0.1690047266392679, ...
        0.1406532597155259, 0.1047900103222502, 0.06309209262997855, ...
        0.02293532201052922];
    pwt7 = [0,0.3818300505051189,0,0.2797053914892767,0,0.1294849661688697,0];
    nodes = [-pnodes(end:-1:1); 0; pnodes];
    wt = [pwt(end:-1:1), 0.2094821410847278, pwt];
    ewt = wt - [pwt7(end:-1:1), 0.4179591836734694, pwt7];
    
    % Integration interval
    tinterval = [a,b];
    
    % Compute the path length and split tinterval.
    minsubs = 10;   % number of subintervals
    pathlen = abs(b-a);
    if pathlen == 0
        error('The integration interval has to be greater than zero.')
    end
    nnew = ceil(pathlen*minsubs/pathlen) - 1;
    idxnew = find(nnew > 0);
    nnew = nnew(idxnew);
    for j = numel(idxnew):-1:1
        k = idxnew(j);
        nnj = nnew(j);
        newpts = tinterval(k) + (1:nnj)./(nnj+1)*(tinterval(k+1)-tinterval(k));
        tinterval = [tinterval(1:k),newpts,tinterval(k+1:end)];
    end
    
    % Remove useless subintervals.
    tinterval(abs(diff(tinterval))==0) = [];
    
    % Initialize array of subintervals of [a,b].
    subs = [tinterval(1:end-1);tinterval(2:end)];
    % Initialize partial sums.
    q_ok = 0;
    q1_ok = 0;
    q2_ok = 0;
    err_ok = 0;
    % The main loop
    while true
        % SUBS contains subintervals of [a,b] where the integral is not
        % sufficiently accurate. The first row of SUBS holds the left end
        % points and the second row, the corresponding right endpoints.
        midpt = sum(subs)/2;   % midpoints of the subintervals
        halfh = diff(subs)/2;  % half the lengths of the subintervals            
        x = bsxfun(@plus,nodes*halfh,midpt);
        x = reshape(x,1,[]);   % function f expects a row vector
        fx = fun(x);
        fx1 = fx.*x;
        fx2 = fx.*x.^2;
        
        fx = reshape(fx,numel(wt),[]);
        fx1 = reshape(fx1,numel(wt),[]);
        fx2 = reshape(fx2,numel(wt),[]);
        
        % Quantities for subintervals.
        qsubs = (wt*fx) .* halfh;
        errsubs = (ewt*fx) .* halfh;
        qsubs1 = (wt*fx1) .* halfh;
        qsubs2 = (wt*fx2) .* halfh;

        % Calculate current values of q and tol.
        q = sum(qsubs) + q_ok;
        q1 = sum(qsubs1) + q1_ok;
        q2 = sum(qsubs2) + q2_ok;
        tol = max(atol,rtol*abs(q));
        
        % Locate subintervals where the approximate integrals are
        % sufficiently accurate and use them to update the partial
        % error sum.
        ndx = find(abs(errsubs) <= (2*tol/pathlen)*halfh);
        err_ok = err_ok + sum(errsubs(ndx));
        
        % Remove errsubs entries for subintervals with accurate
        % approximations.
        errsubs(ndx) = [];
        
        % Remove subintervals with accurate approximations.
        subs(:,ndx) = [];
        if isempty(subs)
            break
        end
        % Update the partial sum for the integral.
        q_ok = q_ok + sum(qsubs(ndx));
        q1_ok = q1_ok + sum(qsubs1(ndx));
        q2_ok = q2_ok + sum(qsubs2(ndx));
        % Split the remaining subintervals in half. Quit if splitting
        % results in too many subintervals.
        nsubs = 2*size(subs,2);
        if nsubs > maxint
            warning('quad_moments: Reached the limit on the maximum number of intervals in use.');
            break
        end
        midpt(ndx) = []; % Remove unneeded midpoints.
        subs = reshape([subs(1,:); midpt; midpt; subs(2,:)],2,[]);
    end
    
    % Scale moments
    m_0 = q;
    m_1 = q1./q;
    m_2 = q2./q;
end 
