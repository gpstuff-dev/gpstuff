function [q,errbnd] = quad_uusi(FUN,a,b,varargin)
%QUADGK  Numerically evaluate integral, adaptive Gauss-Kronrod quadrature.
%   Q = QUADGK(FUN,A,B) attempts to approximate the integral of
%   scalar-valued function FUN from A to B using high order global adaptive
%   quadrature and default error tolerances. The function Y=FUN(X) should
%   accept a vector argument X and return a vector result Y, the integrand
%   evaluated at each element of X. FUN must be a function handle. A and B
%   can be -Inf or Inf. If both are finite, they can be complex. If at
%   least one is complex, the integral is approximated over a straight line
%   path from A to B in the complex plane.
%
%   [Q,ERRBND] = QUADGK(...). ERRBND is an approximate bound on the
%   absolute error, |Q - I|, where I denotes the exact value of the
%   integral.
%
%   [Q,ERRBND] = QUADGK(FUN,A,B,PARAM1,VAL1,PARAM2,VAL2,...) performs
%   the integration with specified values of optional parameters. The
%   available parameters are
%
%   'AbsTol', absolute error tolerance
%   'RelTol', relative error tolerance
%
%       QUADGK attempts to satisfy ERRBND <= max(AbsTol,RelTol*|Q|). This
%       is absolute error control when |Q| is sufficiently small and
%       relative error control when |Q| is larger. A default tolerance
%       value is used when a tolerance is not specified. The default value
%       of 'AbsTol' is 1.e-10 (double), 1.e-5 (single). The default value
%       of 'RelTol' is 1.e-6 (double), 1.e-4 (single). For pure absolute
%       error control use
%         Q = quadgk(FUN,A,B,'AbsTol',ATOL,'RelTol',0)
%       where ATOL > 0. For pure relative error control use
%         Q = quadgk(FUN,A,B,'RelTol',RTOL,'AbsTol',0)
%       Except when using pure absolute error control, the minimum relative
%       tolerance is 100*eps(class(Q)).
%
%   'Waypoints', vector of integration waypoints
%
%       If FUN(X) has discontinuities in the interval of integration, the
%       locations should be supplied as a 'Waypoints' vector. When A, B,
%       and the waypoints are all real, the waypoints must be supplied in
%       strictly increasing or strictly decreasing order, and only the
%       waypoints between A and B are used. Waypoints are not intended for
%       singularities in FUN(X). Singular points should be handled by making 
%       them endpoints of separate integrations and adding the results.
%
%       If A, B, or any entry of the waypoints vector is complex, the
%       integration is performed over a sequence of straight line paths in
%       the complex plane, from A to the first waypoint, from the first
%       waypoint to the second, and so forth, and finally from the last
%       waypoint to B.
%
%   'MaxIntervalCount', maximum number of intervals allowed
%
%       The 'MaxIntervalCount' parameter limits the number of intervals
%       that QUADGK will use at any one time after the first iteration. A
%       warning is issued if QUADGK returns early because of this limit.
%       The default value is 650. Increasing this value is not recommended,
%       but it may be appropriate when ERRBND is small enough that the 
%       desired accuracy has nearly been achieved.
%
%   Notes:
%   QUAD may be most efficient for low accuracies with nonsmooth
%   integrands.  
%   QUADL may be more efficient than QUAD at higher accuracies
%   with smooth integrands.  
%   QUADGK may be most efficient for oscillatory integrands and any smooth
%   integrand at high accuracies. It supports infinite intervals and can
%   handle moderate singularities at the endpoints. It also supports
%   contour integration along piecewise linear paths. 
%   QUADV vectorizes QUAD for array-valued FUN.
%
%   Example:
%   Integrate f(x) = exp(-x^2)*log(x)^2 from 0 to infinity:
%      Q = quadgk(@myfun,0,Inf)
%   where myfun.m is the M-file function:
%      %------------------------%
%      function y = myfun(x)
%      y = exp(-x.^2).*log(x).^2;
%      %------------------------%
%
%   Example:
%   To use a parameter in the integrand:
%      Q = quadgk(@(x)myfun(x,5),0,2);
%   where myfun is the M-file function:
%      %----------------------%
%      function y = myfun(x,c)
%      y = 1./(x.^3-2*x-c);
%      %----------------------%
%
%   Example:
%   Integrate f(z) = 1/(2z-1) in the complex plane over the triangular
%   path from 0 to 1+1i to 1-1i to 0:
%      Q = quadgk(@(z)1./(2*z-1),0,0,'Waypoints',[1+1i,1-1i])
%
%   Class support for inputs A, B, and the output of FUN:
%      float: double, single
%
%   See also QUAD, QUADL, QUADV, DBLQUAD, TRIPLEQUAD, FUNCTION_HANDLE

%   Based on "quadva" by Lawrence F. Shampine.
%   Ref: L.F. Shampine, "Vectorized Adaptive Quadrature in Matlab",
%   Journal of Computational and Applied Mathematics, to appear.

%   Copyright 2007 The MathWorks, Inc.
%   $Revision: 1.1.6.3 $  $Date: 2007/06/14 05:08:52 $

% Variable names in all caps are referenced in nested functions.

% Validate the first three inputs.  
error(nargchk(3,inf,nargin,'struct'));
if ~isa(FUN,'function_handle')
    error('MATLAB:quadgk:funArgNotHandle', ...
        'First input argument must be a function handle.');
end
if ~(isscalar(a) && isfloat(a) && isscalar(b) && isfloat(b))
    error('MATLAB:quadgk:invalidEndpoint', ...
        'A and B must be scalar floats.');
end

% Gauss-Kronrod (7,15) pair. Use symmetry in defining nodes and weights.
pnodes = [ ...
    0.2077849550078985; 0.4058451513773972; 0.5860872354676911; ...
    0.7415311855993944; 0.8648644233597691; 0.9491079123427585; ...
    0.9914553711208126];
pwt = [ ...
    0.2044329400752989, 0.1903505780647854, 0.1690047266392679, ...
    0.1406532597155259, 0.1047900103222502, 0.06309209262997855, ...
    0.02293532201052922];
pwt7 = [0,0.3818300505051189,0,0.2797053914892767,0,0.1294849661688697,0];
NODES = [-pnodes(end:-1:1); 0; pnodes];
WT = [pwt(end:-1:1), 0.2094821410847278, pwt];
EWT = WT - [pwt7(end:-1:1), 0.4179591836734694, pwt7];

% Fixed parameters.
DEFAULT_DOUBLE_ABSTOL = 1.e-10;
DEFAULT_SINGLE_ABSTOL = 1.e-5;
DEFAULT_DOUBLE_RELTOL = 1.e-6;
DEFAULT_SINGLE_RELTOL = 1.e-4;
MININTERVALCOUNT = 10; % Minimum number subintervals to start.

% Initialize options. These variables are treated like global variables
% through the use of nested functions.
RTOL = [];
ATOL = [];
WAYPOINTS = [];
MAXINTERVALCOUNT = 650;

% Process optional input.
parseOptions(varargin{:});

% Initialize the FIRSTFUNEVAL variable.  Some checks will be done just
% after the first evaluation.
FIRSTFUNEVAL = true;

% Handle contour integration.
if ~(isreal(a) && isreal(b) && isreal(WAYPOINTS))
    tinterval = [a,WAYPOINTS,b];
    if any(~isfinite(tinterval))
        error('MATLAB:quadgk:nonFiniteContourError', ...
            'Contour endpoints and waypoints must be finite.');
    end
    % A and B should not be needed, so we do not define them here.
    % Perform the contour integration.
    [q,errbnd] = vadapt(@evalFun,tinterval);
    return
end

% Define A and B and note the direction of integration on real axis.
if b < a
    % Integrate left to right and change sign at the end.
    reversedir = true;
    A = b;
    B = a;
else
    reversedir = false;
    A = a;
    B = b;
end

% Process waypoints vector.
if ~isempty(WAYPOINTS)
    % Make waypoints increasing if they are decreasing.
    if WAYPOINTS(1) >= WAYPOINTS(end)
        WAYPOINTS = fliplr(WAYPOINTS);
    end
    % Validate waypoints.
    if any(diff(WAYPOINTS) <= 0) || any(~isfinite(WAYPOINTS))
        error('MATLAB:quadgk:invalidWaypoints', ...
            'Waypoints must be finite and strictly increase or strictly decrease.');
    end
end

% Construct interval vector with relevant waypoints.
interval = [A, WAYPOINTS(WAYPOINTS>A & WAYPOINTS<B), B];
% Extract A and B from interval vector to regularize possible mixed
% single/double inputs.
A = interval(1);
B = interval(end);

% Identify the task and perform the integration.
if A == B
    % Handles both finite and infinite cases.
    % Return zero or nan of the appropriate class.
    q = midpArea(@evalFun,A,B);
    errbnd = q;
elseif isfinite(A) && isfinite(B)
    if numel(interval) > 2
        % Analytical transformation suggested by K.L. Metlov:
        alpha = 2*sin( asin((A + B - 2*interval(2:end-1))/(A - B))/3 );
        interval = [-1,alpha,1];
    else
        interval = [-1,1];
    end
    [q,errbnd] = vadapt(@f1,interval);
elseif isfinite(A) && isinf(B)
    if numel(interval) > 2
        alpha = sqrt(interval(2:end-1) - A);
        interval = [0,alpha./(1+alpha),1];
    else
        interval = [0,1];
    end
    [q,errbnd] = vadapt(@f2,interval);
elseif isinf(A) && isfinite(B)
    if numel(interval) > 2
        alpha = sqrt(B - interval(2:end-1));
        interval = [-1,-alpha./(1+alpha),0];
    else
        interval = [-1,0];
    end
    [q,errbnd] = vadapt(@f3,interval);
elseif isinf(A) && isinf(B)
    if numel(interval) > 2
        % Analytical transformation suggested by K.L. Metlov:
        alpha = tanh( asinh(2*interval(2:end-1))/2 );
        interval = [-1,alpha,1];
    else
        interval = [-1,1];
    end
    [q,errbnd] = vadapt(@f4,interval);
else % i.e., if isnan(a) || isnan(b)
    q = midpArea(@evalFun,A,B);
    errbnd = q;
end
% Account for integration direction.
if reversedir
    q = -q;
end

%==Nested functions=========================================================

    function [q,errbnd] = vadapt(f,tinterval)
        % Iterative routine to perform the integration.
        % Compute the path length and split tinterval if needed.
        [tinterval,pathlen] = split(tinterval,MININTERVALCOUNT);
        if pathlen == 0
            % Test case: quadgk(@(x)x,1+1i,1+1i);
            q = midpArea(f,tinterval(1),tinterval(end));
            errbnd = q;
            return
        end
        % Initialize array of subintervals of [a,b].
        subs = [tinterval(1:end-1);tinterval(2:end)];
        % Initialize partial sums.
        q_ok = 0;
        err_ok = 0;
        % Initialize main loop
        while true
            % SUBS contains subintervals of [a,b] where the integral is not
            % sufficiently accurate. The first row of SUBS holds the left end
            % points and the second row, the corresponding right endpoints.
            midpt = sum(subs)/2;   % midpoints of the subintervals
            halfh = diff(subs)/2;  % half the lengths of the subintervals
            x = bsxfun(@plus,NODES*halfh,midpt);
            x = reshape(x,1,[]);   % function f expects a row vector
            [fx,too_close] = f(x);
            % Quit if mesh points are too close.
            if too_close
                break
            end
            fx = reshape(fx,numel(WT),[]);
            % Quantities for subintervals.
            qsubs = (WT*fx) .* halfh;
            errsubs = (EWT*fx) .* halfh;
            % Calculate current values of q and tol.
            q = sum(qsubs) + q_ok;
            tol = max(ATOL,RTOL*abs(q));
            % Locate subintervals where the approximate integrals are
            % sufficiently accurate and use them to update the partial
            % error sum.
            ndx = find(abs(errsubs) <= (2*tol/pathlen)*halfh);
            err_ok = err_ok + sum(errsubs(ndx));
            % Remove errsubs entries for subintervals with accurate
            % approximations.
            errsubs(ndx) = [];
            % The approximate error bound is constructed by adding the
            % approximate error bounds for the subintervals with accurate
            % approximations to the 1-norm of the approximate error bounds
            % for the remaining subintervals.  This guards against
            % excessive cancellation of the errors of the remaining
            % subintervals.
            errbnd = abs(err_ok) + norm(errsubs,1);
            % Check for nonfinites.
            if ~(isfinite(q) && isfinite(errbnd))
                warning('MATLAB:quadgk:NonFiniteValue', ...
                    'Infinite or Not-a-Number value encountered.');
                break
            end
            % Test for convergence.
            if errbnd <= tol
                break
            end         
            % Remove subintervals with accurate approximations.
            subs(:,ndx) = [];
            if isempty(subs)
                break
            end
            % Update the partial sum for the integral.
            q_ok = q_ok + sum(qsubs(ndx));
            % Split the remaining subintervals in half. Quit if splitting
            % results in too many subintervals.
            nsubs = 2*size(subs,2);
            if nsubs > MAXINTERVALCOUNT
                warning('MATLAB:quadgk:MaxIntervalCountReached', ...
                    ['Reached the limit on the maximum number of intervals in use.\n', ...
                    'Approximate bound on error is%9.1e. The integral may not exist, or\n', ...
                    'it may be difficult to approximate numerically. Increase MaxIntervalCount\n', ...
                    'to %d to enable QUADGK to continue for another iteration.'], ...
                    errbnd,nsubs);
                break
            end
            midpt(ndx) = []; % Remove unneeded midpoints.
            subs = reshape([subs(1,:); midpt; midpt; subs(2,:)],2,[]);
        end
    end % vadapt

%--------------------------------------------------------------------------

    function q = midpArea(f,a,b)
        % Return q = (b-a)*f((a+b)/2). Although formally correct as a low
        % order quadrature formula, this function is only used to return
        % nan or zero of the appropriate class when a == b, isnan(a), or
        % isnan(b).
        x = (a+b)/2;
        if isfinite(a) && isfinite(b) && ~isfinite(x)
            % Treat overflow, e.g. when finite a and b > realmax/2
            x = a/2 + b/2;
        end
        fx = f(x);
        if ~isfinite(fx)
            warning('MATLAB:quadgk:NonFiniteValue', ...
                'Infinite or Not-a-Number value encountered.');
        end
        q = (b-a)*fx;
    end % midpArea

%--------------------------------------------------------------------------

    function [fx,too_close] = evalFun(x)
        % Evaluate the integrand.
        if FIRSTFUNEVAL
            % Don't check the closeness of the mesh on the first iteration.
            too_close = false;
            fx = FUN(x);
            finalInputChecks(x,fx);
            FIRSTFUNEVAL = false;
        else
            too_close = checkSpacing(x);
            if too_close
                fx = [];
            else
                fx = FUN(x);
            end
        end
    end % evalFun

%--------------------------------------------------------------------------

    function [y,too_close] = f1(t)
        % Transform to weaken singularities at both ends: [a,b] -> [-1,1]
        tt = 0.25*(B-A)*t.*(3 - t.^2) + 0.5*(B+A);
        [y,too_close] = evalFun(tt);
        if ~too_close
            y = 0.75*(B-A)*y.*(1 - t.^2);
        end
    end % f1

%--------------------------------------------------------------------------

    function [y,too_close] = f2(t)
        % Transform to weaken singularity at left end: [a,Inf) -> [0,Inf).
        % Then transform to finite interval: [0,Inf) -> [0,1].
        tt = t ./ (1 - t);
        t2t = A + tt.^2;
        [y,too_close] = evalFun(t2t);
        if ~too_close
            y =  2*tt .* y ./ (1 - t).^2;
        end
    end % f2

%--------------------------------------------------------------------------

    function [y,too_close] = f3(t)
        % Transform to weaken singularity at right end: (-Inf,b] -> (-Inf,b].
        % Then transform to finite interval: (-Inf,b] -> (-1,0].
        tt = t ./ (1 + t);
        t2t = B - tt.^2;
        [y,too_close] = evalFun(t2t);
        if ~too_close
            y = -2*tt .* y ./ (1 + t).^2;
        end
    end % f3

%--------------------------------------------------------------------------

    function [y,too_close] = f4(t)
        % Transform to finite interval: (-Inf,Inf) -> (-1,1).
        tt = t ./ (1 - t.^2);
        [y,too_close] = evalFun(tt);
        if ~too_close
            y = y .* (1 + t.^2) ./ (1 - t.^2).^2;
        end
    end % f4

%--------------------------------------------------------------------------

    function too_close = checkSpacing(x)
        ax = abs(x);
        tcidx = find(abs(diff(x)) <= 100*eps(class(x))*max(ax(1:end-1),ax(2:end)),1);
        too_close = ~isempty(tcidx);
        if too_close
            warning('MATLAB:quadgk:MinStepSize', ...
                'Minimum step size reached near x = %g; singularity possible.', ...
                x(tcidx));
        end
    end % checkSpacing

%--------------------------------------------------------------------------

    function [x,pathlen] = split(x,minsubs)
        % Split subintervals in the interval vector X so that, to working
        % precision, no subinterval is longer than 1/MINSUBS times the
        % total path length. Removes subintervals of zero length, except
        % that the resulting X will always has at least two elements on
        % return, i.e., if the total path length is zero, X will be
        % collapsed into a single interval of zero length.  Also returns
        % the integration path length.
        absdx = abs(diff(x));
        if isreal(x)
            pathlen = x(end) - x(1);
        else
            pathlen = sum(absdx);
        end
        if pathlen > 0
            udelta = minsubs/pathlen;
            nnew = ceil(absdx*udelta) - 1;
            idxnew = find(nnew > 0);
            nnew = nnew(idxnew);
            for j = numel(idxnew):-1:1
                k = idxnew(j);
                nnj = nnew(j);
                % Calculate new points.
                newpts = x(k) + (1:nnj)./(nnj+1)*(x(k+1)-x(k));
                % Insert the new points.
                x = [x(1:k),newpts,x(k+1:end)];
            end
        end
        % Remove useless subintervals.
        x(abs(diff(x))==0) = [];
        if isscalar(x)
            % Return at least two elements.
            x = [x(1),x(1)];
        end
    end % split

%--------------------------------------------------------------------------

    function finalInputChecks(x,fx)
        % Do final input validation with sample input and outputs to the
        % integrand function.
        % Check classes.
        if ~(isfloat(x) && isfloat(fx))
            error('MATLAB:quadgk:UnsupportedClass', ...
                'Supported classes are ''double'' and ''single''.');
        end
        % Check sizes.
        if ~isequal(size(x),size(fx))
            error('MATLAB:quadgk:FxNotSameSizeAsX', ...
                'Output of the function must be the same size as the input.');
        end
        outcls = superiorfloat(x,fx);
        outdbl = strcmp(outcls,'double');
        % Validate tolerances and apply defaults.
        if isempty(RTOL)
            if outdbl
                RTOL = DEFAULT_DOUBLE_RELTOL;
            else
                RTOL = DEFAULT_SINGLE_RELTOL;
            end
        end
        if isempty(ATOL)
            if outdbl
                ATOL = DEFAULT_DOUBLE_ABSTOL;
            else
                ATOL = DEFAULT_SINGLE_ABSTOL;
            end
        end
        % Make sure that RTOL >= 100*eps(outcls) except when
        % using pure absolute error control (ATOL>0 && RTOL==0).
        if ~(ATOL > 0 && RTOL == 0) && RTOL < 100*eps(outcls)
            RTOL = 100*eps(outcls);
            warning('MATLAB:quadgk:increasedRelTol', ...
                'RelTol was increased to 100*eps(''%s'') = %g.',outcls,RTOL);
        end
        if outdbl
            % Single RTOL or ATOL should not force any single precision
            % computations.
            RTOL = double(RTOL);
            ATOL = double(ATOL);
        end
    end % finalInputChecks

%--------------------------------------------------------------------------

    function parseOptions(varargin)
        % Parse optional input arguments.
        k = 1;
        while k < nargin
            propname = lower(varargin{k});
            k = k + 1;
            propvalue = varargin{k};
            switch propname
                case 'reltol'
                    RTOL = propvalue;
                    if ~(isfloat(RTOL) && isscalar(RTOL) && ...
                            isreal(RTOL) && RTOL >= 0)
                        error('MATLAB:quadgk:invalidRelTol','Invalid RelTol');
                    end
                case 'abstol'
                    ATOL = propvalue;
                    if ~(isfloat(ATOL) && isscalar(ATOL) && ...
                            isreal(ATOL) && ATOL >= 0)
                        error('MATLAB:quadgk:invalidAbsTol','Invalid AbsTol');
                    end
                case 'waypoints'
                    WAYPOINTS = propvalue;
                    if ~(isvector(WAYPOINTS) || isequal(WAYPOINTS,[]))
                        error('MATLAB:quadgk:invalidWaypoints', ...
                            'Waypoints must be a vector.');
                    end
                    WAYPOINTS = WAYPOINTS(:).';
                case 'maxintervalcount'
                    MAXINTERVALCOUNT = propvalue;
                    if ~(isscalar(MAXINTERVALCOUNT) && ...
                            MAXINTERVALCOUNT > 0 && ...
                            floor(MAXINTERVALCOUNT) == MAXINTERVALCOUNT)
                        error('MATLAB:quadgk:invalidMaxIntervalCount', ...
                            'MaxIntervalCount must be a positive integer scalar.');
                    end
                otherwise
                    error('MATLAB:quadgk:unknownOption', 'Unrecognized option.');
            end
            k = k + 1;
        end
        if k == nargin
            error('MATLAB:quadgk:PVPairError', ...
                'Invalid option.  Expected option name followed by option value.');
        end
    end % parseOptions

%==========================================================================

end % quadgk
