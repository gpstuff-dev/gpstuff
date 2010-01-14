function [m_0, m_1, m_2] = quad_moments(fun, a, b, varargin)
% quad_moments  Calculate the 0th, 1st and 2nd moment of a given probability distribution
%               
    
    % Set integration parameters.
    MAXINTERVALCOUNT = 650;
    
    if length(varargin) > 1
        RTOL = varargin{1};
    else
        RTOL = 1.e-6;
    end
    if length(varargin) > 2
        ATOL = varargin{2};
    else
        ATOL = 1.e-10;
    end

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
    
    % Integration interval
    tinterval = [a,b];
    
    % Compute the path length and split tinterval.
    minsubs = 10;   % Minimum number subintervals to start.
    absdx = abs(b-a);
    pathlen = absdx;
    if pathlen > 0
        udelta = minsubs/pathlen;
        nnew = ceil(absdx*udelta) - 1;
        idxnew = find(nnew > 0);
        nnew = nnew(idxnew);
        for j = numel(idxnew):-1:1
            k = idxnew(j);
            nnj = nnew(j);
            % Calculate new points.
            newpts = tinterval(k) + (1:nnj)./(nnj+1)*(tinterval(k+1)-tinterval(k));
            % Insert the new points.
            tinterval = [tinterval(1:k),newpts,tinterval(k+1:end)];
        end
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
        x = bsxfun(@plus,NODES*halfh,midpt);
        x = reshape(x,1,[]);   % function f expects a row vector
        fx = fun(x);
        fx1 = fx.*x;
        fx2 = fx.*x.^2;
        
        fx = reshape(fx,numel(WT),[]);
        fx1 = reshape(fx1,numel(WT),[]);
        fx2 = reshape(fx2,numel(WT),[]);
        % Quantities for subintervals.
        qsubs = (WT*fx) .* halfh;
        errsubs = (EWT*fx) .* halfh;
        qsubs1 = (WT*fx1) .* halfh;
        qsubs2 = (WT*fx2) .* halfh;
        % Calculate current values of q and tol.
        q = sum(qsubs) + q_ok;
        q1 = sum(qsubs1) + q1_ok;
        q2 = sum(qsubs2) + q2_ok;
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
        q1_ok = q1_ok + sum(qsubs1(ndx));
        q2_ok = q2_ok + sum(qsubs2(ndx));
        % Split the remaining subintervals in half. Quit if splitting
        % results in too many subintervals.
        nsubs = 2*size(subs,2);
        if nsubs > MAXINTERVALCOUNT
            warning('quadgk2:MaxIntervalCountReached', ...
                    ['Reached the limit on the maximum number of intervals in use.']);
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
