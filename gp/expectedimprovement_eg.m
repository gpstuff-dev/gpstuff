function [EI, EIg] = expectedimprovement_eg(x_new, gp, x, a, invC, fmin, varargin)
% expectedImprovement_eg    Calculate the negative expected improvement and
%                           its gradient for use in Bayesian optimization
%
%function [EI, EIg] = expectedImprovement_eg(x_new, gp, x, a, invC, fmin, const, const2,...)
% Arguments to the function are:
%   x_new  - query point 
%   gp     - GP model for the objective function
%   x      - previous query points for which we have calculated objective
%            function value y 
%   a      - a = C\y;  
%   invC   - [~, C] = gp_trcov(gp,x); invC = inv(C);
%   fmin   - The minimum of the objective function thus far
% Optional arguments are const, const2,... which are structures that
% include information concerning constraints. E.g.: 
%   const.gpc    - cell array of GP models for constraint functions
%   const.invCc  - [~, C{i}] = gp_trcov(gpc{i},x); invCc{i} = inv(C{i});
%                  i=1,...,numberOfConstraints
%   const.ac     - ac{i} = C{i}\y[:,i];
%   const.const  - (numberOfConstraints x 2) Matrix of constraints so that
%                  first column is the minimum and second the maximum
%                  constraint. 
%
% Known bugs: - inf or -inf are not allowed as maximum or minimum for
%               constraints. You need to use a large (small) value instead
%
% Copyright (c) 2015 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Calculate expected improvement if there are training points for it
if ~isempty(x)
    Knx = gp_cov(gp,x_new,x);
    Kn = gp_trvar(gp,x_new);
    Ef = Knx*a; Ef=Ef(1:size(x_new,1));
    invCKnxt = invC*Knx';
    Varf = Kn - sum(Knx.*invCKnxt',2); 
    Varf=max(Varf(1:size(x_new,1)),0);
    
    % expected improvement
    posvar=find(Varf>0);
    CDFpart = zeros(size(Varf));
    PDFpart = zeros(size(Varf));
    tmp = (fmin - Ef(posvar))./sqrt(Varf(posvar));
    CDFpart(posvar) = normcdf(tmp);
    CDFpart(~posvar) = (fmin - Ef(~posvar))>0;
    PDFpart(posvar) = normpdf(tmp);
    EI =  (fmin - Ef).*CDFpart + sqrt(Varf).*PDFpart;
else
    EI = 1;
end
EI = -EI;

% Constrain fullfilment probability
if nargin > 6
    constPr = [];
    constPrderEf = [];
    constPrderVarf = [];
    counter = 1;
    for k1 = 1:length(varargin)
        gpc = varargin{k1}.gpc;
        invCc = varargin{k1}.invCc;
        ac = varargin{k1}.ac;
        xc = varargin{k1}.xc;
        const = varargin{k1}.const;        
        if ~iscell(gpc)
            gpc= {gpc};
            invCc = {invCc};
        end
        for c1=1:length(gpc)
            Knxc = gp_cov(gpc{c1},x_new,xc);
            Knc = gp_trvar(gpc{c1},x_new);
            Efc = Knxc*ac(:,c1);
            invCKnxtc{counter} = invCc{c1}*Knxc';
            Varfc = Knc - sum(Knxc.*invCKnxtc{counter}',2);
            
            aa = (const(c1,2)-Efc)./sqrt(Varfc);
            bb = (const(c1,1)-Efc)./sqrt(Varfc);
            iapu = aa>5 & bb>5; % handle the cases where the numerical accuracy of normcdf is not enough
            constPr(iapu,counter) = (normcdf( -bb(iapu) ) -  normcdf( -aa(iapu) ) );
            constPr(~iapu,counter) = (normcdf( aa(~iapu) ) -  normcdf( bb(~iapu) ) );
%             if aa>5 & bb>5 
%                 constPr(:,counter) = (normcdf( -bb ) -  normcdf( -aa ) );
%             else
%                 constPr(:,counter) = (normcdf( aa ) -  normcdf( bb ) );
%             end
            if nargout>1
                constPrderEf(:,counter) = - ( normpdf( aa ) -  normpdf( bb ) )./sqrt(Varfc);
                constPrderVarf(:,counter) = - (normpdf( aa ).*(const(c1,2)-Efc) ...
                    -  normpdf( bb ).*(const(c1,1)-Efc) )./(2*sqrt(Varfc).^3);
            end
            counter = counter+1;
        end
    end
    EI = EI.*prod(constPr,2);
end

% gradients
if nargout>1 
    if size(x_new,1)>1
        error('expectedImprovement_eg: gradients are returned only for single query point.')
    end
    
    % Calculate Derivative of the expected improvement if there are training points for it
    if ~isempty(x)        
        % derivative of covariance matrix wrt. x
        Kderiv = zeros(length(x_new),1);
        Knxderiv = zeros(length(x_new),size(x,1));
        for i1 = 1:length(gp.cf)
            gpcf = gp.cf{i1};
            DK = gpcf.fh.ginput(gpcf, x_new);
            DKnx = gpcf.fh.ginput(gpcf, x_new, x);
            for j1=1:length(x_new)
                Kderiv(j1) = Kderiv(j1) + DK{j1};
                Knxderiv(j1,:) = Knxderiv(j1,:) + DKnx{j1};
            end
        end
        
        % derivative of EI wrt. Ef and Varf
        dEIdEf = - CDFpart;
        dEIdVarf = PDFpart/(2*sqrt(Varf));
        
        % Derivative of Ef and Varf wrt. x
        dEfdx = Knxderiv(1:size(x_new,1),:)*a(1:size(x,1));
        dVarfdx = Kderiv - (Knxderiv(1:size(x_new,1),:)*invCKnxt(1:size(x,1),1:size(x_new,1)) + (invCKnxt(1:size(x,1),1:size(x_new,1))'*Knxderiv(1:size(x_new,1),:)')');
        
        % Derivative of EI
        EIg = -( dEIdEf*dEfdx + dEIdVarf*dVarfdx )';
    else
        EIg = 0;
    end
    
    % Derivative of the constrained probability
    if nargin > 6
        constg = 0;
        counter = 1;
        for k1 = 1:length(varargin)
            gpc = varargin{k1}.gpc;
            invCc = varargin{k1}.invCc;
            ac = varargin{k1}.ac;
            const = varargin{k1}.const;
            xc = varargin{k1}.xc;
            if ~iscell(gpc)
                gpc= {gpc};
                invCc = {invCc};
            end
            for c1=1:length(gpc)
                gpct = gpc{c1};
                Kderiv = zeros(length(x_new),1);
                Knxderiv = zeros(length(x_new),size(xc,1));
                for i1 = 1:length(gpct.cf)
                    gpcf = gpct.cf{i1};
                    DK = gpcf.fh.ginput(gpcf, x_new);
                    DKnx = gpcf.fh.ginput(gpcf, x_new, xc);
                    for j1=1:length(x_new)
                        Kderiv(j1) = Kderiv(j1) + DK{j1};
                        Knxderiv(j1,:) = Knxderiv(j1,:) + DKnx{j1};
                    end
                end
                % Derivative of Ef and Varf wrt. x
                dEfdx = Knxderiv*ac(:,c1);
                dVarfdx = Kderiv - (Knxderiv*invCKnxtc{counter} + (invCKnxtc{counter}'*Knxderiv')');
                
                %dVarfdx = 0;
                
                constg = constg + ( constPrderEf(:,counter)*dEfdx + constPrderVarf(:,counter)*dVarfdx )'./constPr(counter);
                counter = counter+1;
            end
        end
        
        EIg = EIg*prod(constPr) + EI*constg;
    end
end