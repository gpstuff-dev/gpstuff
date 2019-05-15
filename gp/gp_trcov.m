function [K, C] = gp_trcov(gp, x1, predcf)
%GP_TRCOV  Evaluate training covariance matrix (gp_cov + noise covariance).
%
%  Description
%    K = GP_TRCOV(GP, TX, PREDCF) takes in Gaussian process GP and
%    matrix TX that contains training input vectors to GP. Returns
%    (noiseless) covariance matrix K for latent values, which is
%    formed as a sum of the covariance matrices from covariance
%    functions in gp.cf array. Every element ij of K contains
%    covariance between inputs i and j in TX. PREDCF is an array
%    specifying the indexes of covariance functions, which are used
%    for forming the matrix. If not given, the matrix is formed
%    with all functions.
%
%    [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the (noisy)
%    covariance matrix C for observations y, which is sum of K and
%    diagonal term, for example, from Gaussian noise.
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2006-2010, 2016 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% no covariance functions?
if length(gp.cf)==0 || (nargin>2 && ~isempty(predcf) && predcf(1)==0) 
    K=[];
    C=[];
    if nargout>1 && isfield(gp.lik.fh,'trcov')
        C=sparse(0);
        % Add Gaussian noise to the covariance
        C = C + gp.lik.fh.trcov(gp.lik, x1);
        if ~isempty(gp.jitterSigma2)
            C=C+gp.jitterSigma2;
        end
    end
    return
end

[n,m]=size(x1);
ncf = length(gp.cf);
% Evaluate the covariance without noise
K = sparse(n,n);
if isfield(gp,'deriv') && gp.deriv  % derivative observations in use
    ind_Ddim = x1(:,gp.deriv);
    ind_Ddim_derivs = ind_Ddim(ind_Ddim>0);
    uDdim = unique(ind_Ddim_derivs);
    x1 = x1(:,setdiff(1:m,gp.deriv));   % Take only the non-index columns
    if any(strcmp(gp.type,{'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}))
        error('derivative observations have not been implemented for sparse GPs')
    end
end
% check whether predcf is used
if nargin < 3 || isempty(predcf)
    predcf = 1:ncf;
end
% loop through covariance functions
for i=1:length(predcf)
    gpcf = gp.cf{predcf(i)};
    if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
        if ~isfield(gp,'comp_cf') || (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==predcf(i)))
            gpcf.magnSigma2=1;
        end
    end
    
    % derivative observations in use
    if isfield(gp,'deriv') && gp.deriv
        if (~isfield(gpcf, 'selectedVariables') || any(ismember(gpcf.selectedVariables,uDdim)))
            % !!! Note. The check whether to calculate derivative
            % matrices for a covariance function could/should be made
            % nicer
            if size(x1,2) <2    % One dimensional input
                Kff = gpcf.fh.trcov(gpcf, x1(ind_Ddim==0,:));
                Kdf = gpcf.fh.ginput4(gpcf, x1(ind_Ddim==1,:), x1(ind_Ddim==0,:));
                D = gpcf.fh.ginput2(gpcf, x1(ind_Ddim==1,:), x1(ind_Ddim==1,:));
                
                Kdf=Kdf{1};
                Kfd = Kdf';
                Kdd=D{1};
                % Add all the matrices into a one K matrix
                % K = K + [Kff Kfd; Kdf Kdd];
                Ktemp = [Kff Kfd; Kdf Kdd];
            else
                Ktemp = [];
                % the block of covariance matrix
                Ktemp(ind_Ddim==0,ind_Ddim==0) = gpcf.fh.trcov(gpcf, x1(ind_Ddim==0,:));
                for u1 = 1:length(uDdim)
                    % the blocks on the left side, below Kff
                    if sum(ind_Ddim==0)>0
                        Kdf = gpcf.fh.ginput4(gpcf, x1(ind_Ddim==uDdim(u1),:), x1(ind_Ddim==0,:), uDdim(u1));
                        Ktemp(ind_Ddim==uDdim(u1),ind_Ddim==0) = Kdf{1};
                        Ktemp(ind_Ddim==0,ind_Ddim==uDdim(u1)) = Kdf{1}';
                    end
                    D = gpcf.fh.ginput2(gpcf, x1(ind_Ddim==uDdim(u1),:), x1(ind_Ddim==uDdim(u1),:), uDdim(u1));
                    Ktemp(ind_Ddim==uDdim(u1),ind_Ddim==uDdim(u1)) = D{1};
                    
                    uDdim2 = uDdim(u1+1:end);
                    for u2=1:length(uDdim2)
                        Kdf2 = gpcf.fh.ginput3(gpcf, x1(ind_Ddim==uDdim(u1),:) ,x1(ind_Ddim==uDdim2(u2),:), uDdim(u1), uDdim2(u2));
                        Ktemp(ind_Ddim==uDdim(u1),ind_Ddim==uDdim2(u2)) = Kdf2{1};
                        Ktemp(ind_Ddim==uDdim2(u2),ind_Ddim==uDdim(u1)) = Kdf2{1}';
                    end
                    
                end
            end
        else
            Ktemp = zeros(n,n);
            Ktemp(ind_Ddim==0,ind_Ddim==0) = gpcf.fh.trcov(gpcf, x1(ind_Ddim==0,:));
        end
        K= K+ Ktemp;                
    else
        % Regular GP without derivative observations
        K = K + gpcf.fh.trcov(gpcf, x1);
    end
end

n = size(K,1);
n1 = n+1;
if ~isempty(gp.jitterSigma2)
    if issparse(K)
        K = K + sparse(1:n,1:n,gp.jitterSigma2,n,n);
    else
        K(1:n1:end)=K(1:n1:end) + gp.jitterSigma2;
    end
end
if nargout>1
    C=K;
    if isfield(gp.lik.fh,'trcov')
        C = C + gp.lik.fh.trcov(gp.lik, x1);
    end
end

end

