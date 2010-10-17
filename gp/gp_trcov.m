function [K, C] = gp_trcov(gp, x1, predcf)
%GP_TRCOV Evaluate training covariance matrix (gp_cov + noise covariance). 
%
%         Description
%          K = GP_TRCOV(GP, TX, PREDCF) takes in Gaussian process GP
%          and matrix TX that contains training input vectors to
%          GP. Returns noiseless covariance matrix K, which is formed
%          as a sum of the covariance matrices from covariance
%          functions in gp.cf array. Every element ij of K contains
%          covariance between inputs i and j in TX. PREDCF is an array
%          specifying the indexes of covariance functions, which are
%          used for forming the matrix. If not given, the matrix is
%          formed with all functions.
%
%          [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the noisy
%          covariance matrix C, which is sum of K and the covariance
%          functions is gp.noise array.

% Copyright (c) 2006-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% no covariance functions?
if length(gp.cf)==0
  K=[];
  C=[];
  return
end

% Are gradient observations available; gradobs=1->yes, gradobs=0->no
gradobs=isfield(gp,'grad_obs');

if gradobs==1
    gpcf = gp.cf{1};            % only for sexp at the moment
    [n,m]=size(x1);
    if m==1          
           Kff = feval(gpcf.fh_trcov, gpcf, x1); 
           Gset = feval(gpcf.fh_ginput4, gpcf, x1);
           D = feval(gpcf.fh_ginput2, gpcf, x1, x1);

           Kdf=Gset{1};
           Kfd = Kdf';
           Kdd=D{1};

           % Add all the matrices into a one K matrix
           K = [Kff Kfd; Kdf Kdd];
           [a b] = size(K);

    % MULTIDIMENSIONAL input dim >1
    else
           Kff = feval(gpcf.fh_trcov, gpcf, x1); 
           G= feval(gpcf.fh_ginput4, gpcf, x1);
           D= feval(gpcf.fh_ginput2, gpcf, x1, x1);
           Kdf2 = feval(gpcf.fh_ginput3, gpcf, x1 ,x1);

           Kdf=cat(1,G{1:m});

           % Now build up Kdd m*n x m*n matrix, which contains all the
           % both partial derivative" -matrices
           Kdd=blkdiag(D{1:m});

           % Gather non-diagonal matrices to Kddnodi
           if m==2
                Kddnodi=[zeros(n,n) Kdf2{1};Kdf2{1} zeros(n,n)];
            else
                t1=1;
                Kddnodi=zeros(m*n,m*n);
                for i=1:m-1
                    aa=zeros(m-1,m);
                    t2=t1+m-2-(i-1);
                    aa(m-1,i)=1;
                    k=kron(aa,cat(1,zeros((i)*n,n),Kdf2{t1:t2}));
                    k(1:n*m,:)=[];
                    k=k+k';
                    Kddnodi = Kddnodi + k;
                    t1=t2+1;
                end
           end
           % Sum the diag + no diag matrices
           Kdd=Kdd+Kddnodi;
           Kfd=Kdf';

           % Gather all the matrices into one final matrix K which is the 
           % training covariance matrix
           K = [Kff Kfd; Kdf Kdd];
           [a b] = size(K);
    end
    
    %add jitterSigma2 to the diagonal
    if ~isempty(gp.jitterSigma2)
         a1=a + 1;
         K(1:a1:end)=K(1:a1:end) + gp.jitterSigma2;
    end
    C = K;
    % Add noise to the covariance
    if nargout > 1
        % Add noise to the covariance
        if isfield(gp, 'noisef')
            nn = length(gp.noisef);
            if nn == 1
                    % same noise for obs and grad obs
                    noisef = gp.noisef{1};
                    Noi=feval(noisef.fh_trcov, noisef, x1);
                    x2=repmat(x1,m,1);
                    Noi2=feval(noisef.fh_trcov, noisef, x2);
                    Cff = Kff + Noi;
                    Cdd = Kdd + Noi2;
                    C = [Cff Kfd; Kdf Cdd];
            elseif nn>1
                Cff=Kff;
                Cdd=Kdd;
                   x2=repmat(x1,m,1);
                   for i=1:nn
                       noisef = gp.noisef{i};
                       Cff = Cff + feval(noisef.fh_trcov, noisef, x1);
                       Cdd = Cdd + feval(noisef.fh_trcov, noisef, x2);
                   end
                   C = [Cff Kfd; Kdf Cdd];
            end
        end
    end
    
else
    
    switch gp.type
      case {'FULL' 'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}
        [n,m]=size(x1);
        n1 = n+1;
        ncf = length(gp.cf);

        % Evaluate the covariance without noise
        K = sparse(0);
        if nargin < 3 || isempty(predcf)
            predcf = 1:ncf;
        end
        for i=1:length(predcf)
            gpcf = gp.cf{predcf(i)};
            K = K + feval(gpcf.fh_trcov, gpcf, x1);
        end

        if ~isempty(gp.jitterSigma2)
            if issparse(K)
                K = K + sparse(1:n,1:n,gp.jitterSigma2,n,n);
            else
                K(1:n1:end)=K(1:n1:end) + gp.jitterSigma2;
            end
        end
        if nargout > 1 
            C=K;
            % Add noise to the covariance
            if isfield(gp, 'noisef')
                nn = length(gp.noisef);
                for i=1:nn                
                    noisef = gp.noisef{i};
                    C = C + feval(noisef.fh_trcov, noisef, x1);
                end
            end
        end

      case 'SSGP'
        [n,m]=size(x1);
        n1 = n+1;
        ncf = length(gp.cf);

        % Evaluate the covariance without noise
        gpcf = gp.cf{1};
        K = feval(gpcf.fh_trcov, gpcf, x1);

        if nargout > 1 
            C=sparse(0);
            % Add noise to the covariance
            if isfield(gp, 'noisef')
                nn = length(gp.noisef);
                for i=1:nn
                    noisef = gp.noisef{i};
                    C = C + feval(noisef.fh_trcov, noisef, x1);
                end
            end
        end   
    end
end
