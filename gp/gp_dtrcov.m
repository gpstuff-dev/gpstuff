function [K, C] = gp_dtrcov(gp, x1, x2,predcf)
%GP_DTRCOV  Evaluate training covariance matrix (gp_cov + noise covariance).
%
%  Description
%    K = GP_DTRCOV(GP, X1, XV, PREDCF) takes in Gaussian process GP and
%    matrix X1 that contains training input vectors to GP with XV that 
%    contains the virtual inputs. Returns the covariance matrix K between
%    elements of latent vector [f df/dx_1 df/dx_2, ..., df/dx_d] where f is
%    evaluated at X1 and df/dx_i at XV.
%
%    [K, C] = GP_DTRCOV(GP, TX, PREDCF) returns also the (noisy)
%    covariance matrix C for observations y, which is sum of K and
%    diagonal term, for example, from Gaussian noise.
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2006-2010 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if (isfield(gp,'derivobs') && gp.derivobs)
  ncf=length(gp.cf);
  [n,m]=size(x2);
  if isfield(gp, 'nvd')
    % Only specific dimensions
    ii1=abs(gp.nvd);
  else
    ii1=1:m;
  end
  K=zeros(length(x1)+length(ii1).*length(x2));
  % Loop over covariance functions
  for i=1:ncf
    % Derivative observations
    gpcf = gp.cf{i};           
    if m==1
      Kff = gpcf.fh.trcov(gpcf, x1);
      Gset = gpcf.fh.ginput4(gpcf, x2,x1);
      D = gpcf.fh.ginput2(gpcf, x2, x2);
      Kdf=Gset{1};
      Kfd = Kdf;
      Kdd=D{1};
      
      % Add all the matrices into a one K matrix
      K = K+[Kff Kfd'; Kfd Kdd];
      [a b] = size(K);
      
      % MULTIDIMENSIONAL input dim >1
    else
      Kff = gpcf.fh.trcov(gpcf, x1);
      if ~isequal(x2,x1)
        G = gpcf.fh.ginput4(gpcf, x2,x1);
      else
        G = gpcf.fh.ginput4(gpcf, x2);
      end
      D= gpcf.fh.ginput2(gpcf, x2, x2);
      Kdf2 = gpcf.fh.ginput3(gpcf, x2 ,x2);
      
      Kfd=cat(1,G{ii1});
%       Kfd=[G{1:m}];
      
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
          aa=zeros(1,m);
          t2=t1+m-2-(i-1);
          aa(1,i)=1;
          k=kron(aa,cat(1,zeros((i)*n,n),Kdf2{t1:t2}));
          %k(1:n*m,:)=[];
          k=k+k';
          Kddnodi = Kddnodi + k;
          t1=t2+1;
        end
      end
      % Sum the diag + no diag matrices
      Kdd=Kdd+Kddnodi;
      
      if isfield(gp, 'nvd')
        % Collect the monotonic dimensions
        Kddtmp=[];
        for ii2=1:length(ii1)
          for ii3=ii2:length(ii1)
            Kddtmp((ii2-1)*n+1:ii2*n, (ii3-1)*n+1:ii3*n) = ...
              Kdd((ii1(ii2)-1)*n+1:ii1(ii2)*n,(ii1(ii3)-1)*n+1:ii1(ii3)*n);
            if ii2~=ii3
              Kddtmp((ii3-1)*n+1:ii3*n, (ii2-1)*n+1:ii2*n) = ...
                Kdd((ii1(ii3)-1)*n+1:ii1(ii3)*n,(ii1(ii2)-1)*n+1:ii1(ii2)*n);
            end
          end
        end
        Kdd=Kddtmp;
      end
      
      % Gather all the matrices into one final matrix K which is the
      % training covariance matrix
      K = K+[Kff Kfd'; Kfd Kdd];
      [a b] = size(K);
    end    
  end  
  %add jitterSigma2 to the diagonal
  if ~isempty(gp.jitterSigma2)
    a1=a + 1;
    K(1:a1:end)=K(1:a1:end) + gp.jitterSigma2;
  end
  if nargout > 1
    C = K;
    if isfield(gp,'lik_mono') && isequal(gp.lik.type, 'Gaussian');
      % Add Gaussian noise to the obs part of covariance
      lik = gp.lik;
      Noi=lik.fh.trcov(lik, x1);
      Noi=[Noi zeros(size(Noi,1),size(K,2)-size(Noi,2)); ...
        zeros(size(K,1)-size(Noi,1),size(Noi,2)) ...
        zeros(size(K,1)-size(Noi,1),size(K,2)-size(Noi,2))];
      C=K+Noi;
%       x2=repmat(x1,m,1);
%       Cff = Kff + Noi;
%       C = [Cff Kfd'; Kfd Kdd];
    end
    if ~isempty(gp.jitterSigma2)
      C(1:a1:end)=C(1:a1:end) + gp.jitterSigma2;
    end
  end
end
