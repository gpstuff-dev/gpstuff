function [C, Cinv] = gp_dcov(gp, x1, x2, predcf)
%GP_COV  Evaluate covariance matrix between two input vectors.
%
%  Description
%    C = GPCOV(GP, TX, X, PREDCF) takes in Gaussian process GP and
%    two matrixes TX and X that contain input vectors to GP.
%    Returns covariance matrix C. Every element ij of C contains
%    covariance between inputs i in TX and j in X. PREDCF is an
%    optional array specifying the indexes of covariance functions,
%    which are used for forming the matrix. If empty or not given,
%    the matrix is formed with all functions.

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Are gradient observations available; derivobs=1->yes, derivobs=0->no
Cinv=[];
% Split the training data for normal latent input and gradient inputs
x12=x1;
x11=gp.xv;

% Derivative observations
[n,m]=size(x1);
[n4,m4]=size(x2);
ncf=length(gp.cf);
for i1=1:ncf
  gpcf = gp.cf{i1};    % Grad obs implemented only to sexp currently
  if m==1
    Gset1 = gpcf.fh.ginput4(gpcf, x11, x2);
    Gset2 = gpcf.fh.ginput4(gpcf, x2, x12);
    Kff = gpcf.fh.cov(gpcf, x12, x2);
    Kdd = gpcf.fh.ginput2(gpcf, x11, x2);
    
    Kdf=Gset1{1};
    Kfd=Gset2{1};
    %Kfd = -1.*Kdf;
    C = [Kff Kfd'; Kdf Kdd{1}];
    
    % Input dimension is >1
  else
    [n,m]=size(x11);
    [n2,m2]=size(x2);
    
    Kff = gpcf.fh.cov(gpcf, x12, x2);
    Gset1 = gpcf.fh.ginput4(gpcf, x11,x2);
    Gset2 = gpcf.fh.ginput4(gpcf, x2, x12);
    
    %Gather matrices from Gset (d k(x1,x2) /d x1)
    Kfd=cat(2,Gset1{1:m});
    Kdf=cat(1,Gset1{1:m});
    Kfd22=cat(2,Gset2{1:m});
    Kdf22=cat(1,Gset2{1:m})';
    %   Kfd=-1*Kfd;
    %   Kfd2=-1*Kfd2;
    
    % both x derivatives, same dimension (to diagonal blocks)
    D = gpcf.fh.ginput2(gpcf, x11, x2);
    % both x derivatives, different dimension (non-diagonal blocks)
    Kdf2 = gpcf.fh.ginput3(gpcf, x11 ,x2);
    
    % Now build up Kdd m*n x m*n2 matrix, which contains all the
    % both partial derivative" -matrices
    
    % Add the diagonal matrices
    Kdd=blkdiag(D{1:m});
    % Add the non-diagonal matrices to Kdd
    ii3=0;
    for j=0:m-2
      for i=1+j:m-1
        ii3=ii3+1;
        Kdd(i*n+1:(i+1)*n,j*n2+1:j*n2+n2) = Kdf2{ii3};
        Kdd(j*n+1:j*n+n,i*n2+1:(i+1)*n2) = Kdf2{ii3};
      end
    end
    
    % Gather all the matrices into one final matrix K which is the
    % training covariance matrix
    C = [Kff Kdf22; Kdf Kdd];
    %   C = [Kff; Kdf];
  end
  if i1==1
    CC=C;
  else
    CC=CC+C;
  end
end
C=CC;
