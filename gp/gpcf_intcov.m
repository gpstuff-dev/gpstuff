function gpcf = gpcf_intcov(varargin)
%GPCF_INTCOV    Create an integrated covariance function 
%
%  Description
%    GPCF = GPCF_INTCOV('nin',nin,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates an integrated covariance function structure in which the named
%    parameters have the specified values. Any unspecified parameters are
%    set to default values. Obligatory parameters are 'intArea', which
%    defines the integration areas, and 'cf', which defines the covariance
%    function(s) to be integrated.
%
%    Notes of usage:
%
%    The input matrix X can contain point locations and "lower-left"
%    corners of integrated areas (areas are always intervals, cells, cubes
%    etc.). Last column of the input X has to contain 1:s for integration
%    areas and 0 for point location. For example, if x(3,end) = 1, then the
%    third row of x tells the lower-left corner of an integrated area.
%
%    Field gpcf.intArea tells the lengths of the integration path along
%    each axis.
% 
%    GPCF = GPCF_INTCOV(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for piece wise polynomial (q=2) covariance function [default]
%      IntArea           - Integration path lengths per input dimension.
%      cf                - covariance functions to be integrated
%      NintPoints        - number of samples for areal integration
%                          approximation 
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    NOTES, WARNINGS! 
%    * This function is still experimental. It should return correct
%    answers in 1 and 2 dimensional problems
%    * Evaluation of the integrated covariance is currently implemented
%    using stochastic integration. This is awfully slow 
%      -> in 1d speed up could be obtained using quadrature
%      -> in 2d and higher dimensions speed up is perhaps possible with
%      extensions of quadrature
%      -> Quasi-Monte Carlo has been tried and helps only little
%    * Stochastic integration is highly variable
%      -> gradients can not be evaluated accurately enough for which reason
%      the inference has to be conducted with MCMC (HMC can not be used)
%   
%  See also
%    GP_SET, GPCF_*, PRIOR_*, METRIC_*
  
% Copyright (c) 2007-2011 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  if nargin>0 && ischar(varargin{1}) && ismember(varargin{1},{'init' 'set'})
    % remove init and set
    varargin(1)=[];
  end
  
  ip=inputParser;
  ip.FunctionName = 'GPCF_INTCOV';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('intArea',[], @(x) isvector(x) && all(x>0));
  ip.addParamValue('cf',[], @iscell);
  ip.addParamValue('NintPoints',200, @(x) isscalar(x) && x>0);
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    % Check that SuiteSparse is available
    init=true;
    gpcf.intArea=ip.Results.intArea;
    if isempty(gpcf.intArea)
      error('intArea has to be given for intcov: gpcf_intcov(''intArea'',INTAREA,...)')
    end
    gpcf.type = 'gpcf_intcov';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_intcov')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  
  if init || ~ismember('cf',ip.UsingDefaults)
    % Initialize parameters
    gpcf.cf = {};
    cfs=ip.Results.cf;
    if ~isempty(cfs)
      for i = 1:length(cfs)
        gpcf.cf{i} = cfs{i};
      end
    else
      error('At least one covariance function has to be given in cf');
    end
  end
  
  if init
    % Set the function handles to the nested functions
    gpcf.fh.pak = @gpcf_intcov_pak;
    gpcf.fh.unpak = @gpcf_intcov_unpak;
    gpcf.fh.lp = @gpcf_intcov_lp;
    gpcf.fh.lpg = @gpcf_intcov_lpg;
    gpcf.fh.cfg = @gpcf_intcov_cfg;
    gpcf.fh.ginput = @gpcf_intcov_ginput;
    gpcf.fh.cov = @gpcf_intcov_cov;
    gpcf.fh.trcov  = @gpcf_intcov_trcov;
    gpcf.fh.trvar  = @gpcf_intcov_trvar;
    gpcf.fh.recappend = @gpcf_intcov_recappend;
    
    % help parameters for storing intermediate results
    w0 = [];
    w1 = [];
    datahash0=0;
    datahash1=0;
    Ctrcov = [];
    Ccov = [];
  end

  % Initialize parameters
  if init || ~ismember('intArea',ip.UsingDefaults)
    gpcf.intArea=ip.Results.intArea;
  end
  
  % Initialize parameters
  if init || ~ismember('NintPoints',ip.UsingDefaults)
    gpcf.NintPoints=ip.Results.NintPoints;
  end
  
  function [w,s] = gpcf_intcov_pak(gpcf)
  %GPCF_INTCOV_PAK  Combine GP covariance function parameters into
  %                one vector
  %
  %  Description
  %    W = GPCF_INTCOV_PAK(GPCF) takes a covariance function
  %    structure GPCF and combines the covariance function
  %    parameters and their hyperparameters into a single row
  %    vector W.
  %
  %  See also
  %    GPCF_INTCOV_UNPAK

    ncf = length(gpcf.cf);
    w = []; s = {};
    
    for i=1:ncf
      cf = gpcf.cf{i};
      [wi si] = cf.fh.pak(cf);
      w = [w wi];
      s = [s; si];
    end

  end

  function [gpcf, w] = gpcf_intcov_unpak(gpcf, w)
  %GPCF_INTCOV_UNPAK  Sets the covariance function parameters into
  %                  the structure
  %
  %  Description
  %    [GPCF, W] = GPCF_INTCOV_UNPAK(GPCF, W) takes a covariance
  %    function structure GPCF and a hyper-parameter vector W,
  %    and returns a covariance function structure identical
  %    to the input, except that the covariance hyper-parameters
  %    have been set to the values in W. Deletes the values set to
  %    GPCF from W and returns the modified W.
  %
  %    Assignment is inverse of  
  %       w = [ log(gpcf.magnSigma2)
  %             (hyperparameters of gpcf.magnSigma2)
  %             log(gpcf.lengthScale(:))
  %             (hyperparameters of gpcf.lengthScale)]'
  %
  %  See also
  %    GPCF_INTCOV_PAK

    ncf = length(gpcf.cf);
    
    for i=1:ncf
      cf = gpcf.cf{i};
      [cf, w] = cf.fh.unpak(cf, w);
      gpcf.cf{i} = cf;
    end
    
  end

  function lp = gpcf_intcov_lp(gpcf)
  %GPCF_INTCOV_LP  Evaluate the log prior of covariance function parameters
  %
  %  Description
  %    LP = GPCF_INTCOV_LP(GPCF, X, T) takes a covariance function
  %    structure GPCF and returns log(p(th)), where th collects the
  %    parameters.
  %
  %  See also
  %    GPCF_INTCOV_PAK, GPCF_INTCOV_UNPAK, GPCF_INTCOV_LPG, GP_E

    lp = 0;
    ncf = length(gpcf.cf);
    for i=1:ncf
      cf = gpcf.cf{i};
      lp = lp + cf.fh.lp(cf);
    end
  end

  function lpg = gpcf_intcov_lpg(gpcf)
  %GPCF_INTCOV_LPG  Evaluate gradient of the log prior with respect
  %               to the parameters.
  %
  %  Description
  %    LPG = GPCF_INTCOV_LPG(GPCF) takes a covariance function
  %    structure GPCF and returns LPG = d log (p(th))/dth, where th
  %    is the vector of parameters.
  %
  %  See also
  %    GPCF_INTCOV_PAK, GPCF_INTCOV_UNPAK, GPCF_INTCOV_LP, GP_G

    lpg = [];
    ncf = length(gpcf.cf);
      
    % Evaluate the gradients
    for i=1:ncf
      cf = gpcf.cf{i};
      lpg=[lpg cf.fh.lpg(cf)];
    end
  end
  
  function DKff = gpcf_intcov_cfg(gpcf, x, x2, mask)
  %GPCF_INTCOV_CFG  Evaluate gradient of covariance function
  %                with respect to the parameters
  %
  %  Description
  %    DKff = GPCF_INTCOV_CFG(GPCF, X) takes a covariance function
  %    structure GPCF, a matrix X of input vectors and returns
  %    DKff, the gradients of covariance matrix Kff = k(X,X) with
  %    respect to th (cell array with matrix elements).
  %
  %    DKff = GPCF_INTCOV_CFG(GPCF, X, X2) takes a covariance
  %    function structure GPCF, a matrix X of input vectors and
  %    returns DKff, the gradients of covariance matrix Kff =
  %    k(X,X2) with respect to th (cell array with matrix
  %    elements).
  %
  %    DKff = GPCF_INTCOV_CFG(GPCF, X, [], MASK) takes a covariance
  %    function structure GPCF, a matrix X of input vectors and
  %    returns DKff, the diagonal of gradients of covariance matrix
  %    Kff = k(X,X2) with respect to th (cell array with matrix
  %    elements). This is needed for example with FIC sparse
  %    approximation.
  %
  %  See also
  %   GPCF_INTCOV_PAK, GPCF_INTCOV_UNPAK, GPCF_INTCOV_LP, GP_G

    [n, m] =size(x);

    DKff = {};
    % Evaluate: DKff{1} = d Kff / d magnSigma2
    %           DKff{2} = d Kff / d lengthScale
    % NOTE! Here we have already taken into account that the parameters are transformed
    % through log() and thus dK/dlog(p) = p * dK/dp

    % evaluate the gradient for training covariance
    if nargin == 2
                
        [n1,m1]=size(x);
        
        intInd1 = find(x(:,end)==1);
        pointInd1 = find(x(:,end)==0);
        ncf = length(gpcf.cf);
        numPoints = gpcf.NintPoints;
        intArea = repmat(gpcf.intArea,numPoints,1);
        
        % point-point covariance
        for i1=1:ncf
            cf = gpcf.cf{i1};
            temp = cf.fh.cfg(cf, x(pointInd1,1:end-1));
            for j1 = 1:length(temp)
                [I,J,R] = find(temp{j1});
                DKff{end+1} = sparse(pointInd1(I),pointInd1(J),R,n1,n1);
            end
        end
        
        % point-area covariance
        temp2={};
        for j1=1:length(intInd1)
            intpoints = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
            ii1=1;
            for i1=1:ncf
                cf = gpcf.cf{i1};
                temp = cf.fh.cfg(cf, x(pointInd1,1:end-1),intpoints);
                for k1 = 1:length(temp)
                    temp2{ii1}(:,j1) = mean(temp{k1},2);    
                    ii1=ii1+1;
                end
            end
        end
        for i1=1:length(temp2)
            [I,J,R] = find(temp2{i1});
            temp = sparse(pointInd1(I),intInd1(J),R,n1,n1);
            DKff{i1} = DKff{i1} + temp + temp';
        end
        
        % area-area covariance
        temp2={};
        for j1=1:length(intInd1)
            intpoints1 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
            for k1 = 1:length(intInd1)
                intpoints2 = repmat(x(intInd1(k1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
                ii1=1;
                for i1=1:ncf
                    cf = gpcf.cf{i1};
                    temp = cf.fh.cfg(cf, intpoints1, intpoints2);
                    for l1 = 1:length(temp)
                        temp2{ii1}(j1,k1) = mean(mean(temp{l1}));
                        ii1=ii1+1;
                    end
                end
            end
        end
        for i1=1:length(temp2)
            [I,J,R] = find(temp2{i1});
            temp = sparse(intInd1(I),intInd1(J),R,n1,n1);
            DKff{i1} = DKff{i1} + temp; % + temp'
        end
      
      % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
    elseif nargin == 3
      if size(x,2) ~= size(x2,2)
        error('gpcf_intcov -> _ghyper: The number of columns in x and x2 has to be the same. ')
      end
      
      % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
      %           DKff{2...} = d mask(Kff,I) / d lengthScale
    elseif nargin == 4
      
    end
    
    % check if CS covariances are used. If not change C into full matrix
    % for speed up
    sp = false;
    for i1=1:ncf
        if isfield(gpcf.cf{i1}, 'cs') && gpcf.cf{i1} == 1
            sp=true;
        end
    end
    if ~sp
        for i1=1:length(DKff)
            DKff{i1}=full(DKff{i1});
        end
    end

  end
  
  function DKff = gpcf_intcov_ginput(gpcf, x, x2)
  %GPCF_INTCOV_GINPUT  Evaluate gradient of covariance function with 
  %                   respect to x
  %
  %  Description
  %    DKff = GPCF_INTCOV_GINPUT(GPCF, X) takes a covariance
  %    function structure GPCF, a matrix X of input vectors and
  %    returns DKff, the gradients of covariance matrix Kff =
  %    k(X,X) with respect to X (cell array with matrix elements).
  %
  %    DKff = GPCF_INTCOV_GINPUT(GPCF, X, X2) takes a covariance
  %    function structure GPCF, a matrix X of input vectors and
  %    returns DKff, the gradients of covariance matrix Kff =
  %    k(X,X2) with respect to X (cell array with matrix elements).
  %
  %  See also
  %    GPCF_INTCOV_PAK, GPCF_INTCOV_UNPAK, GPCF_INTCOV_LP, GP_G

  end
  
  
  function C = gpcf_intcov_cov(gpcf, x1, x2, varargin)
  %GP_INTCOV_COV  Evaluate covariance matrix between two input vectors
  %
  %  Description         
  %    C = GP_INTCOV_COV(GP, TX, X) takes in covariance function of
  %    a Gaussian process GP and two matrixes TX and X that contain
  %    input vectors to GP. Returns covariance matrix C. Every
  %    element ij of C contains covariance between inputs i in TX
  %    and j in X.
  %
  %  See also
  %    GPCF_INTCOV_TRCOV, GPCF_INTCOV_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
      x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);
  
  if m1~=m2
      error('the number of columns of X1 and X2 has to be same')
  end
  
  ncf = length(gpcf.cf);
  ww = [];
  for i1=1:ncf
      ww = [ww gpcf.cf{i1}.fh.pak(gpcf.cf{i1})];
  end
  datahash=hash_sha512([x1, x2]);
  fromMem = false;
  if ~isempty(w1)
      if all(size(ww)==size(w1)) && all(abs(ww-w1)<1e-8) && isequal(datahash,datahash1)
          fromMem = true;
      end
  end
  
  if fromMem
      C = Ccov;
  else
      % RandStream.setDefaultStream(RandStream('mt19937ar','seed',100))
      intInd1 = find(x1(:,end)==1);
      intInd2 = find(x2(:,end)==1);
      pointInd1 = find(x1(:,end)==0);
      pointInd2 = find(x2(:,end)==0);
      numPoints = gpcf.NintPoints;
      intArea = repmat(gpcf.intArea,numPoints,1);
      dimInt = numel(gpcf.intArea);
      C = sparse(n1,n2);
      
      % point-point covariance
      if any(x1(:,end)==0) && any(x2(:,end)==0)
          temp=sparse(0);
          for i1=1:ncf
              cf = gpcf.cf{i1};
              temp = temp + cf.fh.cov(cf, x1(pointInd1,1:end-1),x2(pointInd2,1:end-1));
          end
          [I,J,R] = find(temp);
          C = sparse(pointInd1(I),pointInd2(J),R,n1,n2);
      end
      
      % point-area covariance
      if any(x1(:,end)==0) && any(x2(:,end)==1)
          temp=sparse(length(pointInd1),length(intInd2));
          for j1=1:length(intInd2)
              %N=600;
              %[tmp1, tmp2] = meshgrid( linspace(x2(intInd2(j1),1),x2(intInd2(j1),1)+intArea(1,1),N) , linspace(x2(intInd2(j1),2),x2(intInd2(j1),2)+intArea(1,2),N));
              %intpoints = [tmp1(:),tmp2(:)];
              intpoints = repmat(x2(intInd2(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,m2-1);
              %intpoints = repmat(x2(intInd2(j1),1:end-1),numPoints,1) + intArea.*hammersley(m2-1,numPoints)';
              for i1=1:ncf
                  cf = gpcf.cf{i1};
                  temp(:,j1) = temp(:,j1) + mean(cf.fh.cov(cf, x1(pointInd1,1:end-1),intpoints),2);
              end
          end
          [I,J,R] = find(temp);
          C = C + sparse(pointInd1(I),intInd2(J),R,n1,n2);
      end
      
      % area-point covariance
      if any(x1(:,end)==1) && any(x2(:,end)==0)
          temp=sparse(length(pointInd2),length(intInd1));
          for j1=1:length(intInd1)
              intpoints = repmat(x1(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
              for i1=1:ncf
                  cf = gpcf.cf{i1};
                  temp(:,j1) = temp(:,j1) + mean(cf.fh.cov(cf, x2(pointInd2,1:dimInt),intpoints),2);
              end
          end
          [I,J,R] = find(temp');
          C = C + sparse(intInd1(I),pointInd2(J),R,n1,n2);
      end
      
      % area-area covariance
      if any(x1(:,end)==1) && any(x2(:,end)==1)
          temp=sparse(length(intInd1),length(intInd2));
          for j1=1:length(intInd1)
              intpoints1 = repmat(x1(intInd1(j1),1:dimInt),numPoints,1) + intArea.*rand(numPoints,dimInt);
              for k1 = 1:length(intInd2)
                  intpoints2 = repmat(x2(intInd2(k1),1:dimInt),numPoints,1) + intArea.*rand(numPoints,dimInt);
                  for i1=1:ncf
                      cf = gpcf.cf{i1};
                      temp(j1,k1) = temp(j1,k1) + mean(mean(cf.fh.cov(cf, intpoints1, intpoints2)));
                  end
              end
          end
          [I,J,R] = find(temp);
          C = C + sparse(intInd1(I),intInd2(J),R,n1,n2);
      end
      
      % check if CS covariances are used. If not change C into full matrix
      % for speed up
      sp = false;
      for i1=1:ncf
          if isfield(gpcf.cf{i1}, 'cs') && gpcf.cf{i1} == 1
              sp=true;
          end
      end
      if ~sp
          C=full(C);
      end
      
      % store in the memory
      Ccov=C;
      datahash1=datahash;
      w1=ww;
  end
  
  end

  function C = gpcf_intcov_trcov(gpcf, x)
  %GP_INTCOV_TRCOV  Evaluate training covariance matrix of inputs
  %
  %  Description
  %    C = GP_INTCOV_TRCOV(GP, TX) takes in covariance function of a
  %    Gaussian process GP and matrix TX that contains training
  %    input vectors. Returns covariance matrix C. Every element ij
  %    of C contains covariance between inputs i and j in TX.
  %
  %  See also
  %    GPCF_INTCOV_COV, GPCF_INTCOV_TRVAR, GP_COV, GP_TRCOV

    ncf = length(gpcf.cf);
    ww=[];
    for i1=1:ncf
        ww = [ww gpcf.cf{i1}.fh.pak(gpcf.cf{i1})];
    end
    datahash=hash_sha512(x);
    fromMem = false;
    if ~isempty(w0)
        if all(size(ww)==size(w0)) && all(abs(ww-w0)<1e-8) && isequal(datahash,datahash0)
            fromMem = true;
        end
    end
    
    if fromMem
        C = Ctrcov;
    else
        [n1,m1]=size(x);
        
        intInd1 = find(x(:,end)==1);
        pointInd1 = find(x(:,end)==0);
        numPoints = gpcf.NintPoints;
        intArea = repmat(gpcf.intArea,numPoints,1);
        dimInt = numel(gpcf.intArea);
        %intMethod = gpcf.intMethod;
        
        % point-point covariance
        temp=sparse(0);
        for i1=1:ncf
            cf = gpcf.cf{i1};
            temp = temp + cf.fh.trcov(cf, x(pointInd1,1:end-1));
        end
        [I,J,R] = find(temp);
        C = sparse(pointInd1(I),pointInd1(J),R,n1,n1);
        
        % point-area covariance
        temp=sparse(length(pointInd1),length(intInd1));
        randpoints = intArea.*hammersley(dimInt,numPoints)';
        for j1=1:length(intInd1)
            intpoints = repmat(x(intInd1(j1),1:dimInt),numPoints,1) + randpoints;
            %intpoints = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
            for i1=1:ncf
                cf = gpcf.cf{i1};
                temp(:,j1) = temp(:,j1) + mean(cf.fh.cov(cf, x(pointInd1,1:dimInt),intpoints),2);
            end
        end
        [I,J,R] = find(temp);
        temp = sparse(pointInd1(I),intInd1(J),R,n1,n1);
        C = C + temp + temp';
        
        %     % area-area covariance
        %     temp=sparse(length(intInd1),length(intInd1));
        %     for j1=1:length(intInd1)
        %         RandStream.setDefaultStream(RandStream('mt19937ar','seed',100))
        %         intpoints1 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
        %         for k1 = 1:length(intInd1)
        %             RandStream.setDefaultStream(RandStream('mt19937ar','seed',100))
        %             intpoints2 = repmat(x(intInd1(k1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
        %             for i1=1:ncf
        %                 cf = gpcf.cf{i1};
        %                 temp(j1,k1) = temp(j1,k1) + mean(mean(feval(cf.fh.cov, cf, intpoints1, intpoints2)));
        %             end
        %         end
        %     end
        %     [I,J,R] = find(temp);
        %     C = C + sparse(intInd1(I),intInd1(J),R,n1,n1);
        
        % area-area covariance
        temp=sparse(length(intInd1),length(intInd1));
        temp2=zeros(n1,1);
        randpoints = [intArea intArea].*hammersley((dimInt)*2,numPoints)';
        for j1=1:length(intInd1)
            intpoints1 = repmat(x(intInd1(j1),1:dimInt),numPoints,1) + randpoints(:,1:dimInt);
            %intpoints1 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
            for k1 = j1+1:length(intInd1)
                intpoints2 = repmat(x(intInd1(k1),1:dimInt),numPoints,1) + randpoints(:,dimInt+1:2*dimInt);
                %intpoints2 = repmat(x(intInd1(k1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
                for i1=1:ncf
                    cf = gpcf.cf{i1};
                    temp(j1,k1) = temp(j1,k1) + mean(mean(cf.fh.cov(cf, intpoints1, intpoints2)));
                end
            end
        end
        % The covariance matrix seems to get non positive definite. Try to fix
        % it by evaluating all the diagonal elements with same random numbers.
        %randpoints1 = intArea.*rand(numPoints,dimInt);
        %randpoints2 = intArea.*rand(numPoints,dimInt);
        %randpoints = intArea.*hammersley(dimInt,numPoints)';
        for j1=1:length(intInd1)
            intpoints1 = repmat(x(intInd1(j1),1:dimInt),numPoints,1) + randpoints(:,1:dimInt);
            intpoints2 = repmat(x(intInd1(j1),1:dimInt),numPoints,1) + randpoints(:,dimInt+1:2*dimInt);
            %intpoints1 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + randpoints1;
            %intpoints2 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + randpoints2;
            %intpoints = repmat(x(intInd1(j1),1:end-1),numPoints,1) + randpoints2;
            for i1=1:ncf
                cf = gpcf.cf{i1};
                temp2(j1) = temp2(j1) + mean(mean(cf.fh.cov(cf, intpoints1, intpoints2)));
                %temp2(j1) = temp2(j1) + mean(mean(feval(cf.fh.trcov, cf, intpoints)));
            end
        end
        [I,J,R] = find(temp);
        temp = sparse(intInd1(I),intInd1(J),R,n1,n1);
        C = C + temp + temp' + sparse(1:n1,1:n1,temp2);
        
        C = (C+C')/2;
        
        % check if CS covariances are used. If not change C into full matrix
        % for speed up
        sp = false;
        for i1=1:ncf
            if isfield(gpcf.cf{i1}, 'cs') && gpcf.cf{i1} == 1
                sp=true;
            end
        end
        if ~sp
            C=full(C);
        end
        
        % store in the memory
        Ctrcov=C; 
        datahash0=datahash;
        w0=ww;
    end
    
  end

  function C = gpcf_intcov_trvar(gpcf, x)
  %GP_INTCOV_TRVAR  Evaluate training variance vector
  %
  %  Description
  %    C = GP_INTCOV_TRVAR(GPCF, TX) takes in covariance function of
  %    a Gaussian process GPCF and matrix TX that contains training
  %    inputs. Returns variance vector C. Every element i of C
  %    contains variance of input i in TX.
  %
  %  See also
  %    GPCF_INTCOV_COV, GP_COV, GP_TRCOV


    [n1,m1]=size(x);

    intInd1 = find(x(:,end)==1);
    pointInd1 = find(x(:,end)==0);
    ncf = length(gpcf.cf);
    numPoints = gpcf.NintPoints;
    intArea = repmat(gpcf.intArea,numPoints,1);
    
    C = zeros(n1,1);
    
    % point-point covariance
    temp = 0;
    for i1=1:ncf
      cf = gpcf.cf{i1};
      temp = temp + cf.fh.trvar(cf, x(pointInd1,1:end-1));
    end
    C(pointInd1) = temp;
        
    % area-area covariance
    temp=zeros(size(intInd1));
    for j1=1:length(intInd1)
        intpoints1 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
        intpoints2 = repmat(x(intInd1(j1),1:end-1),numPoints,1) + intArea.*rand(numPoints,dimInt);
        for i1=1:ncf
            cf = gpcf.cf{i1};
            temp(j1) = temp(j1) + mean(mean(cf.fh.cov(cf, intpoints1, intpoints2)));
        end
    end
    C(intInd1) = temp;  
  end

  function reccf = gpcf_intcov_recappend(reccf, ri, gpcf)
  %RECAPPEND  Record append
  %
  %  Description
  %    RECCF = GPCF_INTCOV_RECAPPEND(RECCF, RI, GPCF) takes a
  %    covariance function record structure RECCF, record index RI
  %    and covariance function structure GPCF with the current MCMC
  %    samples of the parameters. Returns RECCF which contains all
  %    the old samples and the current samples from GPCF .
  %
  %  See also
  %    GP_MC and GP_MC -> RECAPPEND

    if nargin == 2
      % Initialize record
      reccf.type = 'gpcf_intcov';
      reccf.NintPoints = ri.NintPoints;
      reccf.intArea = ri.intArea;
      
      % Initialize parameters
      ncf = length(ri.cf);
      for i=1:ncf
        cf = ri.cf{i};
        reccf.cf{i} = cf.fh.recappend([], ri.cf{i});
      end
      
      % Set the function handles
      reccf.fh.pak = @gpcf_intcov_pak;
      reccf.fh.unpak = @gpcf_intcov_unpak;
      reccf.fh.e = @gpcf_intcov_lp;
      reccf.fh.lpg = @gpcf_intcov_lpg;
      reccf.fh.cfg = @gpcf_intcov_cfg;
      reccf.fh.cov = @gpcf_intcov_cov;
      reccf.fh.trcov  = @gpcf_intcov_trcov;
      reccf.fh.trvar  = @gpcf_intcov_trvar;
      reccf.fh.recappend = @gpcf_intcov_recappend;
    else
      % Append to the record
      
      %loop over all of the covariance functions
      ncf = length(gpcf.cf);
      reccf.NintPoints(ri,:) = gpcf.NintPoints;
      reccf.intArea(ri,:) = gpcf.intArea;
      for i=1:ncf
        cf = gpcf.cf{i};
        reccf.cf{i} = cf.fh.recappend(reccf.cf{i}, ri, cf);
      end
    end
  end
end

