function gpcf = gpcf_prod(varargin)
%GPCF_PROD  Create a product form covariance function
%
%  Description
%    GPCF = GPCF_PROD('cf', {GPCF_1, GPCF_2, ...}) 
%    creates a product form covariance function
%          GPCF = GPCF_1 .* GPCF_2 .* ... .* GPCF_N
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2009-2010, 2018 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2014 Arno Solin and Jukka Koskenranta

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPCF_PROD';
  ip.addOptional('gpcf', [], @isstruct);
  ip.addParamValue('cf',[], @iscell);
  ip.parse(varargin{:});
  gpcf=ip.Results.gpcf;

  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_prod';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_prod')
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
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_prod_pak;
    gpcf.fh.unpak = @gpcf_prod_unpak;
    gpcf.fh.lp = @gpcf_prod_lp;
    gpcf.fh.lpg = @gpcf_prod_lpg;
    gpcf.fh.cfg = @gpcf_prod_cfg;
    gpcf.fh.cfdg = @gpcf_prod_cfdg;
    gpcf.fh.cfdg2 = @gpcf_prod_cfdg2;
    gpcf.fh.ginput = @gpcf_prod_ginput;
    gpcf.fh.ginput2 = @gpcf_prod_ginput2;
    gpcf.fh.ginput3 = @gpcf_prod_ginput3;
    gpcf.fh.ginput4 = @gpcf_prod_ginput4;
    gpcf.fh.cov = @gpcf_prod_cov;
    gpcf.fh.trcov  = @gpcf_prod_trcov;
    gpcf.fh.trvar  = @gpcf_prod_trvar;
    gpcf.fh.recappend = @gpcf_prod_recappend;
    gpcf.fh.cf2ss = @gpcf_prod_cf2ss;
  end

end

function [w, s, h] = gpcf_prod_pak(gpcf)
%GPCF_PROD_PAK  Combine GP covariance function parameters into one vector
%
%  Description
%    W = GPCF_PROD_PAK(GPCF, W) loops through all the covariance
%    functions and packs their parameters into one vector as
%    described in the respective functions. This is a mandatory 
%    subfunction used for example in energy and gradient computations.
%
%  See also
%    GPCF_PROD_UNPAK
  
  ncf = length(gpcf.cf);
  w = []; s = {}; h=[];
  
  for i=1:ncf
    cf = gpcf.cf{i};
    [wi, si, hi] = cf.fh.pak(cf);
    w = [w wi];
    s = [s; si];
    h = [h hi];
  end
end

function [gpcf, w] = gpcf_prod_unpak(gpcf, w)
%GPCF_PROD_UNPAK  Sets the covariance function parameters into
%                 the structures
%
%  Description
%    [GPCF, W] = GPCF_PROD_UNPAK(GPCF, W) loops through all the
%    covariance functions and unpacks their parameters from W to
%    each covariance function structure. This is a mandatory 
%    subfunction used for example in energy and gradient computations.
% 
%  See also
%    GPCF_PROD_PAK
%
  ncf = length(gpcf.cf);
  
  for i=1:ncf
    cf = gpcf.cf{i};
    [cf, w] = cf.fh.unpak(cf, w);
    gpcf.cf{i} = cf;
  end

end

function lp = gpcf_prod_lp(gpcf)
%GPCF_PROD_LP  Evaluate the log prior of covariance function parameters
%
%  Description
%    LP = GPCF_PROD_LP(GPCF, X, T) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters. This is a mandatory subfunction used for example 
%    in energy computations.
%
%  See also
%    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_LPG, GP_E
  
  lp = 0;
  ncf = length(gpcf.cf);
  for i=1:ncf
    cf = gpcf.cf{i};
    lp = lp + cf.fh.lp(cf);
  end
  
end

function lpg = gpcf_prod_lpg(gpcf)
%GPCF_PROD_LPG  Evaluate gradient of the log prior with respect
%               to the parameters.
%
%  Description
%    LPG = GPCF_PROD_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters. This is a mandatory subfunction 
%    used for example in gradient computations.
%
%  See also
%    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_LP, GP_G
  lpg = [];
  ncf = length(gpcf.cf);
  
  % Evaluate the gradients
  for i=1:ncf
    cf = gpcf.cf{i};
    lpg=[lpg cf.fh.lpg(cf)];
  end

end

function DKff = gpcf_prod_cfg(gpcf, x, x2, mask, i1)
%GPCF_PROD_CFG  Evaluate gradient of covariance function
%               with respect to the parameters.
%
%  Description
%    DKff = GPCF_PROD_CFG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to th (cell array with matrix elements). This is a 
%    mandatory subfunction used in gradient computations.
%
%    DKff = GPCF_PROD_CFG(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_PROD_CFG(GPCF, X, [], MASK) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the diagonal of gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_PROD_CFG(GPCF, X, X2, [], i) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    hyperparameter. This subfunction is needed when using 
%    memory save option in gp_set.
%
%  See also
%    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_LP, GP_G

  [n, m] =size(x);
  ncf = length(gpcf.cf);

  DKff = {};

  if nargin==5
    % Use memory save option
    savememory=1;
    i3=0;
    for k=1:ncf
      % Number of hyperparameters for each covariance function
      cf=gpcf.cf{k};
      i3(k)=cf.fh.cfg(cf,[],[],[],0);
    end
    if i1==0
      % Return number of hyperparameters
      DKff=sum(i3);
      return
    end
    % Help indices
    i3=cumsum(i3);
    ind=find(cumsum(i3 >= i1)==1);
    if ind>1
      i1=[ind i1-i3(ind-1)];
    else
      i1=[ind i1];
    end
  else
    savememory=0;
  end

  % Evaluate: DKff{1} = d Kff / d magnSigma2
  %           DKff{2} = d Kff / d lengthScale
  % NOTE! Here we have already taken into account that the parameters are transformed
  % through log() and thus dK/dlog(p) = p * dK/dp

  % evaluate the gradient for training covariance
  if nargin == 2 || (isempty(x2) && isempty(mask))
        
    % evaluate the individual covariance functions
    for i=1:ncf
      cf = gpcf.cf{i};
      C{i} = cf.fh.trcov(cf, x);
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    DKff = {};
    if ~savememory
      i3=1:ncf;
    else
      i3=i1(1);
    end
    for i=i3
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.cfg(cf, x);
      else
        DK = {cf.fh.cfg(cf,x,[],[],i1(2))};
      end
      
      CC = 1;
      for kk = ind(ind~=i)
        CC = CC.*C{kk};
      end
      for j = 1:length(DK)
        DKff{end+1} = DK{j}.*CC;
      end
    end
    
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 || isempty(mask)
    if size(x,2) ~= size(x2,2)
      error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
        
    % evaluate the individual covariance functions
    for i=1:ncf
      cf = gpcf.cf{i};
      C{i} = cf.fh.cov(cf, x, x2);
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    DKff = {};
    if ~savememory
      i3=1:ncf;
    else
      i3=i1(1);
    end
    for i=i3
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.cfg(cf, x,x2);
      else
        DK = {cf.fh.cfg(cf,x,x2,[],i1(2))};
      end
      
      CC = 1;
      for kk = ind(ind~=i)
        CC = CC.*C{kk};
      end
      
      for j = 1:length(DK)
        DKff{end+1} = DK{j}.*CC;
      end
    end

    
    
    % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
    %           DKff{2...} = d mask(Kff,I) / d lengthScale
  elseif nargin == 4 || nargin == 5
    
    % evaluate the individual covariance functions
    for i=1:ncf
      cf = gpcf.cf{i};
      C{i} = cf.fh.trvar(cf, x);
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    DKff = {};
    if ~savememory
      i3=1:ncf;
    else
      i3=i1(1);
    end
    for i=i3
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.cfg(cf, x, [], 1);
      else
        DK = {cf.fh.cfg(cf, x, [], 1, i1(2))};
      end
      
      CC = 1;
      for kk = ind(ind~=i)
        CC = CC.*C{kk};
      end
      
      for j = 1:length(DK)
        DKff{end+1} = DK{j}.*CC;
      end
    end
  end
  if savememory
    DKff=DKff{1};
  end
end

function DKff = gpcf_prod_cfdg(gpcf, x, x2, dims)
%GPCF_SEXP_CFDG  Evaluate gradient of covariance function, of
%                which has been taken partial derivative with
%                respect to x, with respect to parameters.
%
%  Description
%    DKff = GPCF_SEXP_CFDG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of derivatived covariance matrix
%    dK(df,f)/dhyp = d(d k(X,X)/dx)/dhyp, with respect to the
%    parameters
%
%    Evaluate: DKff{1:m} = d Kff / d magnSigma2
%              DKff{m+1:2m} = d Kff / d lengthScale_m
%    m is the dimension of inputs. If ARD is used, then multiple
%    lengthScales. This subfunction is needed when using derivative 
%    observations.
%
%         dims - is a vector of input dimensions with respect to which the
%                derivatives of the covariance function have been calculated
%                [by default dims=1:size(x,2)]
%
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_SEXP_GINPUT



ncf = length(gpcf.cf);

[~, m] =size(x);
if nargin <3 || isempty(x2)
    x2=x;
end
if nargin < 4 || isempty(dims)
    %dims1 = 1:m;
    error('dims1 needs to be given')
end


% check whether there are any covariance functions that depend on dims
% covariate
ind = [];
DKff = {};
for i1 = 1:ncf
    if (~isfield(gpcf.cf{i1}, 'selectedVariables') || any(ismember(gpcf.cf{i1}.selectedVariables,dims)))
        ind(end+1) = i1;
    end
end
if ~isempty(ind)
    % Calculate derivative
    % calculate the individual covariance functions
    Ci = {};
    C = 1;
    for i1=1:ncf
        Ci{i1} = gpcf.cf{i1}.fh.cov(gpcf.cf{i1}, x, x2);
        C = C.*Ci{i1};
    end
      
    if ncf==2
        if any(ind==1)
            DKdx1 = gpcf.cf{1}.fh.cfdg(gpcf.cf{1}, x, x2, dims);
            DKx1 = gpcf.cf{1}.fh.ginput4(gpcf.cf{1}, x, x2, dims);
        else
            DKdx1{1} = 0;
            DKx1{1} = 0;
        end
        DK1 = gpcf.cf{1}.fh.cfg(gpcf.cf{1}, x, x2);
        if any(ind==2)
            DKdx2 = gpcf.cf{2}.fh.cfdg(gpcf.cf{2}, x, x2, dims);
            DKx2 = gpcf.cf{2}.fh.ginput4(gpcf.cf{2}, x, x2, dims);
        else
            DKdx2{1} = 0;
            DKx2{1} = 0;
        end
        DK2 = gpcf.cf{2}.fh.cfg(gpcf.cf{2}, x, x2);
        
        for i2 = 1:length(DK1)
            DKff{end+1} = DKdx1{i2}.*C./Ci{1} + DK1{i2}.*DKx2{1};
        end
        for i2 = 1:length(DK2)
            DKff{end+1} = DKdx2{i2}.*C./Ci{2} + DK2{i2}.*DKx1{1};
        end
        
    else
        error('not implemented yet')
        for i1=1:ncf
            
            DKdx = gpcf.cf{i1}.fh.cfdg(gpcf.cf{i1}, x, x2, dims);
            DK = gpcf.cf{i1}.fh.cfg(gpcf.cf{i1}, x, x2);
            
            gpcftmp = gpcf;
            gpcftmp.cf = {gpcftmp.cf(1:ncf~=i1)};
            % continue from here
            %DKx2 =
        end
    end
else
    DK = gpcf.fh.lpg(gpcf);
    for i2 = 1:length(DK)
        DKff{end+1} = 0;
    end
end


end


function DKff = gpcf_prod_cfdg2(gpcf, x, x2, dims1, dims2)
%GPCF_SEXP_CFDG  Evaluate gradient of covariance function, of
%                which has been taken partial derivative with
%                respect to x, with respect to parameters.
%
%  Description
%    DKff = GPCF_SEXP_CFDG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of derivatived covariance matrix
%    dK(df,f)/dhyp = d(d k(X,X)/dx)/dhyp, with respect to the
%    parameters
%
%    Evaluate: DKff{1:m} = d Kff / d magnSigma2
%              DKff{m+1:2m} = d Kff / d lengthScale_m
%    m is the dimension of inputs. If ARD is used, then multiple
%    lengthScales. This subfunction is needed when using derivative 
%    observations.
%
%         dims - is a vector of input dimensions with respect to which the
%                derivatives of the covariance function have been calculated
%                [by default dims=1:size(x,2)]
%
%
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_SEXP_GINPUT

ncf = length(gpcf.cf);

[~, m] =size(x);
if nargin <3 || isempty(x2)
    x2=x;
end
if nargin < 4 || isempty(dims1)
    %dims1 = 1:m;
    error('dims1 needs to be given')
end
if nargin < 5 || isempty(dims2)
    %dims2 = 1:m;
    error('dims2 needs to be given')
end

% NOTICE. AS OF NOW we assume that dims1 and dims2 are scalars

% check whether there are any covariance functions that depend on dims
% covariate
ind = [];
DKff = {};
for i1 = 1:ncf
    if (~isfield(gpcf.cf{i1}, 'selectedVariables') || any(ismember(gpcf.cf{i1}.selectedVariables,dims1)) || any(ismember(gpcf.cf{i1}.selectedVariables,dims2)))
        ind(end+1) = i1;
    end
end
if ~isempty(ind)
    % Calculate derivative
    % calculate the individual covariance functions
    Ci = {};
    C = 1;
    for i1=1:ncf
        Ci{i1} = gpcf.cf{i1}.fh.cov(gpcf.cf{i1}, x, x2);
        %C = C.*Ci{i1};
    end
      
    if ncf==2
        if any(ind==1)
            DKdx11 = gpcf.cf{1}.fh.cfdg(gpcf.cf{1}, x, x2, dims1);
            DKdx12 = gpcf.cf{1}.fh.cfdg(gpcf.cf{1}, x2, x, dims2);
            DKdx1_2 = gpcf.cf{1}.fh.cfdg2(gpcf.cf{1}, x, x2, dims1, dims2);
            DKx11 = gpcf.cf{1}.fh.ginput4(gpcf.cf{1}, x, x2, dims1);
            DKx12 = gpcf.cf{1}.fh.ginput4(gpcf.cf{1}, x2, x, dims2);
        else
            DKdx11{1} = 0;
            DKdx12{1} = 0;
            DKdx1_2{1} = 0;
            DKx11{1} = 0;
            DKx12{1} = 0;
        end
        if any(ind==2)
            DKdx21 = gpcf.cf{2}.fh.cfdg(gpcf.cf{2}, x, x2, dims1);
            DKdx22 = gpcf.cf{2}.fh.cfdg(gpcf.cf{2}, x2, x, dims2);
            DKdx2_2 = gpcf.cf{2}.fh.cfdg2(gpcf.cf{2}, x, x2, dims1, dims2);
            DKx21 = gpcf.cf{2}.fh.ginput4(gpcf.cf{2}, x, x2, dims1);
            DKx22 = gpcf.cf{2}.fh.ginput4(gpcf.cf{2}, x2, x, dims2);
        else
            DKdx21{1} = 0;
            DKdx22{1} = 0;
            DKdx2_2{1} = 0;
            DKx21{1} = 0;
            DKx22{1} = 0;
        end
        
        if dims1==dims2
            if any(ind==1)
                D2Kx1 = gpcf.cf{1}.fh.ginput2(gpcf.cf{1}, x, x2, dims1, dims2);
            else
                D2Kx1{1} = 0;
            end
            if any(ind==2)
                D2Kx2 = gpcf.cf{2}.fh.ginput2(gpcf.cf{2}, x, x2, dims1, dims2);
            else
                D2Kx2{1} = 0;
            end
        else
%             D2Kx1 = gpcf.cf{1}.fh.ginput3(gpcf.cf{1}, x, x2, dims1, dims2);
%             D2Kx2 = gpcf.cf{2}.fh.ginput3(gpcf.cf{2}, x, x2, dims1, dims2);
            if any(ind==1)
                D2Kx1 = gpcf.cf{1}.fh.ginput3(gpcf.cf{1}, x, x2, dims1, dims2);
            else
                D2Kx1{1} = 0;
            end
            if any(ind==2)
                D2Kx2 = gpcf.cf{2}.fh.ginput3(gpcf.cf{2}, x, x2, dims1, dims2);
            else
                D2Kx2{1} = 0;
            end
        end
        
        DK1 = gpcf.cf{1}.fh.cfg(gpcf.cf{1}, x, x2);
        DK2 = gpcf.cf{2}.fh.cfg(gpcf.cf{2}, x, x2);
        
        for i2 = 1:length(DK1)
            DKff{end+1} = DKdx1_2{i2}.*Ci{2} + DKdx11{i2}.*DKx22{1}' + DKdx12{i2}'.*DKx21{1} + DK1{i2}.*D2Kx2{1};
        end
        for i2 = 1:length(DK2)
            DKff{end+1} = DKdx2_2{i2}.*Ci{1} + DKdx21{i2}.*DKx12{1}' + DKdx22{i2}'.*DKx11{1} + DK2{i2}.*D2Kx1{1};
        end
        
    else
        error('not implemented yet')
        for i1=1:ncf
            
            DKdx = gpcf.cf{i1}.fh.cfdg(gpcf.cf{i1}, x, x2, dims);
            DK = gpcf.cf{i1}.fh.cfg(gpcf.cf{i1}, x, x2);
            
            gpcftmp = gpcf;
            gpcftmp.cf = {gpcftmp.cf(1:ncf~=i1)};
            % continue from here
            %DKx2 =
        end
    end
else
    DK = gpcf.fh.lpg(gpcf);
    for i2 = 1:length(DK)
        DKff{end+1} = 0;
    end
end

% [n, m] =size(x);
% ncf = length(gpcf.cf);
% 
% if nargin <3 || isempty(x2)
%     x2=x;
% end
% if nargin < 4 || isempty(dims1)
%     %dims1 = 1:m;
%     error('dims1 needs to be given')
% end
% if nargin < 5 || isempty(dims2)
%     %dims2 = 1:m;
%     error('dims2 needs to be given')
% end
% 
% 
% % check whether there are any covariance functions that depend on dims
% % covariate
% ind = [];
% DKff = {};
% for i1 = 1:ncf
%     if (~isfield(gpcf.cf{i1}, 'selectedVariables') || any(ismember(gpcf.cf{i1}.selectedVariables,dims1)) || any(ismember(gpcf.cf{i1}.selectedVariables,dims2)))
%         ind(end+1) = i1;
%     end
% end
% if ~isempty(ind)
%     % Calculate derivative
%     
%     % calculate the individual covariance functions
%     Ci = {};
%     C = 1;
%     for i1=1:ncf
%         Ci{i1} = gpcf.cf{i1}.fh.cov(gpcf.cf{i1}, x, x2);
%         C = C.*Ci{i1};
%     end
%     
%     for i1=1:ncf
%         if any(i1 == ind)
%             DK = gpcf.cf{i1}.fh.cfdg2(gpcf.cf{i1}, x, x2, dims1, dims2);
%             for i2 = 1:length(DK)
%                 DKff{end+1} = DK{i2}.*C./Ci{i1};
%             end
%         else
%             DK = gpcf.cf{i1}.fh.lpg(gpcf.cf{i1});
%             for i2 = 1:length(DK)
%                 DKff{end+1} = 0;
%             end
%         end
%     end
% else
%     DK = gpcf.fh.lpg(gpcf);
%     for i2 = 1:length(DK)
%         DKff{end+1} = 0;
%     end
% end

end



function DKff = gpcf_prod_ginput(gpcf, x, x2, i1)
%GPCF_PROD_GINPUT  Evaluate gradient of covariance function with 
%                  respect to x
%
%  Description
%    DKff = GPCF_PROD_GINPUT(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to X (cell array with matrix elements). This subfunction 
%    is needed when computing gradients with respect to inducing 
%    inputs in sparse approximations.
%
%    DKff = GPCF_PROD_GINPUT(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%    This subfunction is needed when computing gradients with 
%    respect to inducing inputs in sparse approximations.
%
%    DKff = GPCF_PROD_GINPUT(GPCF, X, X2, i) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith
%    covariate in X (cell array with matrix elements). This
%    subfunction is needed when using memory save option in
%    gp_set.
%
%  See also
%    GPCF_PROD_PAK, GPCF_PROD_UNPAK, GPCF_PROD_LP, GP_G
  
  [n, m] =size(x);

  if nargin==4
    % Use memory save option
    savememory=1;
    if i1==0
      % Return number of covariates
      if isfield(gpcf,'selectedVariables')
        DKff=length(gpcf.selectedVariables);
      else
        DKff=m;
      end
      return
    end
  else
    savememory=0;
  end
  % evaluate the gradient for training covariance
  if nargin == 2 || isempty(x2)
    
    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    for i=1:ncf
      cf = gpcf.cf{i};
      C{i} = cf.fh.trcov(cf, x);
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    if ~savememory
      DKff=cellfun(@(a) zeros(n,n), cell(1,numel(x)), 'UniformOutput', 0);
    else
      DKff=cellfun(@(a) zeros(n,n), cell(1,n), 'UniformOutput', 0);
    end
    for i=1:ncf
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.ginput(cf, x);
      else
        DK = cf.fh.ginput(cf, x, [], i1);
      end

      CC = 1;
      for kk = ind(ind~=i)
        CC = CC.*C{kk};
      end
      
      for j = 1:length(DK)
        DKff{j} = DKff{j} + DK{j}.*CC;
      end
    end

    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 ||  nargin == 4
    if size(x,2) ~= size(x2,2)
      error('gpcf_prod -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
    
    ncf = length(gpcf.cf);
    
    % evaluate the individual covariance functions
    for i=1:ncf
      cf = gpcf.cf{i};
      C{i} = cf.fh.cov(cf, x, x2);
    end
    
    % Evaluate the gradients
    ind = 1:ncf;
    if ~savememory
      DKff=cellfun(@(a) zeros(n,n), cell(1,numel(x)), 'UniformOutput', 0);
    else
      DKff=cellfun(@(a) zeros(n,n), cell(1,n), 'UniformOutput', 0);
    end
    for i=1:ncf
      cf = gpcf.cf{i};
      if ~savememory
        DK = cf.fh.ginput(cf, x, x2);
      else
        DK = cf.fh.ginput(cf, x, x2, i1);
      end
      
      CC = 1;
      for kk = ind(ind~=i)
        CC = CC.*C{kk};
      end
      
      for j = 1:length(DK)
        DKff{j} = DKff{j} + DK{j}.*CC;
      end
    end
  end
  
end


function [DKff, DKff1, DKff2]  = gpcf_prod_ginput2(gpcf, x, x2, dims, takeOnlyDiag)
%GPCF_PROD_GINPUT2  Evaluate gradient of covariance function with
%                   respect to both input variables x and x2 (in
%                   same dimension).
%
%  Description
%    DKff = GPCF_PROD_GINPUT2(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of twice derivatived covariance
%    matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix
%    elements). Input variable's dimensions are expected to be
%    same. The function returns also DKff1 and DKff2 which are
%    parts of DKff and needed with CFDG2. DKff = DKff1 -
%    DKff2. This subfunction is needed when using derivative 
%    observations.
%   
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_PROD_GINPUT, GPCF_PROD_GINPUT2, GPCF_PROD_CFDG2 
if nargin < 5
    takeOnlyDiag=[];
end

ncf = length(gpcf.cf);

% check whether there are any covariance functions that depend on dims
% covariate
ind = [];
for i1 = 1:ncf
    if (~isfield(gpcf.cf{i1}, 'selectedVariables') || any(ismember(gpcf.cf{i1}.selectedVariables,dims)))
        ind(end+1) = i1;
    end
end
% Calculate derivative
if ~isempty(ind)
    % evaluate the individual covariance functions
    C = 1;
    K = 0;
    for i1=1:ncf
        if ~isempty(takeOnlyDiag) && strcmp(takeOnlyDiag, 'takeOnlyDiag')
            Ci = gpcf.cf{i1}.fh.trvar(gpcf.cf{i1}, x);
        else
            Ci = gpcf.cf{i1}.fh.cov(gpcf.cf{i1}, x, x2);
        end
        C = C.*Ci;
        if any(i1 == ind)
            Dt = gpcf.cf{i1}.fh.ginput2(gpcf.cf{i1}, x, x2, dims, takeOnlyDiag);
            if length(Dt)>1
                error('gpcf_prod->ginput2: something is wrong')
            end
            K = K+Dt{1}./Ci;
        end
    end
    DKff = {C.*K};
else
    DKff = {0};
end

end


function DKff = gpcf_prod_ginput3(gpcf, x, x2, dims1, dims2)
%GPCF_PROD_GINPUT3  Evaluate gradient of covariance function with
%                   respect to both input variables x and x2 (in
%                   different dimensions).
%
%  Description
%    DKff = GPCF_PROD_GINPUT3(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of twice derivatived covariance
%    matrix K(df,df) = dk(X1,X2)/dX1dX2 (cell array with matrix
%    elements). The derivative is calculated in multidimensional
%    problem between input's observation dimensions which are not
%    same. This subfunction is needed when using derivative 
%    observations.
%
%    ---- !!note this help text needs to be corrected !! ---
%    DKff is a cell array with the following elements:
%      DKff{1} = dk(X1,X2)/dX1_1dX2_2
%      DKff{2} = dk(X1,X2)/dX1_1dX2_3
%       ... 
%      DKff{m-1} = dk(X1,X2)/dX1_1dX2_m
%      DKff{m} = dk(X1,X2)/dX1_2dX2_3
%       ...
%      DKff{m} = dk(X1,X2)/dX1_(m-1)dX2_m
%    where _m denotes the input dimension with respect to which the
%    gradient is calculated.
%     ---- clip ---
%   
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_PROD_GINPUT, GPCF_PROD_GINPUT2, GPCF_PROD_CFDG2 

ncf = length(gpcf.cf);

% check whether there are any covariance functions that depend on dims
% covariate
ind = [];
for i1 = 1:ncf
    if (~isfield(gpcf.cf{i1}, 'selectedVariables') || any(ismember(gpcf.cf{i1}.selectedVariables,dims1)) || any(ismember(gpcf.cf{i1}.selectedVariables,dims2)) )
        ind(end+1) = i1;
    end
end
% Calculate derivative
if ~isempty(ind)
    % evaluate the individual covariance functions
    C = 1;
    K = 0;
    for i1=1:ncf
        Ci{i1} = gpcf.cf{i1}.fh.cov(gpcf.cf{i1}, x, x2);
        C = C.*Ci{i1};
    end
    for i1=1:ncf
        if any(i1 == ind)
            % double derivative of i'th covariance function
            Dt = gpcf.cf{i1}.fh.ginput3(gpcf.cf{i1}, x, x2, dims1, dims2);
            if length(Dt)>1
                error('gpcf_prod->ginput3: something is wrong')
            end
            K = K+Dt{1}./Ci{i1};
            % pairwise derivatives of i'th and j'th covariance functions
            Dti1 = gpcf.cf{i1}.fh.ginput4(gpcf.cf{i1}, x, x2, dims1);
            Dti2 = gpcf.cf{i1}.fh.ginput4(gpcf.cf{i1}, x2, x, dims2);
            for j1=i1+1:ncf
                if any(j1 == ind)
                    Dtj1 = gpcf.cf{j1}.fh.ginput4(gpcf.cf{j1}, x, x2, dims1);
                    Dtj2 = gpcf.cf{j1}.fh.ginput4(gpcf.cf{j1}, x2, x, dims2);
                    
                    K = K + (Dti1{1}.*Dtj2{1}' + Dtj1{1}.*Dti2{1}')./Ci{i1}./Ci{j1};
                end
            end
        end
    end
    DKff = {C.*K};
else
    DKff = {0};
end


end

function DKff = gpcf_prod_ginput4(gpcf, x, x2, dims)
%GPCF_PROD_GINPUT4  Evaluate gradient of covariance function with 
%                  respect to x. Simplified and faster version of
%                  sexp_ginput, returns full matrices.
%
%  Description
%    DKff = GPCF_PROD_GINPUT4takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to X (whole matrix). This subfunction is needed when 
%    using derivative observations.
%   
%    Note! When coding the derivatives of the covariance function, remember
%    to double check them. See gp_cov for lines of code to check the
%    matrices
%
%  See also
%    GPCF_PROD_GINPUT, GPCF_PROD_GINPUT2, GPCF_PROD_CFDG2 

ncf = length(gpcf.cf);

% check whether there are any covariance functions that depend on dims
% covariate
ind = [];
for i1 = 1:ncf
    if (~isfield(gpcf.cf{i1}, 'selectedVariables') || any(ismember(gpcf.cf{i1}.selectedVariables,dims)))
        ind(end+1) = i1;
    end
end
% Calculate derivative
if ~isempty(ind)
    % evaluate the individual covariance functions
    C = 1;
    K = 0;
    for i1=1:ncf
        Ci = gpcf.cf{i1}.fh.cov(gpcf.cf{i1}, x, x2);
        C = C.*Ci;
        if any(i1 == ind)            
            Dt = gpcf.cf{i1}.fh.ginput4(gpcf.cf{i1}, x, x2, dims);
            if length(Dt)>1
                error('gpcf_prod->ginput4: something is wrong')
            end
            K = K+Dt{1}./Ci;
        end
    end
    DKff = {C.*K};
else
    DKff = {0};
end


end



function C = gpcf_prod_cov(gpcf, x1, x2)
%GP_PROD_COV  Evaluate covariance matrix between two input vectors
%
%  Description         
%    C = GP_PROD_COV(GP, TX, X) takes in covariance function of a
%    Gaussian process GP and two matrixes TX and X that contain
%    input vectors to GP. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i in TX
%    and j in X. This is a mandatory subfunction used for example in
%    prediction and energy computations.
%
%
%  See also
%    GPCF_PROD_TRCOV, GPCF_PROD_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
    x2=x1;
  end
  [n1,m1]=size(x1);
  [n2,m2]=size(x2);

  if m1~=m2
    error('the number of columns of X1 and X2 has to be same')
  end

  ncf = length(gpcf.cf);
  
  % evaluate the individual covariance functions
  C = 1;
  for i=1:ncf
    cf = gpcf.cf{i};
    C = C.*cf.fh.cov(cf, x1, x2);
  end        
end

function C = gpcf_prod_trcov(gpcf, x)
%GP_PROD_TRCOV     Evaluate training covariance matrix of inputs
%
%  Description
%    C = GP_PROD_TRCOV(GP, TX) takes in covariance function of a
%    Gaussian process GP and matrix TX that contains training
%    input vectors. Returns covariance matrix C. Every element ij
%    of C contains covariance between inputs i and j in TX. This 
%    is a mandatory subfunction used for example in prediction and 
%    energy computations.
%
%  See also
%    GPCF_PROD_COV, GPCF_PROD_TRVAR, GP_COV, GP_TRCOV
  ncf = length(gpcf.cf);
  
  % evaluate the individual covariance functions
  C = 1;
  for i=1:ncf
    cf = gpcf.cf{i};
    C = C.*cf.fh.trcov(cf, x);
  end
end

function C = gpcf_prod_trvar(gpcf, x)
% GP_PROD_TRVAR     Evaluate training variance vector
%
%  Description
%    C = GP_PROD_TRVAR(GPCF, TX) takes in covariance function of
%    a Gaussian process GPCF and matrix TX that contains training
%    inputs. Returns variance vector C. Every element i of C
%    contains variance of input i in TX. This is a mandatory 
%    subfunction used for example in prediction and energy computations.
%
%  See also
%    GPCF_PROD_COV, GP_COV, GP_TRCOV


  ncf = length(gpcf.cf);
  
  % evaluate the individual covariance functions
  C = 1;
  for i=1:ncf
    cf = gpcf.cf{i};
    C = C.*cf.fh.trvar(cf, x);
  end
end

function reccf = gpcf_prod_recappend(reccf, ri, gpcf)
%RECAPPEND  Record append
%
%  Description
%    RECCF = GPCF_PROD_RECAPPEND(RECCF, RI, GPCF) takes a
%    covariance function record structure RECCF, record index RI
%    and covariance function structure GPCF with the current MCMC
%    samples of the parameters. Returns RECCF which contains all
%    the old samples and the current samples from GPCF. This 
%    subfunction is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC, GP_MC->RECAPPEND
  
  if nargin == 2
    % Initialize the record
    reccf.type = 'gpcf_prod';

    % Initialize parameters
    ncf = length(ri.cf);
    for i=1:ncf
      cf = ri.cf{i};
      reccf.cf{i} = cf.fh.recappend([], ri.cf{i});
    end
    
    % Set the function handles
    reccf.fh.pak = @gpcf_prod_pak;
    reccf.fh.unpak = @gpcf_prod_unpak;
    reccf.fh.lp = @gpcf_prod_lp;
    reccf.fh.lpg = @gpcf_prod_lpg;
    reccf.fh.cfg = @gpcf_prod_cfg;
    reccf.fh.cov = @gpcf_prod_cov;
    reccf.fh.trcov  = @gpcf_prod_trcov;
    reccf.fh.trvar  = @gpcf_prod_trvar;
    reccf.fh.recappend = @gpcf_prod_recappend;
  else
    % Append to the record
    
    % Loop over all of the covariance functions
    ncf = length(gpcf.cf);
    for i=1:ncf
      cf = gpcf.cf{i};
      reccf.cf{i} = cf.fh.recappend(reccf.cf{i}, ri, cf);
    end
  end
end

function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = gpcf_prod_cf2ss(gpcf,x)
%GPCF_PROD_CF2SS Convert the covariance function to state space form
%
%  Description
%    Convert the sum of two covariance functions to the corresponding
%    sum of two state space models. Details on how this is done can
%    be found in the reference:
%
%  References
%    Arno Solin and Simo Sarkka (2014). Explicit link between periodic 
%    covariance functions and state space models. Accepted for 
%    publication in Proceedings of the Seventeenth International 
%    Conference on Artifcial Intelligence and Statistics (AISTATS 2014).
%

  % Check arguments
  if nargin < 2, x = []; end

  % Vector of function handles of conversion functions 
  % from covariance functions to state space 
  cf2ssvect = cell(length(gpcf.cf),1);
  for k = 1:length(gpcf.cf)
     
      % Initial function handle 
      fh = @(y) gpcf.cf{k}.fh.cf2ss(gpcf.cf{k},x);
      
      % Deal with the periodic (deterministic) covariance function
      % as a special case
      if isequal(gpcf.cf{k}.type,'gpcf_periodic') && gpcf.cf{k}.decay == 0
        cf2ssvect{k} = @(y) cf2ss_periodicprod(fh);
      else
        cf2ssvect{k} = fh;
      end
      
  end
  
  % Return model matrices, derivatives and parameter information
  [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = ...
      cf_prod_to_ss(cf2ssvect);
  
end

function [F,L,Pinf1,H,Pinf,dF,dPinf,dPinf1,params] = cf2ss_periodicprod(fh)
% CF2SS_PERIODICPROD - Computes Qc values for periodic covariance production

 [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = fh();
 Pinf1 = Pinf; dPinf1 = dPinf;
end


