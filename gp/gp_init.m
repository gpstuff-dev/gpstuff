function gp = gp_init(do, varargin)
%GP_INIT	Create a Gaussian Process.
%
%	Description
%
%	GP = GP_INIT('INIT', 'TYPE', NIN, 'LIKELIH', GPCF, NOISE, VARARGIN) 
%       Creates a Gaussian Process model with a single output. Takes the number 
%	of inputs  NIN together with string 'LIKELIH' which spesifies likelihood 
%       function used, GPCF array which specify the covariance functions and NOISE 
%       array which specify the noise functions used for Gaussian process. At minimum 
%       one covariance function has to be given. TYPE defines the type of GP, possible 
%       types are  
%       'FULL'        (full GP), 
%       'FIC'         (fully independent conditional), 
%       'PIC_BLOCK'   (block partially independent condional), 
%       'PIC_BAND'    (banded partially independent condional), 
%
%       With VARAGIN can be set fields into different values 
%       VARARGIN = 'FIELD1', VALUE1, 'FIELD2', VALUE2, ... 
%
%	The fields and default values in GP are:
%         type           = 'FULL'
%	  nin            = number of inputs
%	  nout           = number of outputs: always 1
%         cf             = struct of covariance functions
%         noise          = struct of noise functions
%	  jitterSigmas   = jitter term for covariance function
%                          (0.1)
%         p              = prior structure for parameters
%         likelih        = String defining the likelihood function
%                          (Default 'regr' as regression)
%         p.r            = Prior Structure for residual
%                          (defined only in case likelih == 'regr')
%
%       Additional fields when latent values are used (likelih ~='regr'):
%         likelih_e      = String defining the minus log likelihood function
%                          (Default [])
%         likelih_g      = String defining the gradient of minus log likelihood 
%                          function with respect to latent values.
%                          (Default [])
%         fh_latentmc    = Function handle to function which samples the latent values
%                          (not present in regression. If latent values are sampled default @nealmh)
%         latentValues   = Vector of latent values (not needed for regression)
%                          (empty matrix if likelih ~= 'regr')
%
%
%	GP = GPINIT('SET', GP, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in GP.

%
%	See also
%	GPINIT, GP2PAK, GP2UNPAK
%
%

% Copyright (c) 1996,1997 Christopher M Bishop, Ian T Nabney
% Copyright (c) 1998,1999 Aki Vehtari
% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 4
  error('Not enough arguments')
end


% Initialize the Gaussian process
if strcmp(do, 'init')
    
    gp.type = varargin{1};
    gp.nin = varargin{2};
    gp.nout = 1;
  
    % Initialize parameters
    gp.jitterSigmas=0.1;
    
    gp.p=[];
    gp.p.jitterSigmas=[];
    % Set function handle for likelihood. If regression 
    % model is used set also gp.p.r field and if other likelihood
    % set also field gp.latentValues
    if strcmp(varargin{3}, 'regr')
        gp.likelih = 'regr';
        gp.p.r=[];
    else
        gp.likelih = varargin{3};
        gp.fh_latentmc = @latent_hm;
        gp.latentValues = [];
    end  
    
    % Set covariance functions into gpcf
    gp.cf = [];
    gpcf = varargin{4};
    for i = 1:length(gpcf)
        gp.cf{i} = gpcf{i};
    end
    
    % Set noise functions into noise
    gp.noise = [];
    if length(varargin) > 4
        gpnoise = varargin{5};
        for i = 1:length(gpnoise)
            gp.noise{i} = gpnoise{i};
        end
    end
  
    if length(varargin) > 5
        if mod(length(varargin),2) ==0
            error('Wrong number of arguments')
        end
        % Loop through all the parameter values that are changed
        for i=6:2:length(varargin)-1
            switch varargin{i}
              case 'jitterSigmas'
                gp.jitterSigmas = varargin{i+1};
              case 'likelih'
                gp.likelih = varargin{i+1};
                if strcmp(gp.likelih_e, 'regr')
                    gp.p.r=[];
                else
                    gp.latentValues = [];
                end
              case 'likelih_e'
                gp.likelih_e = varargin{i+1};
              case 'likelih_g'
                gp.likelih_g = varargin{i+1};
              case 'fh_latentmc'
                gp.fh_latentmc = varargin{i+1};
              case 'type'
                gp.type = varargin{i+1};
              case 'X_u'
                if size(varargin{i+1},2)~=nin
                    error('The size of X_u has to be u x nin.')
                else
                    gp.X_u = varargin{i+1};
                end
              otherwise
                error('Wrong parameter name!')
            end
        end
    end
end
    
% Set the parameter values of covariance function
if strcmp(do, 'set')
  if mod(nargin,2) ~=0
    error('Wrong number of arguments')
  end
  gp = varargin{1};
  % Loop through all the parameter values that are changed
  for i=2:2:length(varargin)-1
      switch varargin{i}
        case 'jitterSigmas'
          gp.jitterSigmas = varargin{i+1};
        case 'likelih'
          gp.likelih = varargin{i+1};
          if strcmp(gp.likelih, 'regr')
              gp.p.r=[];
          else
              gp.latentValues = [];
          end
        case 'likelih_e'
          gp.likelih_e = varargin{i+1};
        case 'likelih_g'
          gp.likelih_g = varargin{i+1};
        case 'fh_latentmc'
          gp.fh_latentmc = varargin{i+1};
        case 'sparse'
          gp.sparse = varargin{i+1};
        case 'X_u'
          gp.X_u = varargin{i+1};
        otherwise
          error('Wrong parameter name!')
      end    
  end
end


% Add new covariance function
if strcmp(do, 'add_cf')
  gp = varargin{1};
  for i = 2:length(varargin)
    gp.cf{end+1} = varargin{i};
  end
end

