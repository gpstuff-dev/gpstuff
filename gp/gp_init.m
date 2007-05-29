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
%        'FULL'        (full GP), 
%        'FIC'         (fully independent conditional), 
%        'PIC_BLOCK'   (block partially independent condional), 
%        'PIC_BAND'    (banded partially independent condional), 
%
%       With VARAGIN the fields of the GP structure can be set into different values 
%       VARARGIN = 'FIELD1', VALUE1, 'FIELD2', VALUE2, ... 
%
%	GP = GPINIT('SET', GP, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of the fields FIELD1... to the values VALUE1... in GP.
%
%	The fields (minimum number of them) and default values in GP are:
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
%       In sparse GP models the needed fields are following:%
%         X_u            = Inducing inputs in sparse models
%         blocks         = Initializes the blocks for the PIC_BLOCK model
%                          The value for blocks has to be a cell array of type 
%                          {'method', matrix of training inputs, param}. 
%
%                          The possible methods are:
%                            'manual', which takes in the place of param a structure of the index vectors
%                                appointing the data points into blocks. For example, if x is a matrix of data inputs
%                                then x(param{i},:) are the inputs belonging to the ith block.
%         truncated      = Initializes the sparse correlation structure fo the PIC_BAND model
%                          The value for truncated has to be a cell array of type 
%                          {x, R}, where x is the matrix of input (size n x nin) and R is the radius for truncation.
%                          
%                          If value is {x, R, 1} an information about the sparsity structure is printed and plotted.
%                          
%
%       
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
                    if size(varargin{i+1},2)~=gp.nin
                        error('The size of X_u has to be u x nin.')
                    else
                        gp.X_u = varargin{i+1};
                        gp.nind = size(varargin{i+1},1);
                    end
                  case 'blocks'
                    init_blocks(varargin{i+1})
                  case 'truncated'
                    init_truncated(varargin{i+1})
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
              case 'type'
                gp.type = varargin{i+1};
              case 'X_u'
                if size(varargin{i+1},2)~=gp.nin
                    error('The size of X_u has to be u x nin.')
                else
                    gp.X_u = varargin{i+1};
                    gp.nind = size(varargin{i+1},1);
                end
              case 'blocks'
                init_blocks(varargin{i+1})
              case 'truncated'
                init_truncated(varargin{i+1})
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


    function init_blocks(var)
        
        if length(var) ~= 3
            error('Wrong kind of value for the clustering type! See help gp_init!')
        else
            x = var{2};
            switch var{1}
              case 'manual'
                gp.blocktype = 'manual';
                gp.tr_index = var{3};
              case 'farthest_point'
                gp.blocktype = 'farthest_point';
                gp.blockcent = var{3};
              otherwise
                error('Wrong value for the clustering type! See help gp_init!')
            end
        end
    end
    
    function init_truncated(var)
        if length(var) < 2
            error('Wrong kind of value for the truncated type! See help gp_init!')
        end

        x= var{1};
        R= var{2};
        gp.truncated_R = R;
        n = size(x,1);
        
        if size(x,2)~=gp.nin
            error('The size of x for "truncated" has to be n x nin!')
        end
                
        C = sparse([],[],[],n,n,0);
        for i1=2:n
            i1n=(i1-1)*n;
            for i2=1:i1-1
                ii=i1+(i2-1)*n;
                D = 0;
                for i3=1:gp.nin
                    D =D+(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
                end
                if sqrt(D) < R
                    C(ii)=1;
                    C(i1n+i2)=C(ii); 
                end
            end
        end
        C= C + speye(n,n);
        if length(var) == 3
            if var{3} == 1
                spy(C)
                title('the sparsity structure of the covariance matrix')
                fprintf('The density of the sparse correlation matrix is %f \n',nnz(C)/prod(size(C)))
            end
        end
                
        [I,J,s] = find(C);
        gp.tr_index = [I(:) J(:)];
    end
end