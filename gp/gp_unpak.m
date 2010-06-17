function gp = gp_unpak(gp, w, param)
%GP_UNPAK	 Set GP hyper-parameters from vector to structure
%
%	Description
%        GP = GP_UNPAK(GP, W, PARAM) takes an Gaussian Process data
%        structure GP and a parameter vector W, and returns a Gaussian
%        Process data structure identical to the input, except that
%        the parameters has been set to the ones in W. PARAM defines
%        which parameters are present in the W vector. If PARAM is not
%        given the function unpacks all parameters.
%
%        Each of the following strings in PARAM defines one group of
%        parameters to pack:
%         'covariance'     = unpack hyperparameters of covariance 
%                            function
%         'inducing'       = unpack iducing inputs
%                            W = gp.X_u(:)
%         'likelihood'     = unpack parameters of likelihood
%
%        By compining the strings one can pack more than one group of
%        parameters. For example:
%         'covariance+inducing' = unpack covariance function
%                                 parameters and inducing inputs
%         'covariance+likelih' = unpack covariance function parameters
%                                of likelihood parameters
%
%        Inside each group (such as covariance functions) the
%        parameters to be unpacked is defined by the existence of a
%        prior structure. For example, if GP has two covariance
%        functions but only the first one has prior for its parameters
%        then only the parameters of the first one are unpacked. Thus,
%        also inducing inputs require prior if they are to be
%        optimized.
% 
%        gp_pak and gp_unpak functions are used when GP parameters are
%        optimized or sampled with gp_mc. The same PARAM string should
%        be given for all of these functions.
%
%        See also
%        GP_PAK
%

% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 3
    param = gp.infer_params;
end

if size(w,1) > 1
    error(' The vector to be packed has to be row vector! \n')
end

% Unpack the hyperparameters of covariance functions
if ~isempty(strfind(param, 'covariance'))
    ncf = length(gp.cf);
    
    for i=1:ncf
        gpcf = gp.cf{i};
        [gpcf, w] = feval(gpcf.fh_unpak, gpcf, w);
        gp.cf{i} = gpcf;
    end
    
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn
            noise = gp.noise{i};
            [noise, w] = feval(noise.fh_unpak, noise, w);
            gp.noise{i} = noise;
        end
    end
end

% Unpack the inducing inputs
if ~isempty(strfind(param, 'inducing'))
    if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
        lu = length(gp.X_u(:));
        gp.X_u = reshape(w(1:lu), size(gp.X_u));
        if lu < length(w)
            w = w(lu+1:end);
        end
    end
end

% Unpack the hyperparameters of likelihood function
if ~isempty(strfind(param, 'likelihood'))
    if isstruct(gp.likelih)
        [lik w] = feval(gp.likelih.fh_unpak, w, gp.likelih);
        gp.likelih = lik;
    end
end

