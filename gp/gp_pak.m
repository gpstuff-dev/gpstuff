function w = gp_pak(gp, param)
%GP_PAK	 Combine GP hyper-parameters into one vector.
%
%	Description
%	W = GP_PAK(GP) takes a Gaussian Process data structure GP and
%	combines the hyper-parameters into a single row vector W.
%
%	The ordering of the parameters in HP is defined by
%	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
%
%	See also
%	GP_UNPAK
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin > 1   % Pak inducing inputs
   % $$$ 
% $$$ if isfield(gp, 'X_u')
% $$$     w = [w gp.X_u(:)'];
% $$$ end 
    
else
    w=[];
    ncf = length(gp.cf);
    
    for i=1:ncf
        gpcf = gp.cf{i};
        w = feval(gpcf.fh_pak, gpcf, w);
    end
    
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn
            noise = gp.noise{i};
            w = feval(noise.fh_pak, noise, w);
        end
    end
    w = log(w);
end