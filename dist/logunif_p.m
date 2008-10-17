function q = logunif_p(pr)
% LOGUNIF_P Create uniform prior for the logarithm of the parameter
%
%        Description
%        Q = LOGUNIF_P returns a structure that specifies uniform prior 
%        for the logarithm of the parameter. LOGUNIF_P creates
%        function handles to evaluate energy and gradient.
%        

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% create function handles to compute error and gradient for the 
% weight hyper-parameters
    
q.f='logunif'; 
q.fe=str2fun('logunif_e');
q.fg=str2fun('logunif_g');
q.a = [];