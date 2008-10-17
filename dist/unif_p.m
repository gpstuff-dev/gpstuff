function q = unif_p(pr)
% UNIF_P Create uniform prior for the parameter
%
%        Description
%        Q = UNIF_P returns a structure that specifies uniform prior 
%        for the parameter. LOGUNIF_P creates function handles to 
%        evaluate energy and gradient. 
%        

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% create function handles to compute error and gradient for the 
% weight hyper-parameters
    
q.f='unif'; 
q.fe=str2fun('unif_e');
q.fg=str2fun('unif_g');
q.a = [];