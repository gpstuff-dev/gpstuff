function q = loglogunif_p(pr)
% LOGLOGUNIF_P Create uniform prior for the log(log(parameter))
%
%        Description
%        Q = LOGLOGUNIF_P returns a structure that specifies uniform prior 
%        for the logarithm of the parameter. LOGLOGUNIF_P creates
%        function handles to evaluate energy and gradient.
%        

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% create function handles to compute error and gradient for the 
% weight hyper-parameters
    
q.f='loglogunif'; 
q.fe=str2fun('loglogunif_e');
q.fg=str2fun('loglogunif_g');
q.a = [];