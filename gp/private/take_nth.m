function x = take_nth(x,nth)
%TAKE_NTH    Take n'th parameters from MCMC-chains
%
%   x = take_nth(x,n) returns chain containing only
%   n'th simulation sample 
%
%   See also
%     THIN, JOIN
    
% Copyright (c) 1999 Simo S�rkk�
% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
    
    if nargin < 2
        n = 1;
    end
    
    [m,n]=size(x);

    if isstruct(x)
        if (m>1 | n>1)
            % array of structures
            for i=1:(m*n)
                x(i) = take_nth(x(i),n);
            end
        else
            % single structure
            names = fieldnames(x);
            for i=1:size(names,1)
                value = getfield(x,names{i});
                if length(value) > 1
                    x = setfield(x,names{i},take_nth(value,nth));
                elseif iscell(value)
                    x = setfield(x,names{i},{take_nth(value{1},nth)});
                elseif isstruct(value)
                    x = setfield(x,names{i},take_nth(value,nth));
                end
            end
        end
    elseif iscell(x)
        % cell array
        for i=1:(m*n)
            x{i} = take_nth(x{i},nth);
        end
    elseif m > 1
        x = x(nth,:);
    end
end
