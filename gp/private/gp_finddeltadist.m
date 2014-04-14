function deltadist = gp_finddeltadist(cf)
% FINDDELTADIST - Find which covariates are using delta distance
%   
% Copyright (c) 2011      Aki Vehtari

 % This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

deltadist=[];
if ~iscell(cf) && isfield(cf,'cf')
  deltadist=union(deltadist,gp_finddeltadist(cf.cf));
else
  for cfi=1:numel(cf)
    if isfield(cf{cfi},'cf')
      deltadist=union(deltadist,gp_finddeltadist(cf{cfi}.cf));
    else
      if isfield(cf{cfi},'metric')
        if isfield(cf{cfi}.metric,'deltadist')
          deltadist=union(deltadist,find(cf{cfi}.metric.deltadist));
        end
      elseif ismember(cf{cfi}.type,{'gpcf_cat' 'gpcf_mask'}) && ...
          isfield(cf{cfi},'selectedVariables')
        deltadist=union(deltadist,cf{cfi}.selectedVariables);
      end
    end
  end
end
end
