function version = gpstuff_version(str)
% Return version of currently used gpstuff
% 
% gpstuff_version() returns string including release number and release
%                   date
%
% gpstuff_version('date') returns the release date (string)
%
% gpstuff_version('release') returns the release number (double)
%
% Copyright (c) 2013 Ville Tolvanen


if nargin==0
  version='4.2_2013-06-14';
elseif isequal(str, 'date')
  version='2013-06-14';
elseif isequal(str, 'release')
  version=4.2;
end
  


end

