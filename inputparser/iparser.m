function ip = iparser(ip, action, varargin)

if ~exist('OCTAVE_VERSION', 'builtin')
  ip.(sprintf('%s', action))(varargin{:})
else
  ip=ip.(sprintf('%s', action))(varargin{:});
end

end

