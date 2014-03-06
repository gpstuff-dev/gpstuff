function diag_install

    if ispc   % A windows version of Matlab
      if ~exist('OCTAVE_VERSION','builtin')        
        mex -O -output bbprctile winCsource\bbprctile.c
      else       
        mex --output bbprctile.mex winCsource\bbprctile.c
      end
    else
      if ~exist('OCTAVE_VERSION','builtin')
        mex -O -output bbprctile linuxCsource/bbprctile.c
      else
        mex --output bbprctile.mex linuxCsource/bbprctile.c
      end
    end