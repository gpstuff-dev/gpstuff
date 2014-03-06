function mc_install

    
    if ispc   % A windows version of Matlab
      if ~exist('OCTAVE_VERSION','builtin')        
        mex -O -output bbmean winCSource\bbmean.c
        mex -O -output resampres winCsource\resampres.c winCsource\binsgeq.c
        mex -O -output resampsim winCsource\resampsim.c winCsource\binsgeq.c
        mex -O -output resampstr winCsource\resampstr.c winCsource\binsgeq.c
        mex -O -output resampdet winCsource\resampdet.c winCsource\binsgeq.c        
      else
        mex --output bbmean.mex winCSource\bbmean.c
        mex --output resampres.mex winCsource\resampres.c winCsource\binsgeq.c
        mex --output resampsim.mex winCsource\resampsim.c winCsource\binsgeq.c
        mex --output resampstr.mex winCsource\resampstr.c winCsource\binsgeq.c
        mex --output resampdet.mex winCsource\resampdet.c winCsource\binsgeq.c 
      end
    else
      if ~exist('OCTAVE_VERSION','builtin')        
        mex -O -output bbmean linuxCsource/bbmean.c
        mex -O -output resampres linuxCsource/resampres.c linuxCsource/binsgeq.c
        mex -O -output resampsim linuxCsource/resampsim.c linuxCsource/binsgeq.c
        mex -O -output resampstr linuxCsource/resampstr.c linuxCsource/binsgeq.c
        mex -O -output resampdet linuxCsource/resampdet.c linuxCsource/binsgeq.c
      else
        mex --output bbmean.mex linuxCsource/bbmean.c
        mex --output resampres.mex linuxCsource/resampres.c linuxCsource/binsgeq.c
        mex --output resampsim.mex linuxCsource/resampsim.c linuxCsource/binsgeq.c
        mex --output resampstr.mex linuxCsource/resampstr.c linuxCsource/binsgeq.c
        mex --output resampdet.mex linuxCsource/resampdet.c linuxCsource/binsgeq.c
      end
    end
