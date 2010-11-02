function [x] = denormdata(xn,xmean,xstd)
%DENORMDATA  De-normalize normalized data
%
%  Description
%    X = DENORMDATA(XN,XMEAN,XSTD) de-normalize XN using
%    precomputed XMEAN and XSTD.
%
%  See also NORMDATA
%
  
% Copyright (c) 2010 Aki Vehtari

  if nargin<3
    error('Too few arguments')
  end
  x=bsxfun(@plus,bsxfun(@times,xn,xstd),xmean);
