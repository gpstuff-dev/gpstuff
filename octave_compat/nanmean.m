function m = nanmean(x)

nnanx=~isnan(x);
x(~nnanx)=0;
m=sum(x)./sum(nnanx);
m(isinf(m))=NaN;
end

