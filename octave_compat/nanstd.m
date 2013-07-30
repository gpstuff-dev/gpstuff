function sd = nanstd(x)

nanm=nanmean(x);
nnanx=~isnan(x);
x=bsxfun(@minus, x, nanm).^2;
x(~nnanx)=0;
sd=sqrt(sum(x)./(sum(nnanx)-1));
sd(isinf(sd))=NaN;
end

