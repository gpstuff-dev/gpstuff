function y = crosstab(x)
% Count numer of occurences of elements of x in x

ux=unique(x);
y=zeros(size(ux));
for i=1:length(ux)
  y(i) = sum(x==ux(i));
end
if size(y,1)==1
  y=y';
end

end

