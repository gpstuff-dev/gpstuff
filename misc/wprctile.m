function y = wprctile(x, p, w)
% wprctile - 
x=sort(x);
p=p./100;
y=p;
ww=cumsum(w);ww=ww./ww(end);
for j=1:length(p)
  wi=min(find(ww>=p(j)));
  if wi==1
    y(j)=x(1);
  else
    w1=ww(wi-1);x1=x(wi-1);
    y(j)=x1+(x(wi)-x1).*(p(j)-w1)./(ww(wi)-w1);
  end
end
