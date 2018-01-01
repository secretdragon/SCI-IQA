data=round(double(data),4);
score=round(double(score),4);
% label=[];
% label=round([DMOS(:,6);DMOS(:,7);DMOS(:,8);DMOS(:,9)],4);

final = zeros(196,0);
for i = 1:196
    [m,n]=find(score==label2(i));
    k=length(m);
    count=0;
    for j = 1:k
        count=count+data(m(j),n(j));
    end
    final(i)=count/k;
end