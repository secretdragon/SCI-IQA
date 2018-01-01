% % output=DMOS(:,10);
% % DMOS2=DMOS2(785:980)
% output=[output,DMOS(:,11),DMOS(:,12),DMOS(:,13),DMOS(:,14),DMOS(:,15),DMOS(:,16),DMOS(:,17),DMOS(:,18),DMOS(:,19),DMOS(:,1),DMOS(:,20),DMOS(:,2),DMOS(:,3),DMOS(:,4),DMOS(:,5),DMOS(:,6),DMOS(:,7),DMOS(:,8),DMOS(:,9)];
final = final';
 SROCC= corr(final,label2,'type','Spearman');
 PLCC = corr(final,label2,'type','Pearson');
%  newD2 = newD(1:231084);
%  newDT = newD(231085:295764);
% DMOS32=DMOS(1:7,1);
   