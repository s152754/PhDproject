function [muptl,sigptl] = pixeltolength() 

% 1 pixel heeft x lengte in m
mm = 1e-3;
% v = VideoReader('filmpjes/DSCN0155.mp4');
% frame = read(v,120);
% frame1 = frame(1:1080,250:1400);
% level = graythresh(frame1);
% BW = imbinarize(frame1,level);
% figure()
% imshow(frame1)

%%%%%%%% wide range %%%%%%%%%%%
%%% maximum width 
l = 92*mm;
% 1)
x1 = 1284 - 384;
y1 = 305 - 287;
d1 = sqrt(x1^2+y1^2);
ptl1 = l/d1;

% 2)
x2 = 1278 - 380;
y2 = 521 - 503;
d2 = sqrt(x2^2+y2^2);
ptl2 = l/d2;

% 3)
x3 = 1271 - 372;
y3 = 932 - 912;
d3 = sqrt(x3^2+y3^2);
ptl3 = l/d3; 

%%% maximum height
lv = 94*mm;
% 1)
xv1 = 410 - 427;
yv1 = 972 - 52;
dv1 = sqrt(xv1^2+yv1^2);
ptlv1 = lv/dv1;

% 2)
xv2 = 781 - 799;
yv2 = 980 - 60;
dv2 = sqrt(xv2^2+yv2^2);
ptlv2 = lv/dv2;

% 3)
xv3 = 1192 - 1211;
yv3 = 989 - 66;
dv3 = sqrt(xv3^2 + yv3^2);
ptlv3 = lv/dv3;

ptl = [ptl1 ptl2 ptl3 ptlv1 ptlv2 ptlv3];
muptl = mean(ptl);
sigptl = std(ptl);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% small range %%%%%%%%
%%%% maximum height
ls = 40*mm;
% 1)
xs1 = 481 - 471;
ys1 = 289 - 681;
ds1 = sqrt(xs1^2+ys1^2);
ptls1 = ls/ds1;

% 2)
xs2 = 814 - 806;
ys2 = 296 - 688;
ds2 = sqrt(xs2^2+ys2^2);
ptls2 = ls/ds2;

% 3)
xs3 = 1206 - 1198;
ys3 = 303 - 696;
ds3 = sqrt(xs3^2+ys3^2);
ptls3 = ls/ds3; 

%%% maximum length
lsv = 40*mm;
% 1)
xsv1 = 673 - 1064;
ysv1 = 508 - 516;
dsv1 = sqrt(xsv1^2+ysv1^2);
ptlsv1 = lsv/dsv1;

% 2)
xsv2 = 608 - 1072;
ysv2 = 136 - 144;
dsv2 = sqrt(xsv2^2+ysv2^2);
ptlsv2 = lsv/dsv2;

% 3)
xsv3 = 665 - 919;
ysv3 = 919 - 927;
dsv3 = sqrt(xsv3^2 + ysv3^2);
ptlsv3 = lsv/dsv3;

ptls = [ptls1 ptls2 ptls3 ptlsv1 ptlsv2 ptlsv3];
muptls = mean(ptls);
sigptls = std(ptls);


% figure()
% plot(1,ptl1,'o')
% hold on
% plot(1,ptl2,'o')
% plot(1,ptl3,'o')
% plot(1,ptlv1,'o')
% plot(1,ptlv2,'o')
% plot(1,ptlv3,'o')
%}

end

