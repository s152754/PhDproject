clear all; close all; clc;
v = VideoReader('D:\ExperimentGlycerol\GlycerolT_V02_F025_10.MP4'); % change to directory harde schijf
crop1 = 1; crop2 = 1080; crop3 = 250; crop4 = 1400;
for i = 1:15
    iframe = 48+i;
    frame1 = read(v,iframe);
    frame = frame1(crop1:crop2,crop3:crop4);
    figure()
    imshow(frame)
end
