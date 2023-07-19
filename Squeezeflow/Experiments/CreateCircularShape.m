clear all; close all; clc;
% Load data
iframes = importdata('InitialFrames.txt');
icase           = 1; %glycerol: F = 1, F025 = 2, F05 = 3, V02 = 4
SecFrame        = 4;
Frames_Between  = SecFrame;
Frame_init      = iframes.data(:,icase);
Frame_middle    = Frame_init+Frames_Between;

crop1           = 1; 
crop2           = 1080; 
crop3           = 250; 
crop4           = 1400;

sens = 0.9;
ID = 1;
for i=4%i = 1:length(Frame_init)
    filename        = ["D:\ExperimentGlycerol\GlycerolT_V01_" + num2str(i) + ".mp4"] %#ok<NOPTS> 
    v               = VideoReader(filename); %#ok<TNMLP> 

    readFramei      = read(v,Frame_init(i)); 
    Frames_Between = SecFrame;
    ID = 1;
    while Frames_Between < v.NumFrames-Frame_init(i)
        Frame_middle    = Frame_init+Frames_Between;
        readFramem      = read(v,Frame_middle(i));
        Frame_i         = readFramei(crop1:crop2,crop3:crop4);
        Frame_m         = readFramem(crop1:crop2,crop3:crop4);    
        Frame_sub       = imsubtract(Frame_i,Frame_m);
        [BW,intensity]  = createBWimages(Frame_sub);


        figure()
        imshow(Frame_i)
        title("t=" + num2str(0.00,'%.2f') + "s")
            
            

        if Frames_Between == SecFrame
            [centersBright, radiiBright] = imfindcircles(BW,[70 100],'ObjectPolarity','bright','Sensitivity',0.92); %PVPV2 Rmin 30 Rmax 100 %Rmin = 50, Rmax = 100 voor V02 en Rmax = 80 voor V01; shampoo 80 160; PVP 70 120
%             figure()
%             imshow(BW)
%             hold on
%             viscircles(centersBright, radiiBright,'Color','b');
        end
%         Frames_Between %marges verkleinen, dan wordt het beter geschat!
        if Frames_Between < 25
            Rmin = 90;  Rmax = 180;     % glycerol 80, 200; shampoo 100; 
        elseif Frames_Between > 25 && Frames_Between< 500  
            Rmin = 150; Rmax = 230;     % glycerol 150, 350; shampoo 150; pvp 120; pvpV2 100
        elseif Frames_Between < 5000
            Rmin = 200; Rmax = 310;     % glycerol 200, 500; shampoo 150; pvp 150; pvpV2 120
        elseif Frames_Between < 20000
            Rmin = 280; Rmax = 400;     %F05 Rmin = 300, F en F025 hebben 200 en 600; shampoo 150; pvpV2 150; PVP200
        elseif Frames_Between < v.NumFrames
            Rmin = 280; Rmax = 400;     % F025, F -> 300, 650   F05 --> 350, 630; v02 350; shampoo 180
        end

        [centersDark, radiiDark] = imfindcircles(BW,[Rmin Rmax],'ObjectPolarity','dark','Sensitivity',sens);

        while isempty(centersDark) && sens < 1
            sens  = sens + 0.01;
            [centersDark, radiiDark] = imfindcircles(BW,[Rmin Rmax],'ObjectPolarity','dark','Sensitivity',sens);
        end  


        if isempty(radiiDark) || length(radiiDark) > 1
            Rout(i,ID) = NaN;
            time(i,ID) = NaN;
            CentOut(i,ID,1:2) = NaN;
        else
            [muptl,sigptl]      = pixeltolength(); 
            Rout(i,ID)          = radiiDark*muptl;
            Rin(i,1)            = radiiBright*muptl;
            CentOut(i,ID,1:2)   = centersDark*muptl;
            time(i,ID)          = Frames_Between/v.FrameRate;
        end
%         fprintf('figure %d, frame number = %d, time = %d \n',ID+1,Frames_Between, time(i,ID))
%         figure()
%         imshow(BW)
%         hold on
%         viscircles(centersDark, radiiDark,'Color','r','LineWidth',1);% linewidth 5 for images report
%         viscircles(centersBright, radiiBright,'Color','g','LineWidth',5);
     
        if round(time(i,ID)) == 1 || round(time(i,ID)) == 6 || round(time(i,ID)) == 14 || round(time(i,ID)) == 31 || round(time(i,ID)) == 69
            figure()
            imshow(Frame_m)
            title("t=" + num2str(time(i,ID),'%.2f') + "s")
        end

        Frames_Between = round(Frames_Between*1.5);
        sens           = 0.9;
        ID             = ID + 1;        
    end 
    centersBright      = centersBright*muptl;
end

Radius  = [Rin Rout];
Time    = [zeros(length(Frame_init),1) time];

muR     = nanmean(Radius);
stdR    = nanstd(Radius);

% %% Save experimental data
% filename = "RoutExp_GlyV2F025.csv";
% directory = 'C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\';
% % T = table(Time', Radius', muR', stdR');
% % table_path = fullfile(directory, filename);
% % writetable(T,table_path);
% T = table(Time', Radius', muR', stdR');
% T.Properties.VariableNames(1:end) = {'Time','Radius', 'meanRadius', 'stdRadius'};
% table_path = fullfile(directory, filename);
% writetable(T,table_path);

%% check with old measurements
%{
Rold = importdata("Gly_V01_F.txt");
muT = Rold.data(:,1);
muR = Rold.data(:,2);
stdR = Rold.data(:,3);

figure()
plot(Time(1,:),Radius(1,:),'-ob')
hold on
plot(Time(2,:),Radius(2,:),'-og')
plot(Time(3,:),Radius(3,:),'-or')
errorbar(muT,muR,2*stdR,'k')
%}


