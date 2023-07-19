clear all; close all; clc;
directory = 'C:\Users\s152754\PycharmProjects\UncertaintyQuantification\rheometerPVP\'
for i = 1:5
    datafile = importdata(directory + "20221223_SRST_S" + num2str(i) +".txt")
    remnan = datafile.data(:,:);
    remnandat = remnan(sum(isnan(remnan),2)==0,:);
    A = remnandat(:,2:3);
    T = array2table(A,'VariableNames',{'eta', 'rate'})

    writetable(T,directory + "20221219_PVP_SRST_S" + num2str(i+5) + ".csv")
end

%% glycerol Rout
clear all; close all; clc;
directory = "C:\Users\s152754\PycharmProjects\nutils-squeezeflow\RoutExp\";

for i = 1:10
    IR = importdata(directory+filename(i));
    remnan = IR.data(:,:);
    remnandat = remnan(sum(isnan(remnan),2)==0,:);
    A = remnandat(:,2:3);
    if i < 4        
        T = array2table(A,'VariableNames',{'eta', 'rate'});
        writetable(T,directory + "sha2_SRST_S" + num2str(i) + ".csv")
    else
        B = flipud(A);
        TB = array2table(B,'VariableNames',{'eta', 'rate'})
        writetable(TB,directory + "sha2_SRST_S" + num2str(i) + ".csv")
    end
end