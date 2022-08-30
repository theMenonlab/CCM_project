clear all;
D = uigetdir()

i = 1;
for k = 1:220
    Ffileccm =fullfile(D, strcat("test-",num2str(k-1),"-y.png"));
    Ffilepred =fullfile(D, strcat("test-",num2str(k-1),"-f_y.png"));

    j = mod(k-1,11)
    if j==0
        refpred = imread(Ffilepred);
    else
        noisepred = imread(Ffilepred);
        ssim_pred(i,j) = ssim(noisepred,refpred);
        pcorr_pred(i,j) = corr2(noisepred,refpred);
    end


    if j==0
        refccm= imread(Ffileccm);
    else
        noiseccm = imread(Ffileccm);
        ssim_ccm(i,j) = ssim(noiseccm,refccm);
        pcorr_ccm(i,j) = corr2(noiseccm,refccm);
    end

       
    if j == 10
        i = i+1;
    end
   
end

j = 1;
k = 1;
for i = 1:size(pcorr_ccm,1)
    if j > 4
        j = 1;
        k = k+1;
    end
    stdinput(j,k) = std(ssim_ccm(i,:));
    stdpred(j,k) = std(ssim_pred(i,:));
    minput(j,k) = mean(ssim_ccm(i,:));
    mpred(j,k) = mean(ssim_pred(i,:));
    
    stdcorrinput(j,k) = std(pcorr_ccm(i,:));
    stdcorrpred(j,k) = std(pcorr_pred(i,:));
    mcorrinput(j,k) = mean(pcorr_ccm(i,:));
    mcorrpred(j,k) = mean(pcorr_pred(i,:));
    
    j = j+1;
end

x = 1:1:5;
figure
hold on
for i = 1:size(minput,2)
%     errorbar(x,minput(i,:),stdinput(i,:))
%     scatter(mean(minput(:,i)),mean(mpred(:,i)))
    plot(
end
hold off
% xlim([0.5 5.5])
% xticks(0:1:5)



% figure
% hold on;
% for i= 1:size(ssim_ccm,1)
% %     for j = 1: size(ssim_ccm,2)
%     scatter(ssim_ccm(i,1),ssim_pred(i,1));
% %     end
% end  
% hold off;


%{
% % % % % % % % % % % % % % % % % % % 
% Standard deviation and mean plot
for i = 1:10
    stdinput(i) = std(ssim_ccm(:,i));
    stdpred(i) = std(ssim_pred(:,i));
    minput(i) = mean(ssim_ccm(:,i));
    mpred(i) = mean(ssim_pred(:,i));
end
figure
hold on
errorbar(minput(1),mpred(1),stdpred(1),stdpred(1),stdinput(1),stdinput(1),'r')
errorbar(minput(2),mpred(2),stdpred(2),stdpred(2),stdinput(2),stdinput(2),'b')
errorbar(minput(3),mpred(3),stdpred(3),stdpred(3),stdinput(3),stdinput(3),'g')
legend('2% noise','5% noise','10% noise')
xlim([0.5 1])
ylim([0.4 1])
xlabel('CCM input SSIM')
ylabel('Predicted output SSIM')
title('SCNet SSIM sensitivity with percentage Gaussion noise (std range)')
hold off
% % % % % % % % % % % % % % % % % % % 
% Max min range error and mean plot
figure
for i = 1:3
    minput(i) = mean(ssim_ccm(:,i));
    mpred(i) = mean(ssim_pred(:,i));
    maxip(i) = max(ssim_ccm(:,i))-minput(i)
    minip(i) = minput(i)-min(ssim_ccm(:,i))
    maxpred(i) = max(ssim_pred(:,i))-mpred(i)
    minpred(i) = mpred(i)-min(ssim_pred(:,i))
end
hold on
errorbar(minput(1),mpred(1),minpred(1),maxpred(1),minip(1),maxip(1),'r')
errorbar(minput(2),mpred(2),minpred(2),maxpred(2),minip(2),maxip(2),'b')
errorbar(minput(3),mpred(3),minpred(3),maxpred(3),minip(3),maxip(3),'g')
legend('2% noise','5% noise','10% noise')
xlim([0.5 1])
ylim([0.4 1])
xlabel('CCM input SSIM')
ylabel('Predicted output SSIM')
title('SCNet SSIM sensitivity with percentage Gaussion noise (min max range)')
hold off
}%