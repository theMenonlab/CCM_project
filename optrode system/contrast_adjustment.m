clear;

path1 = 'I:\Plant\ref1\';
path2 = 'I:\Plant\ref1_contrast\';

for i = 0:26088
    filename = num2str(i);      
    load([path1,filename]);
    objRef1 = double(objRef1);
    objRef1 = objRef1./max(max(objRef1));
    A1 = adapthisteq(objRef1);
    objRef1_ad_exp1 = exp(A1.*2-1);
    objRef1 = exp((objRef1_ad_exp1./max(max(objRef1_ad_exp1))).*2-1);
    
    save([path2,num2str(i)],'objRef1');                
end