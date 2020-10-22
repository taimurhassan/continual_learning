clc
clear all
close all

load results.mat

conventional = true;

if conventional
    neg = 5;

    actual = actual_combined;
    predicted = predicted_combined_fine_tuning;
    
    TP =0;
    FP = 0;
    FN = 0;
    TN = 0;
    for i = 1:length(actual)
        if actual(i) == predicted(i) 
            if predicted(i) == neg
                TN = TN+1;
            else
                TP = TP+1;
            end
        else
            if actual(i) == neg 
                FP = FP + 1;
            else
                FN = FN + 1;
            end
        end
    end

else
    pos = 1;

    actual = actual_gdxray;
    predicted = predicted_gdxray_resnet101;

    TP =0;
    FP = 0;
    FN = 0;
    TN = 0;
    for i = 1:length(actual)
        if actual(i) == predicted(i) 
            if predicted(i) == pos 
                TP = TP+1;
            else 
                TN = TN+1;
            end
        else
            if actual(i) == pos 
                FN = FN + 1;
            else
                FP = FP + 1;
            end
        end
    end
end

TotalSamples = TP+TN+FP+FN

c = [TP FN; FP TN]

acc = (TP+TN)/(TP+TN+FP+FN)
sen = TP/(TP+FN)
spe = TN/(TN+FP)
pre = TP/(TP+FP)
F1 = (2*sen*pre)/(sen+pre)