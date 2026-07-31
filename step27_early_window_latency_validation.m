clear; clc; close all;
rng(20260718,'twister');

paperRoot=fileparts(mfilename('fullpath'));
resultDir=fullfile(paperRoot,'additional_validation_results','reviewer_improvements');
if ~exist(resultDir,'dir'), mkdir(resultDir); end

raw=load(fullfile(paperRoot,'mill','mill.mat'));
mill=raw.mill;
data=load(fullfile(paperRoot,'mill','results','quality_aware_with_time.mat'));
T=data.timeAwareTable;

fractions=[0.25 0.50 0.75 1.00];
numberOfFractions=numel(fractions);
Fs=250;
lambdaGrid=logspace(-4,4,17);
gateGrid=[0.01 0.10 0.50];
caseList=unique(T.CaseID);
numberOfCases=numel(caseList);
numberOfRecords=height(T);
signalNames=["smcAC","smcDC","vib_table","vib_spindle","AE_table","AE_spindle"];

y=T.VB;
caseID=T.CaseID;
XProcess=T{:,{'DOC','Feed','Material'}};
elapsed=T.ElapsedTime;
originalIndex=T.OriginalIndex;
smcDCBlock=14:26;
otherSensorMask=true(1,78);
otherSensorMask(smcDCBlock)=false;

predictionMatrix=nan(numberOfRecords,numberOfFractions);
caseMAE=nan(numberOfCases,numberOfFractions);
selectedLambdaAll=nan(numberOfCases,numberOfFractions);
selectedLambdaQuality=nan(numberOfCases,numberOfFractions);
selectedGate=nan(numberOfCases,numberOfFractions);
featureExtractionMs=nan(numberOfFractions,1);
inferenceMs=nan(numberOfFractions,1);
validationSeconds=nan(numberOfFractions,1);
meanEndpointSeconds=nan(numberOfFractions,1);
meanObservedSignalPercent=nan(numberOfFractions,1);
allFeatureMatrices=cell(numberOfFractions,1);
allQualityRatios=cell(numberOfFractions,1);

fprintf('========== Early-window and latency validation ==========\n');
for fractionIndex=1:numberOfFractions
    currentFraction=fractions(fractionIndex);
    fprintf('\nStable-window fraction %.0f%%\n',100*currentFraction);

    % One warm-up call removes first-call JIT cost from runtime reporting.
    calculateRecordFeatures(mill(originalIndex(1)),signalNames,currentFraction,Fs);
    extractionTimes=nan(3,1);
    for timingRepeat=1:3
        timer=tic;
        [XSensor,qualityRatio,endpointSeconds,observedPercent]= ...
            buildWindowFeatures(mill,originalIndex,signalNames,currentFraction,Fs);
        extractionTimes(timingRepeat)=toc(timer);
    end
    featureExtractionMs(fractionIndex)=1000*median(extractionTimes)/numberOfRecords;
    meanEndpointSeconds(fractionIndex)=mean(endpointSeconds);
    meanObservedSignalPercent(fractionIndex)=mean(observedPercent);
    allFeatureMatrices{fractionIndex}=XSensor;
    allQualityRatios{fractionIndex}=qualityRatio;

    XAll=[XProcess,XSensor,elapsed,elapsed.^2];
    XQuality=[XProcess,XSensor(:,otherSensorMask),elapsed,elapsed.^2];

    timer=tic;
    for caseIndex=1:numberOfCases
        isTest=caseID==caseList(caseIndex);
        isTrain=~isTest;
        [~,~,predictionMatrix(isTest,fractionIndex), ...
            selectedLambdaAll(caseIndex,fractionIndex), ...
            selectedLambdaQuality(caseIndex,fractionIndex), ...
            selectedGate(caseIndex,fractionIndex)] = ...
            groupedGatedRidgePrediction(XAll,XQuality,qualityRatio,y,caseID, ...
            isTrain,isTest,lambdaGrid,gateGrid);
    end
    validationSeconds(fractionIndex)=toc(timer);

    for caseIndex=1:numberOfCases
        mask=caseID==caseList(caseIndex);
        caseMAE(caseIndex,fractionIndex)=mean(abs( ...
            predictionMatrix(mask,fractionIndex)-y(mask)));
    end

    % Runtime-only deployment profile: fit on all records using the median
    % nested hyperparameters. Accuracy is never computed from this fit.
    lambdaAll=median(selectedLambdaAll(:,fractionIndex));
    lambdaQuality=median(selectedLambdaQuality(:,fractionIndex));
    gateThreshold=mode(selectedGate(:,fractionIndex));
    allModel=fitRidgeRuntime(XAll,y,lambdaAll);
    qualityModel=fitRidgeRuntime(XQuality,y,lambdaQuality);
    inferenceFunction=@() deploymentPrediction(allModel,qualityModel, ...
        XAll,XQuality,qualityRatio,gateThreshold);
    inferenceMs(fractionIndex)=1000*timeit(inferenceFunction)/numberOfRecords;
end

%% Overall metrics
MAE=nan(numberOfFractions,1); RMSE=MAE; R2=MAE; MacroMAE=MAE;
P95AbsoluteError=MAE; MaximumAbsoluteError=MAE;
for fractionIndex=1:numberOfFractions
    residual=predictionMatrix(:,fractionIndex)-y;
    MAE(fractionIndex)=mean(abs(residual));
    RMSE(fractionIndex)=sqrt(mean(residual.^2));
    R2(fractionIndex)=1-sum(residual.^2)/sum((y-mean(y)).^2);
    MacroMAE(fractionIndex)=mean(caseMAE(:,fractionIndex));
    P95AbsoluteError(fractionIndex)=prctile(abs(residual),95);
    MaximumAbsoluteError(fractionIndex)=max(abs(residual));
end
StableWindowFraction=fractions(:);
AvailableOriginalSignalPercent=100*meanObservedSignalPercent;
MeanObservationEndpointSeconds=meanEndpointSeconds;
FeatureExtractionMillisecondsPerRecord=featureExtractionMs;
RidgeInferenceMillisecondsPerRecord=inferenceMs;
TotalOnlineComputeMillisecondsPerRecord=featureExtractionMs+inferenceMs;
NestedValidationSeconds=validationSeconds;
overall=table(StableWindowFraction,AvailableOriginalSignalPercent, ...
    MeanObservationEndpointSeconds,MAE,RMSE,R2,MacroMAE,P95AbsoluteError, ...
    MaximumAbsoluteError,FeatureExtractionMillisecondsPerRecord, ...
    RidgeInferenceMillisecondsPerRecord, ...
    TotalOnlineComputeMillisecondsPerRecord,NestedValidationSeconds);

%% Paired case-level inference versus the full stable window
comparisonCount=numberOfFractions-1;
ComparatorFraction=fractions(1:end-1)'; IndependentCaseN=repmat(numberOfCases,comparisonCount,1);
MeanCaseMAEDifference=nan(comparisonCount,1); BootstrapCILower=MeanCaseMAEDifference;
BootstrapCIUpper=MeanCaseMAEDifference; RawP=MeanCaseMAEDifference;
SignedRankStatistic=MeanCaseMAEDifference;
for comparisonIndex=1:comparisonCount
    difference=caseMAE(:,comparisonIndex)-caseMAE(:,end);
    MeanCaseMAEDifference(comparisonIndex)=mean(difference);
    [RawP(comparisonIndex),~,stats]=signrank(caseMAE(:,comparisonIndex),caseMAE(:,end));
    SignedRankStatistic(comparisonIndex)=stats.signedrank;
    bootstrapMean=nan(10000,1);
    for bootstrapIndex=1:10000
        sampled=randi(numberOfCases,numberOfCases,1);
        bootstrapMean(bootstrapIndex)=mean(difference(sampled));
    end
    interval=prctile(bootstrapMean,[2.5 97.5]);
    BootstrapCILower(comparisonIndex)=interval(1);
    BootstrapCIUpper(comparisonIndex)=interval(2);
end
HolmAdjustedP=holmAdjust(RawP);
statistics=table(ComparatorFraction,IndependentCaseN,MeanCaseMAEDifference, ...
    BootstrapCILower,BootstrapCIUpper,RawP,HolmAdjustedP,SignedRankStatistic);

%% Case and prediction source data
caseTable=array2table(caseMAE,'VariableNames', ...
    compose('StableWindow_%dPercent',round(100*fractions)));
caseTable=addvars(caseTable,caseList,'Before',1,'NewVariableNames','CaseID');
predictionTable=table(T.OriginalIndex,T.CaseID,T.RunID,y, ...
    'VariableNames',{'OriginalIndex','CaseID','RunID','ActualVB'});
for fractionIndex=1:numberOfFractions
    predictionTable.(sprintf('Prediction_%dPercent',round(100*fractions(fractionIndex))))= ...
        predictionMatrix(:,fractionIndex);
end
hyperparameters=table(repmat(caseList,numberOfFractions,1), ...
    repelem(fractions(:),numberOfCases),selectedLambdaAll(:), ...
    selectedLambdaQuality(:),selectedGate(:), ...
    'VariableNames',{'CaseID','StableWindowFraction','LambdaAll', ...
    'LambdaQuality','GateThreshold'});

writetable(overall,fullfile(resultDir,'early_window_overall.csv'));
writetable(statistics,fullfile(resultDir,'early_window_statistics.csv'));
writetable(caseTable,fullfile(resultDir,'early_window_case_metrics.csv'));
writetable(predictionTable,fullfile(resultDir,'early_window_predictions.csv'));
writetable(hyperparameters,fullfile(resultDir,'early_window_hyperparameters.csv'));

%% Submission-grade figure
figure('Color','w','Position',[60 60 1500 960]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile;
plot(100*fractions,MAE,'-o','LineWidth',2,'MarkerSize',7); hold on;
plot(100*fractions,RMSE,'-s','LineWidth',2,'MarkerSize',7);
xlabel('Available fraction of stable region (%)'); ylabel('Error (VB)');
legend({'MAE','RMSE'},'Location','best'); grid on;
title('(a) Prediction error versus observation window');

nexttile;
plot(100*fractions,R2,'-o','LineWidth',2,'MarkerSize',7,'Color',[0.20 0.55 0.35]);
xlabel('Available fraction of stable region (%)'); ylabel('R^2'); grid on;
title('(b) Explained variance');

nexttile;
for caseIndex=1:numberOfCases
    plot(100*fractions,caseMAE(caseIndex,:),'-o','Color',[0.75 0.75 0.75]); hold on;
end
plot(100*fractions,mean(caseMAE,1),'-o','LineWidth',3, ...
    'Color',[0.90 0.30 0.08],'MarkerFaceColor',[0.90 0.30 0.08]);
xlabel('Available fraction of stable region (%)'); ylabel('Case-level MAE'); grid on;
title('(c) Independent-case trajectories');

nexttile;
yyaxis left;
bar(100*fractions,MeanObservationEndpointSeconds,'FaceColor',[0.25 0.55 0.78]);
ylabel('Mean observation endpoint (s)');
yyaxis right;
plot(100*fractions,TotalOnlineComputeMillisecondsPerRecord,'-o', ...
    'LineWidth',2,'Color',[0.90 0.30 0.08]);
ylabel('Compute time per record (ms)');
xlabel('Available fraction of stable region (%)'); grid on;
title('(d) Observation and computational latency');
sgtitle('Early Tool-Wear Prediction under Strict Leave-One-Case-Out Validation', ...
    'FontWeight','bold');
set(gcf,'ToolBar','none');
axesHandles=findall(gcf,'Type','axes');
for axisIndex=1:numel(axesHandles)
    if isprop(axesHandles(axisIndex),'Toolbar') && ~isempty(axesHandles(axisIndex).Toolbar)
        axesHandles(axisIndex).Toolbar.Visible='off';
    end
end
drawnow;
exportgraphics(gcf,fullfile(resultDir,'early_window_latency_validation.png'),'Resolution',300);
exportgraphics(gcf,fullfile(resultDir,'early_window_latency_validation.pdf'),'ContentType','vector');

save(fullfile(resultDir,'early_window_latency_validation.mat'), ...
    'overall','statistics','caseTable','predictionTable','hyperparameters', ...
    'predictionMatrix','caseMAE','allFeatureMatrices','allQualityRatios');

fprintf('\n========== Early-window results ==========\n'); disp(overall);
fprintf('\n========== Paired case-level comparisons ==========\n'); disp(statistics);
fprintf('\nSaved to:\n%s\n',resultDir);

function [X,qualityRatio,endpointSeconds,observedPercent]=buildWindowFeatures( ...
        mill,originalIndex,signalNames,fraction,Fs)
    nRecords=numel(originalIndex); X=nan(nRecords,78);
    qualityRatio=nan(nRecords,1); endpointSeconds=nan(nRecords,1);
    observedPercent=nan(nRecords,1);
    for recordIndex=1:nRecords
        record=mill(originalIndex(recordIndex));
        [features,ratio,endSample,referenceLength]= ...
            calculateRecordFeatures(record,signalNames,fraction,Fs);
        X(recordIndex,:)=features; qualityRatio(recordIndex)=ratio;
        endpointSeconds(recordIndex)=endSample/Fs;
        observedPercent(recordIndex)=endSample/referenceLength;
    end
end

function [features,qualityRatio,endSample,referenceLength]= ...
        calculateRecordFeatures(record,signalNames,fraction,Fs)
    features=nan(1,78); qualityRatio=nan;
    for signalIndex=1:numel(signalNames)
        x=double(record.(signalNames(signalIndex))(:));
        n=numel(x); stableStart=floor(0.25*n)+1; stableEnd=floor(0.75*n);
        stableLength=stableEnd-stableStart+1;
        endSample=min(stableEnd,stableStart+max(1,floor(fraction*stableLength))-1);
        selected=x(stableStart:endSample);
        block=(signalIndex-1)*13+(1:13);
        features(block)=calculateSignalFeatures(selected,Fs);
        if signalNames(signalIndex)=="smcDC"
            qualityRatio=mean(selected>=9.99);
        end
        if signalIndex==1, referenceLength=n; end
    end
end

function features=calculateSignalFeatures(x,Fs)
    x=double(x(:));
    featureMean=mean(x); featureRMS=rms(x); featureStd=std(x);
    featurePeakToPeak=max(x)-min(x); xDetrended=detrend(x);
    featureSkewness=skewness(xDetrended); featureKurtosis=kurtosis(xDetrended);
    detrendedRMS=rms(xDetrended);
    featureCrestFactor=max(abs(xDetrended))/max(detrendedRMS,eps);
    windowLength=min(512,numel(xDetrended)); analysisWindow=hamming(windowLength);
    overlapLength=floor(windowLength/2); nfft=max(1024,2^nextpow2(windowLength));
    [Pxx,f]=pwelch(xDetrended,analysisWindow,overlapLength,nfft,Fs);
    totalPower=sum(Pxx);
    if totalPower<=eps, normalizedPower=ones(size(Pxx))/numel(Pxx);
    else, normalizedPower=Pxx/totalPower; end
    featureMeanFrequency=sum(f.*normalizedPower);
    cumulativePower=cumsum(normalizedPower);
    medianIndex=find(cumulativePower>=0.5,1,'first');
    featureMedianFrequency=f(medianIndex);
    featureSpectralEntropy=-sum(normalizedPower.*log2(normalizedPower+eps))/ ...
        log2(numel(normalizedPower));
    lowMask=f>=0 & f<20; midMask=f>=20 & f<60; highMask=f>=60 & f<=Fs/2;
    featureLowBandRatio=sum(normalizedPower(lowMask));
    featureMidBandRatio=sum(normalizedPower(midMask));
    featureHighBandRatio=sum(normalizedPower(highMask));
    features=[featureMean,featureRMS,featureStd,featurePeakToPeak, ...
        featureSkewness,featureKurtosis,featureCrestFactor, ...
        featureMeanFrequency,featureMedianFrequency,featureSpectralEntropy, ...
        featureLowBandRatio,featureMidBandRatio,featureHighBandRatio];
end

function model=fitRidgeRuntime(X,y,lambda)
    mu=mean(X,1); sigma=std(X,0,1); keep=isfinite(sigma) & sigma>1e-12;
    model.mu=mu(keep); model.sigma=sigma(keep); model.keep=keep;
    Xz=(X(:,keep)-model.mu)./model.sigma; model.yMean=mean(y);
    model.beta=(Xz'*Xz+lambda*eye(sum(keep)))\(Xz'*(y-model.yMean));
end

function yhat=predictRidgeRuntime(model,X)
    Xz=(X(:,model.keep)-model.mu)./model.sigma;
    yhat=model.yMean+Xz*model.beta;
end

function yhat=deploymentPrediction(allModel,qualityModel,XAll,XQuality,ratio,threshold)
    yhat=predictRidgeRuntime(allModel,XAll);
    qualityPrediction=predictRidgeRuntime(qualityModel,XQuality);
    switchMask=ratio>=threshold; yhat(switchMask)=qualityPrediction(switchMask);
end

function adjusted=holmAdjust(p)
    p=p(:); m=numel(p); [sortedP,order]=sort(p);
    adjustedSorted=nan(m,1); running=0;
    for i=1:m
        running=max(running,(m-i+1)*sortedP(i));
        adjustedSorted(i)=min(1,running);
    end
    adjusted=nan(m,1); adjusted(order)=adjustedSorted;
end
