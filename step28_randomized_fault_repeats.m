clear; clc; close all;
rng(20260718,'twister');

paperRoot=fileparts(mfilename('fullpath'));
previousDir=fullfile(paperRoot,'external_validation_results');
resultDir=fullfile(paperRoot,'additional_validation_results','reviewer_improvements');
if ~exist(resultDir,'dir'), mkdir(resultDir); end

load(fullfile(previousDir,'external_validation_complete_results.mat'), ...
    'nuaa','phm','lambdaGrid');

faultTypes=["Saturation","Dropout","Bias","Noise"];
repeatCount=30;
bootstrapCount=10000;

fprintf('========== Preparing locked clean-data policies ==========\n');
nuaaPrepared=prepareDataset(nuaa, ...
    fullfile(paperRoot,'nuaa_orthogonal_bundle_high_resolution.csv'), ...
    "NUAA",["force_z","vibration1","vibration2"], ...
    ["force_z","vibration_x","vibration_y"],lambdaGrid);
phmPrepared=prepareDataset(phm, ...
    fullfile(paperRoot,'phm2010_bundle_high_resolution.csv'), ...
    "PHM2010",["force_z","vibration_x","vibration_y"], ...
    ["force_z","vibration_x","vibration_y"],lambdaGrid);

scenarioRows=cell(repeatCount*2*3*numel(faultTypes),12);
experimentRows=cell(repeatCount*(9+3),9);
scenarioRow=0; experimentRow=0;
repeatSeed=2026071800+(1:repeatCount)';
repeatTimer=tic;

for repeatIndex=1:repeatCount
    rng(repeatSeed(repeatIndex),'twister');
    fprintf('Random fault repeat %d/%d\n',repeatIndex,repeatCount);
    [nuaaScenarios,nuaaExperiments]=runRandomRepeat( ...
        nuaaPrepared,faultTypes,repeatIndex,repeatSeed(repeatIndex));
    [phmScenarios,phmExperiments]=runRandomRepeat( ...
        phmPrepared,faultTypes,repeatIndex,repeatSeed(repeatIndex));
    currentScenarios=[nuaaScenarios;phmScenarios];
    currentExperiments=[nuaaExperiments;phmExperiments];
    scenarioRows(scenarioRow+(1:size(currentScenarios,1)),:)=currentScenarios;
    scenarioRow=scenarioRow+size(currentScenarios,1);
    experimentRows(experimentRow+(1:size(currentExperiments,1)),:)=currentExperiments;
    experimentRow=experimentRow+size(currentExperiments,1);
end
totalRuntimeSeconds=toc(repeatTimer);

scenarioTable=cell2table(scenarioRows,'VariableNames', ...
    {'Repeat','Seed','Dataset','TargetSensor','FaultType','RunLevelN', ...
    'FullMAE','AutoMAE','MAEImprovementPercent','CorrectDetectionRate', ...
    'MeanDurationFraction','MeanSeverity'});
scenarioTable.Dataset=string(scenarioTable.Dataset);
scenarioTable.TargetSensor=string(scenarioTable.TargetSensor);
scenarioTable.FaultType=string(scenarioTable.FaultType);

experimentTable=cell2table(experimentRows,'VariableNames', ...
    {'Repeat','Seed','Dataset','Experiment','RunLevelN','CleanMAE', ...
    'FullRelativeMAE','AutoRelativeMAE','RelativeImprovementPercent'});
experimentTable.Dataset=string(experimentTable.Dataset);
experimentTable.Experiment=string(experimentTable.Experiment);

%% Average the 30 technical fault randomizations before inference
datasets=["NUAA","PHM2010"];
meanExperimentRows=cell(12,9); row=0;
for datasetIndex=1:numel(datasets)
    ds=datasets(datasetIndex);
    experiments=unique(experimentTable.Experiment(experimentTable.Dataset==ds),'stable');
    for experimentIndex=1:numel(experiments)
        mask=experimentTable.Dataset==ds & ...
            experimentTable.Experiment==experiments(experimentIndex);
        row=row+1;
        meanExperimentRows(row,:)={ds,experiments(experimentIndex), ...
            experimentTable.RunLevelN(find(mask,1)),repeatCount, ...
            mean(experimentTable.CleanMAE(mask)), ...
            mean(experimentTable.FullRelativeMAE(mask)), ...
            mean(experimentTable.AutoRelativeMAE(mask)), ...
            std(experimentTable.AutoRelativeMAE(mask)), ...
            100*(mean(experimentTable.FullRelativeMAE(mask))- ...
            mean(experimentTable.AutoRelativeMAE(mask)))/ ...
            mean(experimentTable.FullRelativeMAE(mask))};
    end
end
meanExperimentTable=cell2table(meanExperimentRows,'VariableNames', ...
    {'Dataset','Experiment','RunLevelN','TechnicalRepeatN','CleanMAE', ...
    'FullRelativeMAE','AutoRelativeMAE','AutoRelativeMAESD', ...
    'RelativeImprovementPercent'});
meanExperimentTable.Dataset=string(meanExperimentTable.Dataset);
meanExperimentTable.Experiment=string(meanExperimentTable.Experiment);

%% Independent experiment/cutter inference with Holm correction
statisticsRows=cell(3,13);
statDatasets=["NUAA","PHM2010","Combined"];
rawP=nan(3,1);
for statIndex=1:3
    if statDatasets(statIndex)=="Combined"
        mask=true(height(meanExperimentTable),1);
    else
        mask=meanExperimentTable.Dataset==statDatasets(statIndex);
    end
    fullValue=meanExperimentTable.FullRelativeMAE(mask);
    autoValue=meanExperimentTable.AutoRelativeMAE(mask);
    difference=autoValue-fullValue;
    [rawP(statIndex),~,signedStats]=signrank(autoValue,fullValue);
    bootstrapMean=nan(bootstrapCount,1);
    n=numel(difference);
    for bootstrapIndex=1:bootstrapCount
        sampled=randi(n,n,1);
        bootstrapMean(bootstrapIndex)=mean(difference(sampled));
    end
    ci=prctile(bootstrapMean,[2.5 97.5]);
    statisticsRows(statIndex,:)={statDatasets(statIndex),n,repeatCount, ...
        mean(fullValue),mean(autoValue),mean(difference), ...
        100*(mean(fullValue)-mean(autoValue))/mean(fullValue), ...
        sum(difference<0),sum(difference>0),rawP(statIndex), ...
        signedStats.signedrank,ci(1),ci(2)};
end
HolmAdjustedP=holmAdjust(rawP);
statisticsTable=cell2table(statisticsRows,'VariableNames', ...
    {'Dataset','IndependentUnitN','TechnicalRepeatN','FullRelativeMAE', ...
    'AutoRelativeMAE','MeanPairedDifference','RelativeImprovementPercent', ...
    'UnitsImproved','UnitsWorsened','RawP','SignedRankStatistic', ...
    'BootstrapCILower','BootstrapCIUpper'});
statisticsTable.Dataset=string(statisticsTable.Dataset);
statisticsTable.HolmAdjustedP=HolmAdjustedP;
statisticsTable.SignificantAfterHolm=HolmAdjustedP<0.05;

%% Repeat-level variability is descriptive only
repeatRows=cell(repeatCount,8);
for repeatIndex=1:repeatCount
    mask=experimentTable.Repeat==repeatIndex;
    fullValue=mean(experimentTable.FullRelativeMAE(mask));
    autoValue=mean(experimentTable.AutoRelativeMAE(mask));
    repeatRows(repeatIndex,:)={repeatIndex,repeatSeed(repeatIndex), ...
        fullValue,autoValue,100*(fullValue-autoValue)/fullValue, ...
        min(experimentTable.RelativeImprovementPercent(mask)), ...
        max(experimentTable.RelativeImprovementPercent(mask)),sum(mask)};
end
repeatTable=cell2table(repeatRows,'VariableNames', ...
    {'Repeat','Seed','FullRelativeMAE','AutoRelativeMAE', ...
    'RelativeImprovementPercent','MinimumUnitImprovementPercent', ...
    'MaximumUnitImprovementPercent','IndependentUnitRows'});

%% Detection summary over randomizations
detectionRows=cell(2*3*numel(faultTypes),9); row=0;
for datasetIndex=1:numel(datasets)
    ds=datasets(datasetIndex);
    sensors=unique(scenarioTable.TargetSensor(scenarioTable.Dataset==ds),'stable');
    for sensorIndex=1:numel(sensors)
        for faultIndex=1:numel(faultTypes)
            mask=scenarioTable.Dataset==ds & ...
                scenarioTable.TargetSensor==sensors(sensorIndex) & ...
                scenarioTable.FaultType==faultTypes(faultIndex);
            row=row+1;
            detectionRows(row,:)={ds,sensors(sensorIndex),faultTypes(faultIndex), ...
                repeatCount,mean(scenarioTable.CorrectDetectionRate(mask)), ...
                std(scenarioTable.CorrectDetectionRate(mask)), ...
                mean(scenarioTable.MAEImprovementPercent(mask)), ...
                mean(scenarioTable.MeanDurationFraction(mask)), ...
                mean(scenarioTable.MeanSeverity(mask))};
        end
    end
end
detectionTable=cell2table(detectionRows,'VariableNames', ...
    {'Dataset','TargetSensor','FaultType','TechnicalRepeatN', ...
    'MeanCorrectDetectionRate','SDCorrectDetectionRate', ...
    'MeanMAEImprovementPercent','MeanDurationFraction','MeanSeverity'});
detectionTable.Dataset=string(detectionTable.Dataset);
detectionTable.TargetSensor=string(detectionTable.TargetSensor);
detectionTable.FaultType=string(detectionTable.FaultType);

runtimeTable=table(repeatCount,totalRuntimeSeconds, ...
    totalRuntimeSeconds/repeatCount,height(scenarioTable),height(experimentTable), ...
    'VariableNames',{'TechnicalRepeatN','TotalRuntimeSeconds', ...
    'SecondsPerRepeat','ScenarioRows','ExperimentRepeatRows'});

writetable(scenarioTable,fullfile(resultDir,'random_fault_scenarios.csv'));
writetable(experimentTable,fullfile(resultDir,'random_fault_experiment_repeats.csv'));
writetable(meanExperimentTable,fullfile(resultDir,'random_fault_experiment_means.csv'));
writetable(statisticsTable,fullfile(resultDir,'random_fault_statistics.csv'));
writetable(repeatTable,fullfile(resultDir,'random_fault_repeat_variability.csv'));
writetable(detectionTable,fullfile(resultDir,'random_fault_detection_summary.csv'));
writetable(runtimeTable,fullfile(resultDir,'random_fault_runtime.csv'));

%% Figure
figure('Color','w','Position',[60 60 1500 980]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile;
plot(repeatTable.Repeat,repeatTable.RelativeImprovementPercent,'-o', ...
    'Color',[0.25 0.55 0.78],'MarkerFaceColor',[0.25 0.55 0.78]);
yline(mean(repeatTable.RelativeImprovementPercent),'--','Mean','LineWidth',1.5);
xlabel('Randomization repeat'); ylabel('Relative MAE improvement (%)'); grid on;
title('(a) Robustness across 30 randomized fault realizations');

nexttile;
x=(1:height(meanExperimentTable))';
for i=1:height(meanExperimentTable)
    plot([1 2],[meanExperimentTable.FullRelativeMAE(i), ...
        meanExperimentTable.AutoRelativeMAE(i)],'-o','Color',[0.65 0.65 0.65]); hold on;
end
plot([1 2],[mean(meanExperimentTable.FullRelativeMAE), ...
    mean(meanExperimentTable.AutoRelativeMAE)],'-o','LineWidth',3, ...
    'Color',[0.90 0.30 0.08],'MarkerFaceColor',[0.90 0.30 0.08]);
xlim([0.7 2.3]); set(gca,'XTick',[1 2],'XTickLabel',{'Full sensors','Automatic gate'});
ylabel('Corrupted / clean MAE'); grid on;
title('(b) Independent experiment/cutter means');

nexttile;
faultDetection=nan(numel(faultTypes),numel(datasets));
for datasetIndex=1:numel(datasets)
    for faultIndex=1:numel(faultTypes)
        mask=detectionTable.Dataset==datasets(datasetIndex) & ...
            detectionTable.FaultType==faultTypes(faultIndex);
        faultDetection(faultIndex,datasetIndex)= ...
            mean(detectionTable.MeanCorrectDetectionRate(mask));
    end
end
bar(100*faultDetection);
set(gca,'XTickLabel',faultTypes); ylabel('Correct fault-channel detection (%)');
legend(datasets,'Location','best'); grid on;
title('(c) Detection by randomized fault type');

nexttile;
meanDifference=statisticsTable.MeanPairedDifference;
lowerError=meanDifference-statisticsTable.BootstrapCILower;
upperError=statisticsTable.BootstrapCIUpper-meanDifference;
errorbar(1:3,meanDifference,lowerError,upperError,'o','LineWidth',2, ...
    'MarkerFaceColor',[0.25 0.55 0.78]); hold on; yline(0,'--');
set(gca,'XTick',1:3,'XTickLabel',statisticsTable.Dataset);
ylabel('Automatic gate - full sensors (relative MAE)'); grid on;
title('(d) Experiment-cluster bootstrap 95% CI');

sgtitle('Randomized Multi-Sensor Fault Robustness (30 Technical Repeats)', ...
    'FontWeight','bold');
set(gcf,'ToolBar','none'); axesHandles=findall(gcf,'Type','axes');
for axisIndex=1:numel(axesHandles)
    if isprop(axesHandles(axisIndex),'Toolbar') && ~isempty(axesHandles(axisIndex).Toolbar)
        axesHandles(axisIndex).Toolbar.Visible='off';
    end
end
drawnow;
exportgraphics(gcf,fullfile(resultDir,'randomized_fault_robustness.png'),'Resolution',300);
exportgraphics(gcf,fullfile(resultDir,'randomized_fault_robustness.pdf'),'ContentType','vector');

save(fullfile(resultDir,'randomized_fault_robustness.mat'), ...
    'scenarioTable','experimentTable','meanExperimentTable','statisticsTable', ...
    'repeatTable','detectionTable','runtimeTable','faultTypes','repeatSeed','-v7.3');

fprintf('\n========== Randomized fault statistics ==========\n'); disp(statisticsTable);
fprintf('\n========== Detection summary ==========\n'); disp(detectionTable);
fprintf('\nSaved to:\n%s\n',resultDir);

function P=prepareDataset(D,filePath,datasetName,rawSensorNames,displayNames,lambdaGrid)
    T=readtable(filePath,'VariableNamingRule','preserve');
    groupId=makeRawGroupId(T,datasetName);
    n=numel(D.y); nSensors=numel(rawSensorNames);
    rawSignals=cell(n,nSensors); featureBlocks=cell(1,nSensors);
    cleanQuality=nan(n,nSensors,4);
    for sensorIndex=1:nSensors
        position=find(D.sensorNames==rawSensorNames(sensorIndex),1);
        assert(~isempty(position),'Missing monitored sensor %s.',rawSensorNames(sensorIndex));
        featureBlocks{sensorIndex}=(position-1)*9+(1:9);
    end
    for recordIndex=1:n
        rows=groupId==D.runTable.SourceGroup(recordIndex);
        assert(any(rows),'No raw samples for %s run %d.',datasetName,recordIndex);
        for sensorIndex=1:nSensors
            x=double(T.(char(rawSensorNames(sensorIndex)))(rows));
            x=x(isfinite(x)); rawSignals{recordIndex,sensorIndex}=x;
            cleanQuality(recordIndex,sensorIndex,:)=rawQualityDescriptors(x);
        end
    end
    policies=fitLockedPolicies(D,cleanQuality,featureBlocks,lambdaGrid);
    P.D=D; P.name=datasetName; P.rawSensorNames=rawSensorNames;
    P.displayNames=displayNames; P.rawSignals=rawSignals;
    P.featureBlocks=featureBlocks; P.cleanQuality=cleanQuality;
    P.policies=policies;
end

function policies=fitLockedPolicies(D,cleanQuality,featureBlocks,lambdaGrid)
    cases=unique(D.experiment,'stable'); nSensors=numel(featureBlocks);
    Xfull=[D.processX,D.sensorX,D.timeX];
    policies=repmat(struct, numel(cases),1);
    for caseIndex=1:numel(cases)
        isTrain=D.experiment~=cases(caseIndex);
        policies(caseIndex).case=cases(caseIndex);
        lambda=groupedTuneLambda(Xfull(isTrain,:),D.y(isTrain), ...
            D.experiment(isTrain),lambdaGrid);
        policies(caseIndex).fullModel=fitRidge(Xfull(isTrain,:),D.y(isTrain),lambda);
        policies(caseIndex).keepMasks=cell(1,nSensors);
        policies(caseIndex).dropModels=cell(1,nSensors);
        for sensorIndex=1:nSensors
            keep=true(1,size(D.sensorX,2)); keep(featureBlocks{sensorIndex})=false;
            Xdrop=[D.processX,D.sensorX(:,keep),D.timeX];
            lambdaDrop=groupedTuneLambda(Xdrop(isTrain,:),D.y(isTrain), ...
                D.experiment(isTrain),lambdaGrid);
            policies(caseIndex).keepMasks{sensorIndex}=keep;
            policies(caseIndex).dropModels{sensorIndex}= ...
                fitRidge(Xdrop(isTrain,:),D.y(isTrain),lambdaDrop);
        end
        policies(caseIndex).detector=fitQualityDetector( ...
            D.sensorX(isTrain,:),cleanQuality(isTrain,:,:),featureBlocks);
    end
end

function [scenarioRows,experimentRows]=runRandomRepeat(P,faultTypes,repeatIndex,seed)
    D=P.D; n=numel(D.y); nSensors=numel(P.displayNames); nFaults=numel(faultTypes);
    cases=unique(D.experiment,'stable');
    scenarioRows=cell(nSensors*nFaults,12);
    caseFullSum=zeros(numel(cases),1); caseAutoSum=zeros(numel(cases),1);
    caseCount=zeros(numel(cases),1);
    row=0;
    for targetSensor=1:nSensors
        for faultIndex=1:nFaults
            Xvariant=D.sensorX; quality=P.cleanQuality;
            durations=nan(n,1); severities=nan(n,1);
            for recordIndex=1:n
                [corrupted,durations(recordIndex),severities(recordIndex)]= ...
                    injectRandomFault(P.rawSignals{recordIndex,targetSensor}, ...
                    faultTypes(faultIndex));
                Xvariant(recordIndex,P.featureBlocks{targetSensor})= ...
                    signalFeatures(corrupted);
                quality(recordIndex,targetSensor,:)=rawQualityDescriptors(corrupted);
            end
            [fullPrediction,autoPrediction,flags]=predictLocked(P,Xvariant,quality);
            fullAbsolute=abs(fullPrediction-D.y); autoAbsolute=abs(autoPrediction-D.y);
            row=row+1;
            scenarioRows(row,:)={repeatIndex,seed,P.name,P.displayNames(targetSensor), ...
                faultTypes(faultIndex),n,mean(fullAbsolute),mean(autoAbsolute), ...
                100*(mean(fullAbsolute)-mean(autoAbsolute))/mean(fullAbsolute), ...
                mean(flags==targetSensor),mean(durations),mean(severities)};
            for caseIndex=1:numel(cases)
                mask=D.experiment==cases(caseIndex);
                caseFullSum(caseIndex)=caseFullSum(caseIndex)+sum(fullAbsolute(mask));
                caseAutoSum(caseIndex)=caseAutoSum(caseIndex)+sum(autoAbsolute(mask));
                caseCount(caseIndex)=caseCount(caseIndex)+sum(mask);
            end
        end
    end
    experimentRows=cell(numel(cases),9);
    for caseIndex=1:numel(cases)
        mask=D.experiment==cases(caseIndex);
        policy=P.policies(caseIndex);
        cleanX=[D.processX(mask,:),D.sensorX(mask,:),D.timeX(mask,:)];
        cleanPrediction=predictRidge(policy.fullModel,cleanX);
        cleanMAE=mean(abs(cleanPrediction-D.y(mask)));
        fullRelative=(caseFullSum(caseIndex)/caseCount(caseIndex))/cleanMAE;
        autoRelative=(caseAutoSum(caseIndex)/caseCount(caseIndex))/cleanMAE;
        experimentRows(caseIndex,:)={repeatIndex,seed,P.name,cases(caseIndex), ...
            sum(mask),cleanMAE,fullRelative,autoRelative, ...
            100*(fullRelative-autoRelative)/fullRelative};
    end
end

function [fullPrediction,autoPrediction,flagsAll]=predictLocked(P,Xvariant,quality)
    D=P.D; cases=unique(D.experiment,'stable'); n=numel(D.y);
    nSensors=numel(P.featureBlocks); fullPrediction=nan(n,1);
    autoPrediction=nan(n,1); flagsAll=zeros(n,1);
    for caseIndex=1:numel(cases)
        mask=D.experiment==cases(caseIndex); policy=P.policies(caseIndex);
        Xfull=[D.processX(mask,:),Xvariant(mask,:),D.timeX(mask,:)];
        fullPred=predictRidge(policy.fullModel,Xfull);
        dropPred=nan(sum(mask),nSensors);
        for routeSensor=1:nSensors
            keep=policy.keepMasks{routeSensor};
            Xdrop=[D.processX(mask,:),Xvariant(mask,keep),D.timeX(mask,:)];
            dropPred(:,routeSensor)=predictRidge(policy.dropModels{routeSensor},Xdrop);
        end
        flags=detectQualityFaults(Xvariant(mask,:),quality(mask,:,:), ...
            policy.detector,P.featureBlocks);
        autoPred=routePredictions(fullPred,dropPred,flags);
        fullPrediction(mask)=fullPred; autoPrediction(mask)=autoPred; flagsAll(mask)=flags;
    end
end

function [xc,durationFraction,severity]=injectRandomFault(x,faultType)
    xc=double(x(:)); n=numel(xc); scale=std(xc,0,'omitnan');
    if ~isfinite(scale) || scale<1e-10
        scale=max(1e-6,0.01*max(abs(xc),[],'omitnan'));
    end
    if faultType=="Saturation" || faultType=="Dropout"
        durationFraction=0.10+0.60*rand; severity=durationFraction;
    elseif faultType=="Bias"
        durationFraction=0.20+0.80*rand; severity=0.50+2.50*rand;
    else
        durationFraction=0.20+0.80*rand; severity=0.25+1.75*rand;
    end
    blockLength=min(n,max(1,round(durationFraction*n)));
    startIndex=randi(max(1,n-blockLength+1)); idx=startIndex:(startIndex+blockLength-1);
    switch faultType
        case "Saturation"
            xc(idx)=max(xc,[],'omitnan');
        case "Dropout"
            xc(idx)=0;
        case "Bias"
            direction=2*(rand>0.5)-1; xc(idx)=xc(idx)+direction*severity*scale;
        case "Noise"
            xc(idx)=xc(idx)+severity*scale*randn(blockLength,1);
    end
end

function groupId=makeRawGroupId(T,datasetName)
    tag=string(T.experiment_tag); timestamp=double(T.timestamp);
    if datasetName=="NUAA"
        [groupId,~,~]=findgroups(tag,double(T.experiment_csv_n));
    else
        groupId=zeros(height(T),1); nextGroup=0; tags=unique(tag,'stable');
        for tagIndex=1:numel(tags)
            rows=find(tag==tags(tagIndex)); [sortedTimestamp,order]=sort(timestamp(rows));
            rows=rows(order); increments=diff(sortedTimestamp);
            cadence=median(increments(increments>0),'omitnan');
            localRun=cumsum([true;increments<0.1*cadence]);
            groupId(rows)=nextGroup+localRun; nextGroup=nextGroup+max(localRun);
        end
    end
end

function q=rawQualityDescriptors(x)
    x=double(x(:)); x=x(isfinite(x));
    if isempty(x), q=[1 1 1 0]; return; end
    tolerance=max(1e-12,1e-8*max(1,max(abs(x))));
    rail=max(mean(abs(x-min(x))<=tolerance),mean(abs(x-max(x))<=tolerance));
    zero=mean(abs(x)<=tolerance);
    if numel(x)>1
        dx=diff(x); flat=mean(abs(dx)<=tolerance); diffRms=sqrt(mean(dx.^2));
    else
        flat=1; diffRms=0;
    end
    q=[rail,zero,flat,diffRms/max(std(x,0),1e-10)]; q(~isfinite(q))=0;
end

function f=signalFeatures(x)
    x=double(x(:)); x=x(isfinite(x));
    if isempty(x), f=zeros(1,9); return; end
    mu=mean(x); rootMeanSquare=sqrt(mean(x.^2));
    if numel(x)>1, sigma=std(x,0); else, sigma=0; end
    peakToPeak=max(x)-min(x); med=median(x); madValue=median(abs(x-med));
    if sigma>max(eps(abs(mu)),1e-12)
        z=(x-mu)./sigma; skewValue=mean(z.^3); kurtValue=mean(z.^4);
    else
        skewValue=0; kurtValue=0;
    end
    crestFactor=max(abs(x))/max(rootMeanSquare,eps);
    f=[mu,rootMeanSquare,sigma,peakToPeak,skewValue,kurtValue,med,madValue,crestFactor];
    f(~isfinite(f))=0;
end

function detector=fitQualityDetector(Xsensor,cleanQuality,featureBlocks)
    nSensors=numel(featureBlocks); selectedStats=[1 2 3 4 7 8 9];
    detector.center=cell(1,nSensors); detector.scale=cell(1,nSensors);
    detector.threshold=zeros(1,nSensors);
    for sensorIndex=1:nSensors
        rawQ=squeeze(cleanQuality(:,sensorIndex,:));
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)),rawQ];
        center=median(Q,1,'omitnan'); scale=1.4826*median(abs(Q-center),1,'omitnan');
        fallback=std(Q,0,1,'omitnan'); bad=~isfinite(scale)|scale<1e-9;
        scale(bad)=fallback(bad); scale(~isfinite(scale)|scale<1e-9)=1e-9;
        score=max(abs((Q-center)./scale),[],2);
        detector.center{sensorIndex}=center; detector.scale{sensorIndex}=scale;
        detector.threshold(sensorIndex)=max(4,prctile(score,99));
    end
end

function flags=detectQualityFaults(Xsensor,quality,detector,featureBlocks)
    n=size(Xsensor,1); nSensors=numel(featureBlocks); selectedStats=[1 2 3 4 7 8 9];
    normalizedScore=zeros(n,nSensors);
    for sensorIndex=1:nSensors
        rawQ=squeeze(quality(:,sensorIndex,:));
        if n==1, rawQ=reshape(rawQ,1,[]); end
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)),rawQ];
        score=max(abs((Q-detector.center{sensorIndex})./detector.scale{sensorIndex}),[],2);
        normalizedScore(:,sensorIndex)=score./detector.threshold(sensorIndex);
    end
    [bestScore,flags]=max(normalizedScore,[],2); flags(bestScore<=1)=0;
end

function routed=routePredictions(fullPrediction,dropPrediction,flags)
    routed=fullPrediction;
    for sensorIndex=1:size(dropPrediction,2)
        mask=flags==sensorIndex; routed(mask)=dropPrediction(mask,sensorIndex);
    end
end

function lambda=groupedTuneLambda(X,y,groups,lambdaGrid)
    cases=unique(groups,'stable'); scores=nan(numel(lambdaGrid),1);
    for lambdaIndex=1:numel(lambdaGrid)
        foldMAE=nan(numel(cases),1);
        for caseIndex=1:numel(cases)
            isValidation=groups==cases(caseIndex); isTrain=~isValidation;
            model=fitRidge(X(isTrain,:),y(isTrain),lambdaGrid(lambdaIndex));
            foldMAE(caseIndex)=mean(abs(predictRidge(model,X(isValidation,:))-y(isValidation)));
        end
        scores(lambdaIndex)=mean(foldMAE);
    end
    [~,best]=min(scores); lambda=lambdaGrid(best);
end

function model=fitRidge(X,y,lambda)
    X=double(X); y=double(y(:)); muX=mean(X,1,'omitnan'); muX(~isfinite(muX))=0;
    for columnIndex=1:size(X,2)
        bad=~isfinite(X(:,columnIndex)); X(bad,columnIndex)=muX(columnIndex);
    end
    sigmaX=std(X,0,1); sigmaX(~isfinite(sigmaX)|sigmaX<1e-12)=1;
    Xz=(X-muX)./sigmaX; muY=mean(y); yc=y-muY;
    beta=(Xz'*Xz+lambda*eye(size(Xz,2)))\(Xz'*yc);
    model.muX=muX; model.sigmaX=sigmaX; model.muY=muY; model.beta=beta;
end

function yhat=predictRidge(model,X)
    X=double(X);
    for columnIndex=1:size(X,2)
        bad=~isfinite(X(:,columnIndex)); X(bad,columnIndex)=model.muX(columnIndex);
    end
    yhat=model.muY+((X-model.muX)./model.sigmaX)*model.beta;
end

function adjusted=holmAdjust(p)
    p=p(:); m=numel(p); [sortedP,order]=sort(p); adjustedSorted=nan(m,1); running=0;
    for i=1:m
        running=max(running,(m-i+1)*sortedP(i)); adjustedSorted(i)=min(1,running);
    end
    adjusted=nan(m,1); adjusted(order)=adjustedSorted;
end
