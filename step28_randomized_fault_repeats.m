clear; clc; close all;
trainingSeedBase=str2double(getenv('SQTR_DROPOUT_SEED_BASE'));
if ~isfinite(trainingSeedBase), trainingSeedBase=20260718; end
rng(trainingSeedBase,'twister');

paperRoot=fileparts(mfilename('fullpath'));
dataRoot=string(getenv('SQTR_RAW_DATA_DIR'));
if strlength(dataRoot)==0, dataRoot=string(paperRoot); end
previousDir=fullfile(paperRoot,'external_validation_results');
resultDir=string(getenv('SQTR_RANDOM_FAULT_RESULT_DIR'));
if strlength(resultDir)==0
    resultDir=fullfile(paperRoot,'additional_validation_results','reviewer_improvements');
end
if ~exist(resultDir,'dir'), mkdir(resultDir); end

load(fullfile(previousDir,'external_validation_complete_results.mat'), ...
    'nuaa','phm','lambdaGrid');

faultSet=lower(string(getenv('SQTR_FAULT_SET')));
if faultSet=="structured"
    faultTypes=["SoftClipping","IntermittentDropout", ...
        "GradualDrift","HeteroscedasticNoise"];
elseif faultSet=="extended"
    faultTypes=["Saturation","Dropout","Bias","Noise", ...
        "SoftClipping","IntermittentDropout", ...
        "GradualDrift","HeteroscedasticNoise"];
else
    faultTypes=["Saturation","Dropout","Bias","Noise"];
end
repeatCount=str2double(getenv('SQTR_RANDOM_FAULT_REPEAT_COUNT'));
if ~isfinite(repeatCount) || repeatCount<1, repeatCount=30; end
repeatCount=round(repeatCount);
bootstrapCount=10000;
dropoutRates=[0.25 0.50 1.00];
detectorFloor=str2double(getenv('SQTR_DETECTOR_FLOOR'));
if ~isfinite(detectorFloor), detectorFloor=4; end
detectorQuantile=str2double(getenv('SQTR_DETECTOR_QUANTILE'));
if ~isfinite(detectorQuantile), detectorQuantile=99; end
exportRecordAudit=str2double(getenv('SQTR_EXPORT_RECORD_AUDIT'));
if ~isfinite(exportRecordAudit), exportRecordAudit=1; end
exportRecordAudit=logical(exportRecordAudit);
faultCardinality=str2double(getenv('SQTR_FAULT_CARDINALITY'));
if ~isfinite(faultCardinality), faultCardinality=1; end
faultCardinality=round(faultCardinality);
assert(ismember(faultCardinality,[1 2]), ...
    'SQTR_FAULT_CARDINALITY must be 1 or 2.');

fprintf('========== Preparing locked clean-data policies ==========\n');
nuaaPrepared=prepareDataset(nuaa, ...
    fullfile(dataRoot,'nuaa_orthogonal_bundle_high_resolution.csv'), ...
    "NUAA",["force_z","vibration1","vibration2"], ...
    ["force_z","vibration_x","vibration_y"],lambdaGrid,dropoutRates, ...
    trainingSeedBase,detectorFloor,detectorQuantile,faultCardinality);
phmPrepared=prepareDataset(phm, ...
    fullfile(dataRoot,'phm2010_bundle_high_resolution.csv'), ...
    "PHM2010",["force_z","vibration_x","vibration_y"], ...
    ["force_z","vibration_x","vibration_y"],lambdaGrid,dropoutRates, ...
    trainingSeedBase,detectorFloor,detectorQuantile,faultCardinality);

if faultCardinality==2
    runConcurrentPairExperiment(nuaaPrepared,phmPrepared,faultTypes, ...
        repeatCount,bootstrapCount,resultDir,detectorFloor, ...
        detectorQuantile,trainingSeedBase,faultSet);
    return;
end

scenarioRows=cell(repeatCount*2*3*numel(faultTypes),16);
experimentRows=cell(repeatCount*(9+3),17);
if exportRecordAudit
    recordRows=cell(repeatCount*3*numel(faultTypes)* ...
        (numel(nuaaPrepared.D.y)+numel(phmPrepared.D.y)),25);
else
    recordRows=cell(0,25);
end
scenarioRow=0; experimentRow=0; recordRow=0;
repeatSeed=2026071800+(1:repeatCount)';
repeatTimer=tic;

for repeatIndex=1:repeatCount
    rng(repeatSeed(repeatIndex),'twister');
    fprintf('Random fault repeat %d/%d\n',repeatIndex,repeatCount);
    [nuaaScenarios,nuaaExperiments,nuaaRecords]=runRandomRepeat( ...
        nuaaPrepared,faultTypes,repeatIndex,repeatSeed(repeatIndex),exportRecordAudit);
    [phmScenarios,phmExperiments,phmRecords]=runRandomRepeat( ...
        phmPrepared,faultTypes,repeatIndex,repeatSeed(repeatIndex),exportRecordAudit);
    currentScenarios=[nuaaScenarios;phmScenarios];
    currentExperiments=[nuaaExperiments;phmExperiments];
    currentRecords=[nuaaRecords;phmRecords];
    scenarioRows(scenarioRow+(1:size(currentScenarios,1)),:)=currentScenarios;
    scenarioRow=scenarioRow+size(currentScenarios,1);
    experimentRows(experimentRow+(1:size(currentExperiments,1)),:)=currentExperiments;
    experimentRow=experimentRow+size(currentExperiments,1);
    recordRows(recordRow+(1:size(currentRecords,1)),:)=currentRecords;
    recordRow=recordRow+size(currentRecords,1);
end
totalRuntimeSeconds=toc(repeatTimer);

scenarioTable=cell2table(scenarioRows,'VariableNames', ...
    {'Repeat','Seed','Dataset','TargetSensor','FaultType','RunLevelN', ...
    'FullMAE','MedianReplacementMAE','MaskDropoutMAE','AutoMAE', ...
    'MedianImprovementPercent','MaskDropoutImprovementPercent', ...
    'AutoImprovementPercent','CorrectDetectionRate', ...
    'MeanDurationFraction','MeanSeverity'});
scenarioTable.Dataset=string(scenarioTable.Dataset);
scenarioTable.TargetSensor=string(scenarioTable.TargetSensor);
scenarioTable.FaultType=string(scenarioTable.FaultType);

experimentTable=cell2table(experimentRows,'VariableNames', ...
    {'Repeat','Seed','Dataset','Experiment','RunLevelN','CleanFullMAE', ...
    'CleanMedianReplacementMAE','CleanMaskDropoutMAE','CleanAutoMAE', ...
    'CleanTriggerRate', ...
    'FullRelativeMAE','MedianReplacementRelativeMAE','MaskDropoutRelativeMAE', ...
    'AutoRelativeMAE','MedianImprovementPercent', ...
    'MaskDropoutImprovementPercent','AutoImprovementPercent'});
experimentTable.Dataset=string(experimentTable.Dataset);
experimentTable.Experiment=string(experimentTable.Experiment);

if exportRecordAudit
    recordTable=cell2table(recordRows,'VariableNames', ...
        {'Repeat','Seed','Dataset','Experiment','RecordIndex','TargetSensor', ...
        'FaultType','DurationFraction','Severity','ActualWear','DetectedSensor', ...
        'DetectorOutcome','FullAbsoluteError','TargetMedianAbsoluteError', ...
        'TargetMaskAbsoluteError','TargetExclusionAbsoluteError', ...
        'DetectorGuidedMedianAbsoluteError','DetectorGuidedMaskAbsoluteError', ...
        'SQTRAbsoluteError','OracleMinimumAbsoluteError','OracleBestAction', ...
        'SQTRRegret','CorrectDetection','DetectorMiss','WrongChannelDetection'});
    recordTable.Dataset=string(recordTable.Dataset);
    recordTable.Experiment=string(recordTable.Experiment);
    recordTable.TargetSensor=string(recordTable.TargetSensor);
    recordTable.FaultType=string(recordTable.FaultType);
    recordTable.DetectorOutcome=string(recordTable.DetectorOutcome);
    recordTable.OracleBestAction=string(recordTable.OracleBestAction);
else
    recordTable=table();
end

%% Average the 30 technical fault randomizations before inference
datasets=["NUAA","PHM2010"];
meanExperimentRows=cell(12,19); row=0;
for datasetIndex=1:numel(datasets)
    ds=datasets(datasetIndex);
    experiments=unique(experimentTable.Experiment(experimentTable.Dataset==ds),'stable');
    for experimentIndex=1:numel(experiments)
        mask=experimentTable.Dataset==ds & ...
            experimentTable.Experiment==experiments(experimentIndex);
        row=row+1;
        meanExperimentRows(row,:)={ds,experiments(experimentIndex), ...
            experimentTable.RunLevelN(find(mask,1)),repeatCount, ...
            mean(experimentTable.CleanFullMAE(mask)), ...
            mean(experimentTable.CleanMedianReplacementMAE(mask)), ...
            mean(experimentTable.CleanMaskDropoutMAE(mask)), ...
            mean(experimentTable.CleanAutoMAE(mask)), ...
            mean(experimentTable.CleanTriggerRate(mask)), ...
            mean(experimentTable.FullRelativeMAE(mask)), ...
            mean(experimentTable.MedianReplacementRelativeMAE(mask)), ...
            mean(experimentTable.MaskDropoutRelativeMAE(mask)), ...
            mean(experimentTable.AutoRelativeMAE(mask)), ...
            std(experimentTable.MedianReplacementRelativeMAE(mask)), ...
            std(experimentTable.MaskDropoutRelativeMAE(mask)), ...
            std(experimentTable.AutoRelativeMAE(mask)), ...
            100*(mean(experimentTable.FullRelativeMAE(mask))- ...
            mean(experimentTable.MedianReplacementRelativeMAE(mask)))/ ...
            mean(experimentTable.FullRelativeMAE(mask)), ...
            100*(mean(experimentTable.FullRelativeMAE(mask))- ...
            mean(experimentTable.MaskDropoutRelativeMAE(mask)))/ ...
            mean(experimentTable.FullRelativeMAE(mask)), ...
            100*(mean(experimentTable.FullRelativeMAE(mask))- ...
            mean(experimentTable.AutoRelativeMAE(mask)))/ ...
            mean(experimentTable.FullRelativeMAE(mask))};
    end
end
meanExperimentTable=cell2table(meanExperimentRows,'VariableNames', ...
    {'Dataset','Experiment','RunLevelN','TechnicalRepeatN','CleanFullMAE', ...
    'CleanMedianReplacementMAE','CleanMaskDropoutMAE','CleanAutoMAE', ...
    'CleanTriggerRate', ...
    'FullRelativeMAE','MedianReplacementRelativeMAE','MaskDropoutRelativeMAE', ...
    'AutoRelativeMAE','MedianReplacementRelativeMAESD','MaskDropoutRelativeMAESD', ...
    'AutoRelativeMAESD','MedianImprovementPercent','MaskDropoutImprovementPercent', ...
    'AutoImprovementPercent'});
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

%% Deployable-baseline comparison family: comparator minus SQTR
comparisonNames=["Detector-guided median replacement";"Mask-aware sensor-dropout ridge"];
comparisonValues=[meanExperimentTable.MedianReplacementRelativeMAE, ...
    meanExperimentTable.MaskDropoutRelativeMAE];
autoValue=meanExperimentTable.AutoRelativeMAE;
deployableRows=cell(numel(comparisonNames),12); deployableRawP=nan(numel(comparisonNames),1);
for comparisonIndex=1:numel(comparisonNames)
    comparatorValue=comparisonValues(:,comparisonIndex);
    difference=comparatorValue-autoValue;
    [deployableRawP(comparisonIndex),~,signedStats]=signrank(comparatorValue,autoValue);
    bootstrapMean=nan(bootstrapCount,1); n=numel(difference);
    for bootstrapIndex=1:bootstrapCount
        sampled=randi(n,n,1);
        bootstrapMean(bootstrapIndex)=mean(difference(sampled));
    end
    ci=prctile(bootstrapMean,[2.5 97.5]);
    deployableRows(comparisonIndex,:)={comparisonNames(comparisonIndex),n,repeatCount, ...
        mean(comparatorValue),mean(autoValue),mean(difference),ci(1),ci(2), ...
        100*mean(difference)/mean(comparatorValue),sum(difference>0), ...
        sum(difference<0),signedStats.signedrank};
end
deployableStatisticsTable=cell2table(deployableRows,'VariableNames', ...
    {'Comparator','IndependentUnitN','TechnicalRepeatN','ComparatorRelativeMAE', ...
    'SQTRRelativeMAE','MeanDifferenceComparatorMinusSQTR','BootstrapCILower', ...
    'BootstrapCIUpper','RelativeReductionPercent','UnitsFavoringSQTR', ...
    'UnitsFavoringComparator','SignedRankStatistic'});
deployableStatisticsTable.Comparator=string(deployableStatisticsTable.Comparator);
deployableStatisticsTable.RawP=deployableRawP;
deployableStatisticsTable.HolmAdjustedP=holmAdjust(deployableRawP);
deployableStatisticsTable.SignificantAfterHolm=deployableStatisticsTable.HolmAdjustedP<0.05;

selectedPolicyRows=cell(12,5); selectedRow=0;
preparedSets={nuaaPrepared,phmPrepared};
for preparedIndex=1:numel(preparedSets)
    prepared=preparedSets{preparedIndex};
    for policyIndex=1:numel(prepared.policies)
        selectedRow=selectedRow+1;
        selectedPolicyRows(selectedRow,:)={prepared.name,prepared.policies(policyIndex).case, ...
            prepared.policies(policyIndex).dropoutRate, ...
            prepared.policies(policyIndex).dropoutLambda, ...
            prepared.policies(policyIndex).dropoutSeed};
    end
end
selectedPolicyTable=cell2table(selectedPolicyRows,'VariableNames', ...
    {'Dataset','HeldOutUnit','SelectedDropoutRate','SelectedLambda','TrainingSeed'});
selectedPolicyTable.Dataset=string(selectedPolicyTable.Dataset);
selectedPolicyTable.HeldOutUnit=string(selectedPolicyTable.HeldOutUnit);

%% Repeat-level variability is descriptive only
repeatRows=cell(repeatCount,8);
for repeatIndex=1:repeatCount
    mask=experimentTable.Repeat==repeatIndex;
    fullValue=mean(experimentTable.FullRelativeMAE(mask));
    autoValue=mean(experimentTable.AutoRelativeMAE(mask));
    repeatRows(repeatIndex,:)={repeatIndex,repeatSeed(repeatIndex), ...
        fullValue,autoValue,100*(fullValue-autoValue)/fullValue, ...
        min(experimentTable.AutoImprovementPercent(mask)), ...
        max(experimentTable.AutoImprovementPercent(mask)),sum(mask)};
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
                mean(scenarioTable.AutoImprovementPercent(mask)), ...
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
configurationTable=table(detectorFloor,detectorQuantile,repeatCount, ...
    exportRecordAudit,faultSet,trainingSeedBase, ...
    'VariableNames',{'DetectorFloor','DetectorQuantile','TechnicalRepeatN', ...
    'ExportRecordAudit','FaultSet','TrainingSeedBase'});

writetable(scenarioTable,fullfile(resultDir,'random_fault_scenarios.csv'));
writetable(experimentTable,fullfile(resultDir,'random_fault_experiment_repeats.csv'));
writetable(meanExperimentTable,fullfile(resultDir,'random_fault_experiment_means.csv'));
writetable(statisticsTable,fullfile(resultDir,'random_fault_statistics.csv'));
writetable(repeatTable,fullfile(resultDir,'random_fault_repeat_variability.csv'));
writetable(detectionTable,fullfile(resultDir,'random_fault_detection_summary.csv'));
writetable(runtimeTable,fullfile(resultDir,'random_fault_runtime.csv'));
writetable(deployableStatisticsTable,fullfile(resultDir,'deployable_baseline_statistics.csv'));
writetable(selectedPolicyTable,fullfile(resultDir,'mask_dropout_selected_hyperparameters.csv'));
writetable(configurationTable,fullfile(resultDir,'random_fault_configuration.csv'));
actionAuditTable=table();
unitActionAuditTable=table();
if exportRecordAudit
    writetable(recordTable,fullfile(resultDir,'random_fault_record_action_audit.csv'));

    %% Failure decomposition and oracle action envelope
    actionNames=["Full";"TargetMedian";"TargetMask";"TargetExclusion"];
    auditRows=cell(2*3*numel(faultTypes),20); auditRow=0;
    for datasetIndex=1:numel(datasets)
        ds=datasets(datasetIndex);
        sensors=unique(recordTable.TargetSensor(recordTable.Dataset==ds),'stable');
        for sensorIndex=1:numel(sensors)
            for faultIndex=1:numel(faultTypes)
                mask=recordTable.Dataset==ds & ...
                    recordTable.TargetSensor==sensors(sensorIndex) & ...
                    recordTable.FaultType==faultTypes(faultIndex);
                oracleFractions=zeros(1,numel(actionNames));
                for actionIndex=1:numel(actionNames)
                    oracleFractions(actionIndex)=mean( ...
                        recordTable.OracleBestAction(mask)==actionNames(actionIndex));
                end
                auditRow=auditRow+1;
                auditRows(auditRow,:)={ds,sensors(sensorIndex),faultTypes(faultIndex), ...
                    repeatCount,sum(mask),mean(recordTable.CorrectDetection(mask)), ...
                    mean(recordTable.DetectorMiss(mask)), ...
                    mean(recordTable.WrongChannelDetection(mask)), ...
                    mean(recordTable.FullAbsoluteError(mask)), ...
                    mean(recordTable.TargetMedianAbsoluteError(mask)), ...
                    mean(recordTable.TargetMaskAbsoluteError(mask)), ...
                    mean(recordTable.TargetExclusionAbsoluteError(mask)), ...
                    mean(recordTable.SQTRAbsoluteError(mask)), ...
                    mean(recordTable.OracleMinimumAbsoluteError(mask)), ...
                    mean(recordTable.SQTRRegret(mask)), ...
                    oracleFractions(1),oracleFractions(2),oracleFractions(3), ...
                    oracleFractions(4), ...
                    mean(recordTable.SQTRRegret(mask & recordTable.CorrectDetection))};
            end
        end
    end
    actionAuditTable=cell2table(auditRows,'VariableNames', ...
        {'Dataset','TargetSensor','FaultType','TechnicalRepeatN','RecordScenarioRows', ...
        'CorrectDetectionRate','MissRate','WrongChannelRate','FullMAE', ...
        'TargetMedianMAE','TargetMaskMAE','TargetExclusionMAE','SQTRMAE', ...
        'OracleEnvelopeMAE','MeanSQTRRegret','OracleFullFraction', ...
        'OracleMedianFraction','OracleMaskFraction','OracleExclusionFraction', ...
        'MeanRegretWhenDetectionCorrect'});
    actionAuditTable.Dataset=string(actionAuditTable.Dataset);
    actionAuditTable.TargetSensor=string(actionAuditTable.TargetSensor);
    actionAuditTable.FaultType=string(actionAuditTable.FaultType);
    writetable(actionAuditTable,fullfile(resultDir,'random_fault_action_regret_summary.csv'));

    unitList=unique(recordTable(:,{'Dataset','Experiment'}),'rows','stable');
    unitAuditRows=cell(height(unitList),14);
    for unitIndex=1:height(unitList)
        mask=recordTable.Dataset==unitList.Dataset(unitIndex) & ...
            recordTable.Experiment==unitList.Experiment(unitIndex);
        unitAuditRows(unitIndex,:)={unitList.Dataset(unitIndex), ...
            unitList.Experiment(unitIndex),repeatCount,sum(mask), ...
            mean(recordTable.FullAbsoluteError(mask)), ...
            mean(recordTable.TargetMedianAbsoluteError(mask)), ...
            mean(recordTable.TargetMaskAbsoluteError(mask)), ...
            mean(recordTable.TargetExclusionAbsoluteError(mask)), ...
            mean(recordTable.SQTRAbsoluteError(mask)), ...
            mean(recordTable.OracleMinimumAbsoluteError(mask)), ...
            mean(recordTable.SQTRRegret(mask)), ...
            mean(recordTable.CorrectDetection(mask)), ...
            mean(recordTable.DetectorMiss(mask)), ...
            mean(recordTable.WrongChannelDetection(mask))};
    end
    unitActionAuditTable=cell2table(unitAuditRows,'VariableNames', ...
        {'Dataset','Experiment','TechnicalRepeatN','RecordScenarioRows', ...
        'FullMAE','TargetMedianMAE','TargetMaskMAE','TargetExclusionMAE', ...
        'SQTRMAE','OracleEnvelopeMAE','MeanSQTRRegret','CorrectDetectionRate', ...
        'MissRate','WrongChannelRate'});
    unitActionAuditTable.Dataset=string(unitActionAuditTable.Dataset);
    unitActionAuditTable.Experiment=string(unitActionAuditTable.Experiment);
    writetable(unitActionAuditTable,fullfile(resultDir,'random_fault_unit_action_audit.csv'));
end

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
    'repeatTable','detectionTable','runtimeTable','deployableStatisticsTable', ...
    'selectedPolicyTable','recordTable','actionAuditTable','unitActionAuditTable', ...
    'configurationTable','faultTypes','repeatSeed','dropoutRates', ...
    'detectorFloor','detectorQuantile','exportRecordAudit','-v7.3');

fprintf('\n========== Randomized fault statistics ==========\n'); disp(statisticsTable);
fprintf('\n========== Detection summary ==========\n'); disp(detectionTable);
fprintf('\n========== Deployable-baseline comparisons ==========\n'); disp(deployableStatisticsTable);
fprintf('\nSaved to:\n%s\n',resultDir);

function P=prepareDataset(D,filePath,datasetName,rawSensorNames,displayNames, ...
        lambdaGrid,dropoutRates,baseSeed,detectorFloor,detectorQuantile, ...
        faultCardinality)
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
    policies=fitLockedPolicies(D,cleanQuality,featureBlocks,lambdaGrid, ...
        dropoutRates,baseSeed,detectorFloor,detectorQuantile,faultCardinality);
    P.D=D; P.name=datasetName; P.rawSensorNames=rawSensorNames;
    P.displayNames=displayNames; P.rawSignals=rawSignals;
    P.featureBlocks=featureBlocks; P.cleanQuality=cleanQuality;
    P.policies=policies;
end

function policies=fitLockedPolicies(D,cleanQuality,featureBlocks,lambdaGrid, ...
        dropoutRates,baseSeed,detectorFloor,detectorQuantile,faultCardinality)
    cases=unique(D.experiment,'stable'); nSensors=numel(featureBlocks);
    Xfull=[D.processX,D.sensorX,D.timeX];
    policies=repmat(struct, numel(cases),1);
    for caseIndex=1:numel(cases)
        isTrain=D.experiment~=cases(caseIndex);
        policies(caseIndex).case=cases(caseIndex);
        lambda=groupedTuneLambda(Xfull(isTrain,:),D.y(isTrain), ...
            D.experiment(isTrain),lambdaGrid);
        policies(caseIndex).fullModel=fitRidge(Xfull(isTrain,:),D.y(isTrain),lambda);
        policies(caseIndex).sensorMedians=median(D.sensorX(isTrain,:),1,'omitnan');
        dropoutSeed=baseSeed+caseIndex;
        [dropoutRate,dropoutLambda]=groupedTuneMaskDropout( ...
            Xfull(isTrain,:),D.y(isTrain),D.experiment(isTrain), ...
            featureBlocks,size(D.processX,2),lambdaGrid,dropoutRates,dropoutSeed);
        [XdropoutTrain,yDropoutTrain]=augmentMaskDropoutTraining( ...
            Xfull(isTrain,:),D.y(isTrain),featureBlocks,size(D.processX,2), ...
            policies(caseIndex).sensorMedians,dropoutRate,dropoutSeed);
        policies(caseIndex).dropoutRate=dropoutRate;
        policies(caseIndex).dropoutLambda=dropoutLambda;
        policies(caseIndex).dropoutSeed=dropoutSeed;
        policies(caseIndex).dropoutModel=fitRidge(XdropoutTrain,yDropoutTrain,dropoutLambda);
        if faultCardinality==2
            [multiDropoutRate,multiDropoutLambda]=groupedTuneMultiMaskDropout( ...
                Xfull(isTrain,:),D.y(isTrain),D.experiment(isTrain), ...
                featureBlocks,size(D.processX,2),lambdaGrid,dropoutRates,dropoutSeed);
            [XmultiDropoutTrain,yMultiDropoutTrain]=augmentMultiMaskDropoutTraining( ...
                Xfull(isTrain,:),D.y(isTrain),featureBlocks,size(D.processX,2), ...
                policies(caseIndex).sensorMedians,multiDropoutRate,dropoutSeed);
            policies(caseIndex).multiDropoutRate=multiDropoutRate;
            policies(caseIndex).multiDropoutLambda=multiDropoutLambda;
            policies(caseIndex).multiDropoutModel=fitRidge( ...
                XmultiDropoutTrain,yMultiDropoutTrain,multiDropoutLambda);
        end
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
            D.sensorX(isTrain,:),cleanQuality(isTrain,:,:),featureBlocks, ...
            detectorFloor,detectorQuantile);
        if faultCardinality==2
            policies(caseIndex).subsetKeepMasks=cell(7,1);
            policies(caseIndex).subsetModels=cell(7,1);
            for subsetCode=1:7
                selectedSensors=find(bitget(subsetCode,1:nSensors));
                if numel(selectedSensors)==1
                    sensorIndex=selectedSensors(1);
                    policies(caseIndex).subsetKeepMasks{subsetCode}= ...
                        policies(caseIndex).keepMasks{sensorIndex};
                    policies(caseIndex).subsetModels{subsetCode}= ...
                        policies(caseIndex).dropModels{sensorIndex};
                else
                    keep=true(1,size(D.sensorX,2));
                    for sensorIndex=selectedSensors
                        keep(featureBlocks{sensorIndex})=false;
                    end
                    Xsubset=[D.processX,D.sensorX(:,keep),D.timeX];
                    lambdaSubset=groupedTuneLambda( ...
                        Xsubset(isTrain,:),D.y(isTrain), ...
                        D.experiment(isTrain),lambdaGrid);
                    policies(caseIndex).subsetKeepMasks{subsetCode}=keep;
                    policies(caseIndex).subsetModels{subsetCode}= ...
                        fitRidge(Xsubset(isTrain,:),D.y(isTrain),lambdaSubset);
                end
            end
        end
    end
end

function [scenarioRows,experimentRows,recordRows]=runRandomRepeat( ...
        P,faultTypes,repeatIndex,seed,exportRecordAudit)
    D=P.D; n=numel(D.y); nSensors=numel(P.displayNames); nFaults=numel(faultTypes);
    cases=unique(D.experiment,'stable');
    scenarioRows=cell(nSensors*nFaults,16);
    caseFullSum=zeros(numel(cases),1); caseMedianSum=zeros(numel(cases),1);
    caseMaskDropoutSum=zeros(numel(cases),1); caseAutoSum=zeros(numel(cases),1);
    caseCount=zeros(numel(cases),1);
    if exportRecordAudit
        recordRows=cell(n*nSensors*nFaults,25);
    else
        recordRows=cell(0,25);
    end
    row=0; recordRow=0;
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
            [fullPrediction,medianPrediction,maskDropoutPrediction,autoPrediction,flags]= ...
                predictLocked(P,Xvariant,quality);
            fullAbsolute=abs(fullPrediction-D.y);
            medianAbsolute=abs(medianPrediction-D.y);
            maskDropoutAbsolute=abs(maskDropoutPrediction-D.y);
            autoAbsolute=abs(autoPrediction-D.y);
            if exportRecordAudit
                [targetMedianPrediction,targetMaskPrediction,targetExclusionPrediction]= ...
                    predictKnownTarget(P,Xvariant,targetSensor);
                targetMedianAbsolute=abs(targetMedianPrediction-D.y);
                targetMaskAbsolute=abs(targetMaskPrediction-D.y);
                targetExclusionAbsolute=abs(targetExclusionPrediction-D.y);
                candidateErrors=[fullAbsolute,targetMedianAbsolute, ...
                    targetMaskAbsolute,targetExclusionAbsolute];
                [oracleMinimum,oracleIndex]=min(candidateErrors,[],2);
                oracleAction=["Full","TargetMedian","TargetMask","TargetExclusion"];
                detectorOutcome=repmat("Correct",n,1);
                detectorOutcome(flags==0)="Miss";
                detectorOutcome(flags>0 & flags~=targetSensor)="WrongChannel";
                for recordIndex=1:n
                    recordRow=recordRow+1;
                    recordRows(recordRow,:)={repeatIndex,seed,P.name, ...
                        D.experiment(recordIndex),recordIndex,P.displayNames(targetSensor), ...
                        faultTypes(faultIndex),durations(recordIndex), ...
                        severities(recordIndex),D.y(recordIndex),flags(recordIndex), ...
                        detectorOutcome(recordIndex),fullAbsolute(recordIndex), ...
                        targetMedianAbsolute(recordIndex),targetMaskAbsolute(recordIndex), ...
                        targetExclusionAbsolute(recordIndex),medianAbsolute(recordIndex), ...
                        maskDropoutAbsolute(recordIndex),autoAbsolute(recordIndex), ...
                        oracleMinimum(recordIndex),oracleAction(oracleIndex(recordIndex)), ...
                        autoAbsolute(recordIndex)-oracleMinimum(recordIndex), ...
                        flags(recordIndex)==targetSensor,flags(recordIndex)==0, ...
                        flags(recordIndex)>0 && flags(recordIndex)~=targetSensor};
                end
            end
            row=row+1;
            scenarioRows(row,:)={repeatIndex,seed,P.name,P.displayNames(targetSensor), ...
                faultTypes(faultIndex),n,mean(fullAbsolute),mean(medianAbsolute), ...
                mean(maskDropoutAbsolute),mean(autoAbsolute), ...
                100*(mean(fullAbsolute)-mean(medianAbsolute))/mean(fullAbsolute), ...
                100*(mean(fullAbsolute)-mean(maskDropoutAbsolute))/mean(fullAbsolute), ...
                100*(mean(fullAbsolute)-mean(autoAbsolute))/mean(fullAbsolute), ...
                mean(flags==targetSensor),mean(durations),mean(severities)};
            for caseIndex=1:numel(cases)
                mask=D.experiment==cases(caseIndex);
                caseFullSum(caseIndex)=caseFullSum(caseIndex)+sum(fullAbsolute(mask));
                caseMedianSum(caseIndex)=caseMedianSum(caseIndex)+sum(medianAbsolute(mask));
                caseMaskDropoutSum(caseIndex)=caseMaskDropoutSum(caseIndex)+sum(maskDropoutAbsolute(mask));
                caseAutoSum(caseIndex)=caseAutoSum(caseIndex)+sum(autoAbsolute(mask));
                caseCount(caseIndex)=caseCount(caseIndex)+sum(mask);
            end
        end
    end
    [cleanFullPrediction,cleanMedianPrediction,cleanMaskDropoutPrediction, ...
        cleanAutoPrediction,cleanFlags]=predictLocked(P,D.sensorX,P.cleanQuality);
    experimentRows=cell(numel(cases),17);
    for caseIndex=1:numel(cases)
        mask=D.experiment==cases(caseIndex);
        cleanFullMAE=mean(abs(cleanFullPrediction(mask)-D.y(mask)));
        cleanMedianMAE=mean(abs(cleanMedianPrediction(mask)-D.y(mask)));
        cleanMaskDropoutMAE=mean(abs(cleanMaskDropoutPrediction(mask)-D.y(mask)));
        cleanAutoMAE=mean(abs(cleanAutoPrediction(mask)-D.y(mask)));
        cleanTriggerRate=mean(cleanFlags(mask)>0);
        fullRelative=(caseFullSum(caseIndex)/caseCount(caseIndex))/cleanFullMAE;
        medianRelative=(caseMedianSum(caseIndex)/caseCount(caseIndex))/cleanFullMAE;
        maskDropoutRelative=(caseMaskDropoutSum(caseIndex)/caseCount(caseIndex))/cleanFullMAE;
        autoRelative=(caseAutoSum(caseIndex)/caseCount(caseIndex))/cleanFullMAE;
        experimentRows(caseIndex,:)={repeatIndex,seed,P.name,cases(caseIndex), ...
            sum(mask),cleanFullMAE,cleanMedianMAE,cleanMaskDropoutMAE,cleanAutoMAE, ...
            cleanTriggerRate,fullRelative,medianRelative,maskDropoutRelative,autoRelative, ...
            100*(fullRelative-medianRelative)/fullRelative, ...
            100*(fullRelative-maskDropoutRelative)/fullRelative, ...
            100*(fullRelative-autoRelative)/fullRelative};
    end
end

function [targetMedianPrediction,targetMaskPrediction,targetExclusionPrediction]= ...
        predictKnownTarget(P,Xvariant,targetSensor)
    D=P.D; cases=unique(D.experiment,'stable'); n=numel(D.y);
    targetMedianPrediction=nan(n,1); targetMaskPrediction=nan(n,1);
    targetExclusionPrediction=nan(n,1);
    for caseIndex=1:numel(cases)
        mask=D.experiment==cases(caseIndex); policy=P.policies(caseIndex);
        targetFlags=repmat(targetSensor,sum(mask),1);
        [XmedianSensor,maskMatrix]=applyDetectorReplacement( ...
            Xvariant(mask,:),targetFlags,P.featureBlocks,policy.sensorMedians);
        Xmedian=[D.processX(mask,:),XmedianSensor,D.timeX(mask,:)];
        targetMedianPrediction(mask)=predictRidge(policy.fullModel,Xmedian);
        targetMaskPrediction(mask)=predictRidge(policy.dropoutModel,[Xmedian,maskMatrix]);
        keep=policy.keepMasks{targetSensor};
        Xdrop=[D.processX(mask,:),Xvariant(mask,keep),D.timeX(mask,:)];
        targetExclusionPrediction(mask)=predictRidge( ...
            policy.dropModels{targetSensor},Xdrop);
    end
end

function [fullPrediction,medianPrediction,maskDropoutPrediction,autoPrediction,flagsAll]=predictLocked(P,Xvariant,quality)
    D=P.D; cases=unique(D.experiment,'stable'); n=numel(D.y);
    nSensors=numel(P.featureBlocks); fullPrediction=nan(n,1);
    medianPrediction=nan(n,1); maskDropoutPrediction=nan(n,1);
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
        [XmedianSensor,maskMatrix]=applyDetectorReplacement( ...
            Xvariant(mask,:),flags,P.featureBlocks,policy.sensorMedians);
        Xmedian=[D.processX(mask,:),XmedianSensor,D.timeX(mask,:)];
        medianPred=predictRidge(policy.fullModel,Xmedian);
        dropoutPred=predictRidge(policy.dropoutModel,[Xmedian,maskMatrix]);
        autoPred=routePredictions(fullPred,dropPred,flags);
        fullPrediction(mask)=fullPred; medianPrediction(mask)=medianPred;
        maskDropoutPrediction(mask)=dropoutPred; autoPrediction(mask)=autoPred;
        flagsAll(mask)=flags;
    end
end

function [Xreplaced,maskMatrix]=applyDetectorReplacement(Xsensor,flags,featureBlocks,sensorMedians)
    Xreplaced=Xsensor; n=size(Xsensor,1); nSensors=numel(featureBlocks);
    maskMatrix=zeros(n,nSensors);
    for sensorIndex=1:nSensors
        rows=flags==sensorIndex;
        if any(rows)
            Xreplaced(rows,featureBlocks{sensorIndex})= ...
                repmat(sensorMedians(featureBlocks{sensorIndex}),sum(rows),1);
            maskMatrix(rows,sensorIndex)=1;
        end
    end
end

function [bestRate,bestLambda]=groupedTuneMaskDropout(X,y,groups,featureBlocks,nProcess,lambdaGrid,dropoutRates,baseSeed)
    cases=unique(groups,'stable'); scores=nan(numel(dropoutRates),numel(lambdaGrid));
    for rateIndex=1:numel(dropoutRates)
        for lambdaIndex=1:numel(lambdaGrid)
            foldMAE=nan(numel(cases),1);
            for caseIndex=1:numel(cases)
                isValidation=groups==cases(caseIndex); isTrain=~isValidation;
                sensorColumns=cellfun(@(b)nProcess+b,featureBlocks,'UniformOutput',false);
                medians=median(X(isTrain,:),1,'omitnan');
                [Xaug,yaug]=augmentMaskDropoutTraining(X(isTrain,:),y(isTrain), ...
                    featureBlocks,nProcess,medians(nProcess+(1:max(cellfun(@max,featureBlocks)))), ...
                    dropoutRates(rateIndex),baseSeed+1000*caseIndex+100*rateIndex);
                model=fitRidge(Xaug,yaug,lambdaGrid(lambdaIndex));
                Xvalidation=X(isValidation,:); yvalidation=y(isValidation);
                stateErrors=abs(predictRidge(model,[Xvalidation,zeros(sum(isValidation),numel(featureBlocks))])-yvalidation);
                for sensorIndex=1:numel(featureBlocks)
                    Xstate=Xvalidation; columns=sensorColumns{sensorIndex};
                    Xstate(:,columns)=repmat(medians(columns),sum(isValidation),1);
                    maskState=zeros(sum(isValidation),numel(featureBlocks));
                    maskState(:,sensorIndex)=1;
                    stateErrors=[stateErrors;abs(predictRidge(model,[Xstate,maskState])-yvalidation)]; %#ok<AGROW>
                end
                foldMAE(caseIndex)=mean(stateErrors);
            end
            scores(rateIndex,lambdaIndex)=mean(foldMAE);
        end
    end
    [~,bestLinear]=min(scores,[],'all','linear');
    [bestRateIndex,bestLambdaIndex]=ind2sub(size(scores),bestLinear);
    bestRate=dropoutRates(bestRateIndex); bestLambda=lambdaGrid(bestLambdaIndex);
end

function [Xaug,yaug]=augmentMaskDropoutTraining(X,y,featureBlocks,nProcess,sensorMedians,dropoutRate,seed)
    n=size(X,1); nSensors=numel(featureBlocks); stream=RandStream('mt19937ar','Seed',seed);
    selected=find(rand(stream,n,1)<dropoutRate);
    if isempty(selected), selected=randi(stream,n,1,1); end
    selectedSensor=randi(stream,nSensors,numel(selected),1);
    clean=[X,zeros(n,nSensors)]; masked=X(selected,:); masks=zeros(numel(selected),nSensors);
    for rowIndex=1:numel(selected)
        sensorIndex=selectedSensor(rowIndex);
        columns=nProcess+featureBlocks{sensorIndex};
        masked(rowIndex,columns)=sensorMedians(featureBlocks{sensorIndex});
        masks(rowIndex,sensorIndex)=1;
    end
    Xaug=[clean;masked,masks]; yaug=[y(:);y(selected)];
end

function [bestRate,bestLambda]=groupedTuneMultiMaskDropout( ...
        X,y,groups,featureBlocks,nProcess,lambdaGrid,dropoutRates,baseSeed)
    cases=unique(groups,'stable'); nSensors=numel(featureBlocks);
    subsetCodes=1:(2^nSensors-1);
    scores=nan(numel(dropoutRates),numel(lambdaGrid));
    sensorColumns=cellfun(@(b)nProcess+b,featureBlocks,'UniformOutput',false);
    for rateIndex=1:numel(dropoutRates)
        for lambdaIndex=1:numel(lambdaGrid)
            foldMAE=nan(numel(cases),1);
            for caseIndex=1:numel(cases)
                isValidation=groups==cases(caseIndex); isTrain=~isValidation;
                medians=median(X(isTrain,:),1,'omitnan');
                sensorMedians=medians(nProcess+(1:max(cellfun(@max,featureBlocks))));
                [Xaug,yaug]=augmentMultiMaskDropoutTraining( ...
                    X(isTrain,:),y(isTrain),featureBlocks,nProcess,sensorMedians, ...
                    dropoutRates(rateIndex),baseSeed+1000*caseIndex+100*rateIndex);
                model=fitRidge(Xaug,yaug,lambdaGrid(lambdaIndex));
                Xvalidation=X(isValidation,:); yvalidation=y(isValidation);
                stateErrors=abs(predictRidge(model, ...
                    [Xvalidation,zeros(sum(isValidation),nSensors)])-yvalidation);
                for subsetCode=subsetCodes
                    Xstate=Xvalidation;
                    maskState=zeros(sum(isValidation),nSensors);
                    selectedSensors=find(bitget(subsetCode,1:nSensors));
                    for sensorIndex=selectedSensors
                        columns=sensorColumns{sensorIndex};
                        Xstate(:,columns)=repmat(medians(columns),sum(isValidation),1);
                        maskState(:,sensorIndex)=1;
                    end
                    stateErrors=[stateErrors;abs(predictRidge( ...
                        model,[Xstate,maskState])-yvalidation)]; %#ok<AGROW>
                end
                foldMAE(caseIndex)=mean(stateErrors);
            end
            scores(rateIndex,lambdaIndex)=mean(foldMAE);
        end
    end
    [~,bestLinear]=min(scores,[],'all','linear');
    [bestRateIndex,bestLambdaIndex]=ind2sub(size(scores),bestLinear);
    bestRate=dropoutRates(bestRateIndex); bestLambda=lambdaGrid(bestLambdaIndex);
end

function [Xaug,yaug]=augmentMultiMaskDropoutTraining( ...
        X,y,featureBlocks,nProcess,sensorMedians,dropoutRate,seed)
    n=size(X,1); nSensors=numel(featureBlocks);
    stream=RandStream('mt19937ar','Seed',seed);
    selected=find(rand(stream,n,1)<dropoutRate);
    if isempty(selected), selected=randi(stream,n,1,1); end
    subsetCodes=1:(2^nSensors-1);
    selectedCodes=subsetCodes(randi(stream,numel(subsetCodes),numel(selected),1));
    clean=[X,zeros(n,nSensors)]; masked=X(selected,:);
    masks=zeros(numel(selected),nSensors);
    for rowIndex=1:numel(selected)
        selectedSensors=find(bitget(selectedCodes(rowIndex),1:nSensors));
        for sensorIndex=selectedSensors
            columns=nProcess+featureBlocks{sensorIndex};
            masked(rowIndex,columns)=sensorMedians(featureBlocks{sensorIndex});
            masks(rowIndex,sensorIndex)=1;
        end
    end
    Xaug=[clean;masked,masks]; yaug=[y(:);y(selected)];
end

function [xc,durationFraction,severity]=injectRandomFault(x,faultType)
    xc=double(x(:)); n=numel(xc); scale=std(xc,0,'omitnan');
    if ~isfinite(scale) || scale<1e-10
        scale=max(1e-6,0.01*max(abs(xc),[],'omitnan'));
    end
    if faultType=="Saturation" || faultType=="Dropout"
        durationFraction=0.10+0.60*rand; severity=durationFraction;
    elseif faultType=="SoftClipping"
        durationFraction=0.20+0.60*rand; severity=0.50+2.50*rand;
    elseif faultType=="IntermittentDropout"
        durationFraction=0.10+0.40*rand; severity=durationFraction;
    elseif faultType=="GradualDrift"
        durationFraction=0.40+0.60*rand; severity=0.50+2.50*rand;
    elseif faultType=="HeteroscedasticNoise"
        durationFraction=0.40+0.60*rand; severity=0.50+1.50*rand;
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
        case "SoftClipping"
            localCenter=median(xc(idx),'omitnan');
            limit=scale/max(severity,1e-6);
            xc(idx)=localCenter+limit*tanh((xc(idx)-localCenter)/limit);
        case "IntermittentDropout"
            burstN=randi([3 8]);
            targetSampleN=max(burstN,round(durationFraction*n));
            burstLength=max(1,round(targetSampleN/burstN));
            affected=false(n,1);
            for burstIndex=1:burstN
                burstStart=randi(max(1,n-burstLength+1));
                affected(burstStart:(burstStart+burstLength-1))=true;
            end
            xc(affected)=0;
            durationFraction=mean(affected);
            severity=durationFraction;
        case "GradualDrift"
            direction=2*(rand>0.5)-1;
            ramp=linspace(0,1,blockLength)';
            xc(idx)=xc(idx)+direction*severity*scale*ramp;
        case "HeteroscedasticNoise"
            localScale=linspace(0.05,severity,blockLength)';
            xc(idx)=xc(idx)+scale*localScale.*randn(blockLength,1);
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

function detector=fitQualityDetector(Xsensor,cleanQuality,featureBlocks, ...
        detectorFloor,detectorQuantile)
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
        detector.threshold(sensorIndex)=max(detectorFloor, ...
            prctile(score,detectorQuantile));
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

function runConcurrentPairExperiment(nuaaPrepared,phmPrepared,faultTypes, ...
        repeatCount,bootstrapCount,resultDir,detectorFloor, ...
        detectorQuantile,trainingSeedBase,faultSet)
    pairList=nchoosek(1:3,2);
    repeatSeed=2026081500+(1:repeatCount)';
    scenarioRows=cell(repeatCount*2*size(pairList,1)*numel(faultTypes),20);
    experimentRows=cell(repeatCount*12,23);
    unitDetectorRows=cell(repeatCount*12*size(pairList,1)*numel(faultTypes),12);
    scenarioRow=0; experimentRow=0; unitDetectorRow=0;
    timer=tic;
    for repeatIndex=1:repeatCount
        rng(repeatSeed(repeatIndex),'twister');
        fprintf('Concurrent-pair repeat %d/%d\n',repeatIndex,repeatCount);
        [nuaaScenario,nuaaExperiment,nuaaUnitDetector]=runConcurrentPairRepeat( ...
            nuaaPrepared,faultTypes,pairList,repeatIndex,repeatSeed(repeatIndex));
        [phmScenario,phmExperiment,phmUnitDetector]=runConcurrentPairRepeat( ...
            phmPrepared,faultTypes,pairList,repeatIndex,repeatSeed(repeatIndex));
        currentScenario=[nuaaScenario;phmScenario];
        currentExperiment=[nuaaExperiment;phmExperiment];
        currentUnitDetector=[nuaaUnitDetector;phmUnitDetector];
        scenarioRows(scenarioRow+(1:size(currentScenario,1)),:)=currentScenario;
        scenarioRow=scenarioRow+size(currentScenario,1);
        experimentRows(experimentRow+(1:size(currentExperiment,1)),:)=currentExperiment;
        experimentRow=experimentRow+size(currentExperiment,1);
        unitDetectorRows(unitDetectorRow+(1:size(currentUnitDetector,1)),:)=currentUnitDetector;
        unitDetectorRow=unitDetectorRow+size(currentUnitDetector,1);
    end
    runtimeSeconds=toc(timer);

    scenarioTable=cell2table(scenarioRows,'VariableNames', ...
        {'Repeat','Seed','Dataset','TargetPair','FaultType','RunLevelN', ...
        'FullMAE','MedianReplacementMAE','SingleMaskTrainMAE','MultiMaskTrainMAE', ...
        'SingleSQTR_MAE','SubsetRoutingMAE','OraclePairExclusionMAE', ...
        'ExactSetDetectionRate','BothTargetDetectedRate', ...
        'OnlyOneTargetDetectedRate','NeitherTargetDetectedRate', ...
        'ExtraChannelTriggerRate','MeanDurationFraction','MeanSeverity'});
    scenarioTable.Dataset=string(scenarioTable.Dataset);
    scenarioTable.TargetPair=string(scenarioTable.TargetPair);
    scenarioTable.FaultType=string(scenarioTable.FaultType);

    experimentTable=cell2table(experimentRows,'VariableNames', ...
        {'Repeat','Seed','Dataset','Experiment','RunLevelN','CleanFullMAE', ...
        'CleanMedianMAE','CleanSingleMaskTrainMAE','CleanMultiMaskTrainMAE', ...
        'CleanSingleSQTR_MAE','CleanSubsetRoutingMAE','CleanAnyTriggerRate', ...
        'FullRelativeMAE','MedianRelativeMAE','SingleMaskTrainRelativeMAE', ...
        'MultiMaskTrainRelativeMAE','SingleSQTR_RelativeMAE', ...
        'SubsetRoutingRelativeMAE','OraclePairExclusionRelativeMAE', ...
        'SubsetImprovementPercent','SingleImprovementPercent', ...
        'SingleMaskImprovementPercent','MultiMaskImprovementPercent'});
    experimentTable.Dataset=string(experimentTable.Dataset);
    experimentTable.Experiment=string(experimentTable.Experiment);

    unitDetectorTable=cell2table(unitDetectorRows,'VariableNames', ...
        {'Repeat','Seed','Dataset','Experiment','TargetPair','FaultType', ...
        'RunLevelN','ExactSetDetectionRate','BothTargetDetectedRate', ...
        'OnlyOneTargetDetectedRate','NeitherTargetDetectedRate', ...
        'ExtraChannelTriggerRate'});
    unitDetectorTable.Dataset=string(unitDetectorTable.Dataset);
    unitDetectorTable.Experiment=string(unitDetectorTable.Experiment);
    unitDetectorTable.TargetPair=string(unitDetectorTable.TargetPair);
    unitDetectorTable.FaultType=string(unitDetectorTable.FaultType);

    datasets=["NUAA","PHM2010"];
    meanRows=cell(12,21); meanRow=0;
    for datasetIndex=1:numel(datasets)
        ds=datasets(datasetIndex);
        units=unique(experimentTable.Experiment(experimentTable.Dataset==ds),'stable');
        for unitIndex=1:numel(units)
            mask=experimentTable.Dataset==ds & experimentTable.Experiment==units(unitIndex);
            meanRow=meanRow+1;
            meanRows(meanRow,:)={ds,units(unitIndex), ...
                experimentTable.RunLevelN(find(mask,1)),repeatCount, ...
                mean(experimentTable.CleanAnyTriggerRate(mask)), ...
                mean(experimentTable.FullRelativeMAE(mask)), ...
                mean(experimentTable.MedianRelativeMAE(mask)), ...
                mean(experimentTable.SingleMaskTrainRelativeMAE(mask)), ...
                mean(experimentTable.MultiMaskTrainRelativeMAE(mask)), ...
                mean(experimentTable.SingleSQTR_RelativeMAE(mask)), ...
                mean(experimentTable.SubsetRoutingRelativeMAE(mask)), ...
                mean(experimentTable.OraclePairExclusionRelativeMAE(mask)), ...
                std(experimentTable.MedianRelativeMAE(mask)), ...
                std(experimentTable.SingleMaskTrainRelativeMAE(mask)), ...
                std(experimentTable.MultiMaskTrainRelativeMAE(mask)), ...
                std(experimentTable.SingleSQTR_RelativeMAE(mask)), ...
                std(experimentTable.SubsetRoutingRelativeMAE(mask)), ...
                mean(experimentTable.SubsetImprovementPercent(mask)), ...
                mean(experimentTable.SingleImprovementPercent(mask)), ...
                mean(experimentTable.SingleMaskImprovementPercent(mask)), ...
                mean(experimentTable.MultiMaskImprovementPercent(mask))};
        end
    end
    meanExperimentTable=cell2table(meanRows,'VariableNames', ...
        {'Dataset','Experiment','RunLevelN','TechnicalRepeatN', ...
        'CleanAnyTriggerRate','FullRelativeMAE','MedianRelativeMAE', ...
        'SingleMaskTrainRelativeMAE','MultiMaskTrainRelativeMAE', ...
        'SingleSQTR_RelativeMAE','SubsetRoutingRelativeMAE', ...
        'OraclePairExclusionRelativeMAE','MedianRelativeMAE_SD', ...
        'SingleMaskTrainRelativeMAE_SD','MultiMaskTrainRelativeMAE_SD', ...
        'SingleSQTR_RelativeMAE_SD','SubsetRoutingRelativeMAE_SD', ...
        'SubsetImprovementPercent','SingleImprovementPercent', ...
        'SingleMaskImprovementPercent','MultiMaskImprovementPercent'});
    meanExperimentTable.Dataset=string(meanExperimentTable.Dataset);
    meanExperimentTable.Experiment=string(meanExperimentTable.Experiment);

    responseNames=["Median replacement";"Single-mask-trained dropout"; ...
        "Multi-mask-trained dropout";"Single-channel SQTR";"Subset routing"];
    responseValues=[meanExperimentTable.MedianRelativeMAE, ...
        meanExperimentTable.SingleMaskTrainRelativeMAE, ...
        meanExperimentTable.MultiMaskTrainRelativeMAE, ...
        meanExperimentTable.SingleSQTR_RelativeMAE, ...
        meanExperimentTable.SubsetRoutingRelativeMAE];
    fixedValues=meanExperimentTable.FullRelativeMAE;
    comparisonRows=cell(5,12); rawP=nan(5,1);
    rng(2026081511,'twister');
    for responseIndex=1:5
        response=responseValues(:,responseIndex);
        difference=fixedValues-response; % positive favours response
        [rawP(responseIndex),~,signedStats]=signrank( ...
            fixedValues,response,'tail','both','method','exact');
        boot=nan(bootstrapCount,1); n=numel(difference);
        for b=1:bootstrapCount
            idx=randi(n,n,1); boot(b)=mean(difference(idx));
        end
        ci=prctile(boot,[2.5 97.5]);
        comparisonRows(responseIndex,:)={responseNames(responseIndex),n, ...
            repeatCount,mean(fixedValues),mean(response),mean(difference), ...
            ci(1),ci(2),sum(difference>0),sum(difference<0), ...
            sum(difference==0),signedStats.signedrank};
    end
    holmP=holmAdjust(rawP);
    strategyStatistics=cell2table(comparisonRows,'VariableNames', ...
        {'Response','IndependentUnitN','TechnicalRepeatN','FixedRelativeMAE', ...
        'ResponseRelativeMAE','FixedMinusResponse','BootstrapCI_Lower', ...
        'BootstrapCI_Upper','UnitsFavourResponse','UnitsFavourFixed', ...
        'TiedUnits','SignedRankStatistic'});
    strategyStatistics.RawP=rawP;
    strategyStatistics.HolmAdjustedP=holmP;
    strategyStatistics.SignificantAfterHolm=holmP<0.05;

    singleMinusSubset=meanExperimentTable.SingleSQTR_RelativeMAE- ...
        meanExperimentTable.SubsetRoutingRelativeMAE;
    [subsetP,~,subsetStats]=signrank( ...
        meanExperimentTable.SingleSQTR_RelativeMAE, ...
        meanExperimentTable.SubsetRoutingRelativeMAE, ...
        'tail','both','method','exact');
    boot=nan(bootstrapCount,1); n=numel(singleMinusSubset);
    for b=1:bootstrapCount
        idx=randi(n,n,1); boot(b)=mean(singleMinusSubset(idx));
    end
    ci=prctile(boot,[2.5 97.5]);
    subsetVsSingle=table(n,repeatCount, ...
        mean(meanExperimentTable.SingleSQTR_RelativeMAE), ...
        mean(meanExperimentTable.SubsetRoutingRelativeMAE), ...
        mean(singleMinusSubset),ci(1),ci(2), ...
        sum(singleMinusSubset>0),sum(singleMinusSubset<0), ...
        subsetStats.signedrank,subsetP, ...
        'VariableNames',{'IndependentUnitN','TechnicalRepeatN', ...
        'SingleSQTR_RelativeMAE','SubsetRoutingRelativeMAE', ...
        'SingleMinusSubset','BootstrapCI_Lower','BootstrapCI_Upper', ...
        'UnitsFavourSubset','UnitsFavourSingle','SignedRankStatistic','ExactP'});

    datasetRows=cell(2,12);
    for datasetIndex=1:2
        mask=meanExperimentTable.Dataset==datasets(datasetIndex);
        datasetRows(datasetIndex,:)={datasets(datasetIndex),sum(mask),repeatCount, ...
            mean(meanExperimentTable.FullRelativeMAE(mask)), ...
            mean(meanExperimentTable.MedianRelativeMAE(mask)), ...
            mean(meanExperimentTable.SingleMaskTrainRelativeMAE(mask)), ...
            mean(meanExperimentTable.MultiMaskTrainRelativeMAE(mask)), ...
            mean(meanExperimentTable.SingleSQTR_RelativeMAE(mask)), ...
            mean(meanExperimentTable.SubsetRoutingRelativeMAE(mask)), ...
            mean(meanExperimentTable.OraclePairExclusionRelativeMAE(mask)), ...
            mean(meanExperimentTable.CleanAnyTriggerRate(mask)), ...
            sum(meanExperimentTable.SubsetRoutingRelativeMAE(mask)< ...
                meanExperimentTable.FullRelativeMAE(mask))};
    end
    datasetSummary=cell2table(datasetRows,'VariableNames', ...
        {'Dataset','IndependentUnitN','TechnicalRepeatN','FullRelativeMAE', ...
        'MedianRelativeMAE','SingleMaskTrainRelativeMAE', ...
        'MultiMaskTrainRelativeMAE','SingleSQTR_RelativeMAE', ...
        'SubsetRoutingRelativeMAE','OraclePairExclusionRelativeMAE', ...
        'CleanAnyTriggerRate','UnitsFavourSubset'});

    configurationTable=table(detectorFloor,detectorQuantile,repeatCount, ...
        faultSet,trainingSeedBase,runtimeSeconds, ...
        'VariableNames',{'DetectorFloor','DetectorQuantile', ...
        'TechnicalRepeatN','FaultSet','TrainingSeedBase','RuntimeSeconds'});

    writetable(scenarioTable,fullfile(resultDir,'concurrent_pair_scenarios.csv'));
    writetable(experimentTable,fullfile(resultDir,'concurrent_pair_unit_repeats.csv'));
    writetable(unitDetectorTable,fullfile(resultDir,'concurrent_pair_unit_detector_states.csv'));
    writetable(meanExperimentTable,fullfile(resultDir,'concurrent_pair_unit_means.csv'));
    writetable(strategyStatistics,fullfile(resultDir,'concurrent_pair_strategy_statistics.csv'));
    writetable(subsetVsSingle,fullfile(resultDir,'concurrent_pair_subset_vs_single.csv'));
    writetable(datasetSummary,fullfile(resultDir,'concurrent_pair_dataset_summary.csv'));
    writetable(configurationTable,fullfile(resultDir,'concurrent_pair_configuration.csv'));
    save(fullfile(resultDir,'concurrent_pair_results.mat'), ...
        'scenarioTable','experimentTable','unitDetectorTable','meanExperimentTable', ...
        'strategyStatistics','subsetVsSingle','datasetSummary', ...
        'configurationTable','faultTypes','pairList','repeatSeed','-v7.3');

    fprintf('\n========== Concurrent-pair strategy statistics ==========\n');
    disp(strategyStatistics);
    fprintf('\n========== Subset routing versus single-channel SQTR ==========\n');
    disp(subsetVsSingle);
    fprintf('\nSaved concurrent-pair results to:\n%s\n',resultDir);
end

function [scenarioRows,experimentRows,unitDetectorRows]=runConcurrentPairRepeat( ...
        P,faultTypes,pairList,repeatIndex,seed)
    D=P.D; n=numel(D.y); nPairs=size(pairList,1); nFaults=numel(faultTypes);
    units=unique(D.experiment,'stable');
    scenarioRows=cell(nPairs*nFaults,20);
    fullSum=zeros(numel(units),1); medianSum=zeros(numel(units),1);
    singleMaskSum=zeros(numel(units),1); multiMaskSum=zeros(numel(units),1);
    singleSum=zeros(numel(units),1); subsetSum=zeros(numel(units),1);
    oracleSum=zeros(numel(units),1); unitCount=zeros(numel(units),1);
    unitDetectorRows=cell(numel(units)*nPairs*nFaults,12);
    row=0; detectorRow=0;
    for pairIndex=1:nPairs
        targetPair=pairList(pairIndex,:);
        targetMask=false(1,3); targetMask(targetPair)=true;
        pairLabel=strjoin(P.displayNames(targetPair),'+');
        for faultIndex=1:nFaults
            Xvariant=D.sensorX; quality=P.cleanQuality;
            durations=nan(n,2); severities=nan(n,2);
            for recordIndex=1:n
                for pairMember=1:2
                    sensorIndex=targetPair(pairMember);
                    [corrupted,durations(recordIndex,pairMember), ...
                        severities(recordIndex,pairMember)]=injectRandomFault( ...
                        P.rawSignals{recordIndex,sensorIndex},faultTypes(faultIndex));
                    Xvariant(recordIndex,P.featureBlocks{sensorIndex})= ...
                        signalFeatures(corrupted);
                    quality(recordIndex,sensorIndex,:)=rawQualityDescriptors(corrupted);
                end
            end
            [fullPrediction,medianPrediction,singleMaskPrediction, ...
                multiMaskPrediction,singlePrediction,subsetPrediction, ...
                oraclePrediction,flagMatrix]=predictConcurrentLocked( ...
                P,Xvariant,quality,targetPair);
            fullAbsolute=abs(fullPrediction-D.y);
            medianAbsolute=abs(medianPrediction-D.y);
            singleMaskAbsolute=abs(singleMaskPrediction-D.y);
            multiMaskAbsolute=abs(multiMaskPrediction-D.y);
            singleAbsolute=abs(singlePrediction-D.y);
            subsetAbsolute=abs(subsetPrediction-D.y);
            oracleAbsolute=abs(oraclePrediction-D.y);
            exactSet=all(flagMatrix==repmat(targetMask,n,1),2);
            targetDetectedCount=sum(flagMatrix(:,targetPair),2);
            bothTarget=targetDetectedCount==2;
            onlyOneTarget=targetDetectedCount==1;
            neitherTarget=targetDetectedCount==0;
            nonTarget=setdiff(1:3,targetPair);
            extraTrigger=flagMatrix(:,nonTarget);
            row=row+1;
            scenarioRows(row,:)={repeatIndex,seed,P.name,pairLabel, ...
                faultTypes(faultIndex),n,mean(fullAbsolute),mean(medianAbsolute), ...
                mean(singleMaskAbsolute),mean(multiMaskAbsolute), ...
                mean(singleAbsolute),mean(subsetAbsolute),mean(oracleAbsolute), ...
                mean(exactSet),mean(bothTarget),mean(onlyOneTarget), ...
                mean(neitherTarget),mean(extraTrigger), ...
                mean(durations,'all'),mean(severities,'all')};
            for unitIndex=1:numel(units)
                mask=D.experiment==units(unitIndex);
                detectorRow=detectorRow+1;
                unitDetectorRows(detectorRow,:)={repeatIndex,seed,P.name, ...
                    units(unitIndex),pairLabel,faultTypes(faultIndex),sum(mask), ...
                    mean(exactSet(mask)),mean(bothTarget(mask)), ...
                    mean(onlyOneTarget(mask)),mean(neitherTarget(mask)), ...
                    mean(extraTrigger(mask))};
                fullSum(unitIndex)=fullSum(unitIndex)+sum(fullAbsolute(mask));
                medianSum(unitIndex)=medianSum(unitIndex)+sum(medianAbsolute(mask));
                singleMaskSum(unitIndex)=singleMaskSum(unitIndex)+sum(singleMaskAbsolute(mask));
                multiMaskSum(unitIndex)=multiMaskSum(unitIndex)+sum(multiMaskAbsolute(mask));
                singleSum(unitIndex)=singleSum(unitIndex)+sum(singleAbsolute(mask));
                subsetSum(unitIndex)=subsetSum(unitIndex)+sum(subsetAbsolute(mask));
                oracleSum(unitIndex)=oracleSum(unitIndex)+sum(oracleAbsolute(mask));
                unitCount(unitIndex)=unitCount(unitIndex)+sum(mask);
            end
        end
    end

    [cleanFull,cleanMedian,cleanSingleMask,cleanMultiMask, ...
        cleanSingle,cleanSubset,~,cleanFlags]= ...
        predictConcurrentLocked(P,D.sensorX,P.cleanQuality,pairList(1,:));
    experimentRows=cell(numel(units),23);
    for unitIndex=1:numel(units)
        mask=D.experiment==units(unitIndex);
        cleanFullMAE=mean(abs(cleanFull(mask)-D.y(mask)));
        cleanMedianMAE=mean(abs(cleanMedian(mask)-D.y(mask)));
        cleanSingleMaskMAE=mean(abs(cleanSingleMask(mask)-D.y(mask)));
        cleanMultiMaskMAE=mean(abs(cleanMultiMask(mask)-D.y(mask)));
        cleanSingleMAE=mean(abs(cleanSingle(mask)-D.y(mask)));
        cleanSubsetMAE=mean(abs(cleanSubset(mask)-D.y(mask)));
        cleanTriggerRate=mean(any(cleanFlags(mask,:),2));
        fullRelative=(fullSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        medianRelative=(medianSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        singleMaskRelative=(singleMaskSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        multiMaskRelative=(multiMaskSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        singleRelative=(singleSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        subsetRelative=(subsetSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        oracleRelative=(oracleSum(unitIndex)/unitCount(unitIndex))/cleanFullMAE;
        experimentRows(unitIndex,:)={repeatIndex,seed,P.name,units(unitIndex), ...
            sum(mask),cleanFullMAE,cleanMedianMAE,cleanSingleMaskMAE, ...
            cleanMultiMaskMAE,cleanSingleMAE,cleanSubsetMAE, ...
            cleanTriggerRate,fullRelative,medianRelative, ...
            singleMaskRelative,multiMaskRelative,singleRelative, ...
            subsetRelative,oracleRelative, ...
            100*(fullRelative-subsetRelative)/fullRelative, ...
            100*(fullRelative-singleRelative)/fullRelative, ...
            100*(fullRelative-singleMaskRelative)/fullRelative, ...
            100*(fullRelative-multiMaskRelative)/fullRelative};
    end
end

function [fullPrediction,medianPrediction,singleMaskPrediction, ...
        multiMaskPrediction,singlePrediction,subsetPrediction, ...
        oraclePrediction,flagMatrixAll]=predictConcurrentLocked( ...
        P,Xvariant,quality,targetPair)
    D=P.D; units=unique(D.experiment,'stable'); n=numel(D.y);
    fullPrediction=nan(n,1); medianPrediction=nan(n,1);
    singleMaskPrediction=nan(n,1); multiMaskPrediction=nan(n,1);
    singlePrediction=nan(n,1); subsetPrediction=nan(n,1);
    oraclePrediction=nan(n,1); flagMatrixAll=false(n,3);
    targetCode=sum(2.^(targetPair-1));
    for unitIndex=1:numel(units)
        mask=D.experiment==units(unitIndex); policy=P.policies(unitIndex);
        Xfull=[D.processX(mask,:),Xvariant(mask,:),D.timeX(mask,:)];
        fullPred=predictRidge(policy.fullModel,Xfull);
        scores=qualityNormalizedScores(Xvariant(mask,:),quality(mask,:,:), ...
            policy.detector,P.featureBlocks);
        flagMatrix=scores>1;
        [bestScore,singleFlags]=max(scores,[],2);
        singleFlags(bestScore<=1)=0;
        dropPred=nan(sum(mask),3);
        for sensorIndex=1:3
            keep=policy.keepMasks{sensorIndex};
            Xdrop=[D.processX(mask,:),Xvariant(mask,keep),D.timeX(mask,:)];
            dropPred(:,sensorIndex)=predictRidge(policy.dropModels{sensorIndex},Xdrop);
        end
        singlePred=routePredictions(fullPred,dropPred,singleFlags);
        [XmedianSensor,maskMatrix]=applyMultiReplacement( ...
            Xvariant(mask,:),flagMatrix,P.featureBlocks,policy.sensorMedians);
        Xmedian=[D.processX(mask,:),XmedianSensor,D.timeX(mask,:)];
        medianPred=predictRidge(policy.fullModel,Xmedian);
        singleMaskPred=predictRidge(policy.dropoutModel,[Xmedian,maskMatrix]);
        multiMaskPred=predictRidge(policy.multiDropoutModel,[Xmedian,maskMatrix]);
        subsetCode=sum(flagMatrix.*[1 2 4],2);
        subsetPred=fullPred;
        for code=1:7
            rows=subsetCode==code;
            if any(rows)
                keep=policy.subsetKeepMasks{code};
                Xsubset=[D.processX(mask,:),Xvariant(mask,keep),D.timeX(mask,:)];
                candidate=predictRidge(policy.subsetModels{code},Xsubset);
                subsetPred(rows)=candidate(rows);
            end
        end
        keep=policy.subsetKeepMasks{targetCode};
        Xoracle=[D.processX(mask,:),Xvariant(mask,keep),D.timeX(mask,:)];
        oraclePred=predictRidge(policy.subsetModels{targetCode},Xoracle);
        fullPrediction(mask)=fullPred;
        medianPrediction(mask)=medianPred;
        singleMaskPrediction(mask)=singleMaskPred;
        multiMaskPrediction(mask)=multiMaskPred;
        singlePrediction(mask)=singlePred;
        subsetPrediction(mask)=subsetPred;
        oraclePrediction(mask)=oraclePred;
        flagMatrixAll(mask,:)=flagMatrix;
    end
end

function [Xreplaced,maskMatrix]=applyMultiReplacement( ...
        Xsensor,flagMatrix,featureBlocks,sensorMedians)
    Xreplaced=Xsensor; maskMatrix=double(flagMatrix);
    for sensorIndex=1:numel(featureBlocks)
        rows=flagMatrix(:,sensorIndex);
        if any(rows)
            Xreplaced(rows,featureBlocks{sensorIndex})=repmat( ...
                sensorMedians(featureBlocks{sensorIndex}),sum(rows),1);
        end
    end
end

function normalizedScore=qualityNormalizedScores( ...
        Xsensor,quality,detector,featureBlocks)
    n=size(Xsensor,1); nSensors=numel(featureBlocks); selectedStats=[1 2 3 4 7 8 9];
    normalizedScore=zeros(n,nSensors);
    for sensorIndex=1:nSensors
        rawQ=squeeze(quality(:,sensorIndex,:));
        if n==1, rawQ=reshape(rawQ,1,[]); end
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)),rawQ];
        score=max(abs((Q-detector.center{sensorIndex})./ ...
            detector.scale{sensorIndex}),[],2);
        normalizedScore(:,sensorIndex)=score./detector.threshold(sensorIndex);
    end
end
