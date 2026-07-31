clear; clc; close all;

paperRoot=fileparts(mfilename('fullpath'));
resultDir=fullfile(paperRoot,'additional_validation_results','reviewer_improvements');
if ~exist(resultDir,'dir'), mkdir(resultDir); end

filePath=fullfile(paperRoot,'phm2010_bundle_high_resolution.csv');
T=readtable(filePath,'VariableNamingRule','preserve');
tag=string(T.experiment_tag); timestamp=double(T.timestamp); wear=double(T.tool_wear);
tags=unique(tag,'stable');
thresholdMultipliers=[0.0001 0.0003 0.001 0.003 0.01 0.03 0.10 0.30 0.70 1.10];
referenceMultiplier=0.10;
expectedRuns=315;

sensitivityRows=cell(numel(tags)*numel(thresholdMultipliers),10);
summaryRows=cell(numel(tags),17);
gapRows=cell(sum(cellfun(@(x)sum(tag==x)-1,cellstr(tags))),5);
sensitivityRow=0; gapRow=0;

for tagIndex=1:numel(tags)
    rows=find(tag==tags(tagIndex));
    [sortedTimestamp,order]=sort(timestamp(rows));
    sortedRows=rows(order); sortedWear=wear(sortedRows);
    increments=diff(sortedTimestamp);
    positive=increments(increments>0);
    cadence=median(positive,'omitnan');
    referenceBoundary=increments<referenceMultiplier*cadence;
    referenceRuns=1+sum(referenceBoundary);
    monotonicInOriginalOrder=all(diff(timestamp(rows))>=0);
    sortedOrderMatchesFileOrder=isequal(sortedRows,rows);

    wearTolerance=max(1e-12,1e-10*max(1,max(abs(sortedWear),[],'omitnan')));
    wearBoundary=abs(diff(sortedWear))>wearTolerance;
    wearRuns=1+sum(wearBoundary);
    wearBoundaryDetected=safeDivide(sum(referenceBoundary & wearBoundary),sum(wearBoundary));
    timestampBoundaryWithWearChange=safeDivide( ...
        sum(referenceBoundary & wearBoundary),sum(referenceBoundary));
    repeatedWearAcrossTimestampBoundary=sum(referenceBoundary & ~wearBoundary);
    missedWearBoundaries=sum(~referenceBoundary & wearBoundary);
    boundaryIncrementMax=max(increments(referenceBoundary));
    withinIncrementMin=min(increments(~referenceBoundary));
    separationRatio=withinIncrementMin/max(boundaryIncrementMax,eps);

    summaryRows(tagIndex,:)={tags(tagIndex),numel(rows),cadence,referenceMultiplier, ...
        referenceRuns,expectedRuns,referenceRuns==expectedRuns,wearRuns, ...
        wearBoundaryDetected,timestampBoundaryWithWearChange, ...
        repeatedWearAcrossTimestampBoundary,missedWearBoundaries, ...
        boundaryIncrementMax,withinIncrementMin,separationRatio, ...
        monotonicInOriginalOrder,sortedOrderMatchesFileOrder};

    for multiplierIndex=1:numel(thresholdMultipliers)
        multiplier=thresholdMultipliers(multiplierIndex);
        candidateBoundary=increments<multiplier*cadence;
        runCount=1+sum(candidateBoundary);
        identical=isequal(candidateBoundary,referenceBoundary);
        sensitivityRow=sensitivityRow+1;
        sensitivityRows(sensitivityRow,:)={tags(tagIndex),multiplier, ...
            multiplier*cadence,runCount,expectedRuns,runCount-expectedRuns, ...
            identical,monotonicInOriginalOrder,sortedOrderMatchesFileOrder,cadence};
    end

    for incrementIndex=1:numel(increments)
        gapRow=gapRow+1;
        gapRows(gapRow,:)={tags(tagIndex),incrementIndex,increments(incrementIndex), ...
            increments(incrementIndex)/cadence,referenceBoundary(incrementIndex)};
    end
end

sensitivityTable=cell2table(sensitivityRows,'VariableNames', ...
    {'Cutter','ThresholdMultiplier','AbsoluteThresholdSeconds','DetectedRunCount', ...
    'ExpectedRunCount','RunCountDifference','IdenticalToReferenceGrouping', ...
    'TimestampMonotonicInFileOrder','SortedOrderMatchesFileOrder','TypicalCadenceSeconds'});
sensitivityTable.Cutter=string(sensitivityTable.Cutter);
sensitivityTable.IdenticalToReferenceGrouping=logical(sensitivityTable.IdenticalToReferenceGrouping);
sensitivityTable.TimestampMonotonicInFileOrder=logical(sensitivityTable.TimestampMonotonicInFileOrder);
sensitivityTable.SortedOrderMatchesFileOrder=logical(sensitivityTable.SortedOrderMatchesFileOrder);

summaryTable=cell2table(summaryRows,'VariableNames', ...
    {'Cutter','RawRowN','TypicalCadenceSeconds','ReferenceMultiplier', ...
    'TimestampRunCount','ExpectedRunCount','MatchesExpected315', ...
    'WearChangeRunCount','WearBoundaryRecallByTimestamp', ...
    'TimestampBoundaryPrecisionAgainstWearChange', ...
    'TimestampBoundariesWithRepeatedWear','WearChangesMissedByTimestamp', ...
    'MaximumBoundaryIncrementSeconds','MinimumWithinCutIncrementSeconds', ...
    'CadenceSeparationRatio','TimestampMonotonicInFileOrder', ...
    'SortedOrderMatchesFileOrder'});
summaryTable.Cutter=string(summaryTable.Cutter);
summaryTable.MatchesExpected315=logical(summaryTable.MatchesExpected315);
summaryTable.TimestampMonotonicInFileOrder=logical( ...
    summaryTable.TimestampMonotonicInFileOrder);
summaryTable.SortedOrderMatchesFileOrder=logical( ...
    summaryTable.SortedOrderMatchesFileOrder);

gapTable=cell2table(gapRows(1:gapRow,:),'VariableNames', ...
    {'Cutter','GapIndex','IncrementSeconds','IncrementToCadenceRatio','IsBoundary'});
gapTable.Cutter=string(gapTable.Cutter);
gapTable.IsBoundary=logical(gapTable.IsBoundary);

%% Stable threshold interval inferred without labels
stableRows=sensitivityTable(sensitivityTable.IdenticalToReferenceGrouping,:);
stableMinimum=min(stableRows.ThresholdMultiplier);
stableMaximum=max(stableRows.ThresholdMultiplier);
auditConclusion=table(all(summaryTable.MatchesExpected315), ...
    all(summaryTable.WearBoundaryRecallByTimestamp==1), ...
    all(summaryTable.TimestampMonotonicInFileOrder), ...
    all(summaryTable.SortedOrderMatchesFileOrder),stableMinimum,stableMaximum, ...
    min(summaryTable.CadenceSeparationRatio), ...
    sum(summaryTable.TimestampBoundariesWithRepeatedWear), ...
    'VariableNames',{'AllCuttersMatch315','AllWearChangesAtTimestampBoundaries', ...
    'TimestampMonotonicInFileOrder','SortingPreservesFileOrder', ...
    'StableMultiplierMinimum','StableMultiplierMaximum', ...
    'MinimumCadenceSeparationRatio','RepeatedWearTimestampBoundaryN'});

writetable(sensitivityTable,fullfile(resultDir,'phm_segmentation_sensitivity.csv'));
writetable(summaryTable,fullfile(resultDir,'phm_segmentation_cutter_audit.csv'));
writetable(gapTable,fullfile(resultDir,'phm_timestamp_gap_source_data.csv'));
writetable(auditConclusion,fullfile(resultDir,'phm_segmentation_conclusion.csv'));

%% Figure
figure('Color','w','Position',[60 60 1500 960]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile;
for tagIndex=1:numel(tags)
    mask=sensitivityTable.Cutter==tags(tagIndex);
    semilogx(sensitivityTable.ThresholdMultiplier(mask), ...
        sensitivityTable.DetectedRunCount(mask),'-o','LineWidth',2); hold on;
end
yline(expectedRuns,'--','Expected 315 cuts','LineWidth',1.5);
xline(referenceMultiplier,':','Reference 0.1','LineWidth',1.5);
set(gca,'YScale','log'); ylim([1 1e5]);
xlabel('Boundary threshold / typical cadence'); ylabel('Detected cuts');
legend(tags,'Location','northwest'); grid on;
title('(a) Segmentation threshold sensitivity');

nexttile;
boundary=gapTable.IsBoundary;
histogram(log10(gapTable.IncrementToCadenceRatio(~boundary)),50, ...
    'FaceColor',[0.25 0.55 0.78]); hold on;
histogram(log10(gapTable.IncrementToCadenceRatio(boundary)),30, ...
    'FaceColor',[0.90 0.30 0.08]);
xline(log10(referenceMultiplier),'--','Reference threshold','LineWidth',1.5);
set(gca,'YScale','log'); ylim([1 1e6]);
xlabel('log_{10}(timestamp increment / cadence)'); ylabel('Gap count');
legend({'Within-cut gaps','Cut boundaries'},'Location','northwest'); grid on;
title('(b) Separation of cadence regimes');

nexttile;
bar(categorical(summaryTable.Cutter), ...
    [summaryTable.TimestampRunCount,summaryTable.WearChangeRunCount]);
ylabel('Detected sequential groups');
legend({'Timestamp-only cuts','Wear-change groups'},'Location','northwest'); grid on;
title('(c) Timestamp segmentation does not depend on label changes');

nexttile;
bar(categorical(summaryTable.Cutter),100*[ ...
    summaryTable.WearBoundaryRecallByTimestamp, ...
    summaryTable.TimestampBoundaryPrecisionAgainstWearChange]);
ylim([0 105]); ylabel('Boundary agreement (%)');
legend({'Wear-change recall','Timestamp boundaries with wear change'}, ...
    'Location','southwest'); grid on;
title('(d) Label audit performed only after segmentation');

sgtitle('PHM2010 Label-Free Timestamp Segmentation Audit','FontWeight','bold');
set(gcf,'ToolBar','none'); axesHandles=findall(gcf,'Type','axes');
for axisIndex=1:numel(axesHandles)
    if isprop(axesHandles(axisIndex),'Toolbar') && ~isempty(axesHandles(axisIndex).Toolbar)
        axesHandles(axisIndex).Toolbar.Visible='off';
    end
end
drawnow;
exportgraphics(gcf,fullfile(resultDir,'phm_segmentation_audit.png'),'Resolution',300);
exportgraphics(gcf,fullfile(resultDir,'phm_segmentation_audit.pdf'),'ContentType','vector');

%% Audit note
noteFile=fullfile(resultDir,'README_phm_segmentation_audit.txt');
fid=fopen(noteFile,'w'); assert(fid>0,'Could not create audit note.');
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'PHM2010 timestamp segmentation audit\n\n');
fprintf(fid,'Segmentation inputs: experiment_tag and timestamp only.\n');
fprintf(fid,'The target tool_wear field is never read when constructing groups.\n');
fprintf(fid,'Reference threshold: timestamp increment < 0.1 x cutter-specific median positive cadence.\n');
fprintf(fid,'Expected protocol count: 315 cuts per cutter (c1, c4, c6).\n');
fprintf(fid,'Official challenge page: https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/\n');
fprintf(fid,'The wear field is used only after grouping to audit boundary correspondence.\n');
fprintf(fid,'Stable multiplier interval producing an identical grouping: %.4g to %.4g.\n', ...
    stableMinimum,stableMaximum);
fprintf(fid,'All three cutters match 315 cuts: %d.\n',all(summaryTable.MatchesExpected315));
fprintf(fid,'All wear changes occur at a timestamp boundary: %d.\n', ...
    all(summaryTable.WearBoundaryRecallByTimestamp==1));
clear cleanup;

save(fullfile(resultDir,'phm_segmentation_audit.mat'), ...
    'sensitivityTable','summaryTable','gapTable','auditConclusion');

fprintf('\n========== PHM segmentation audit ==========\n'); disp(summaryTable);
fprintf('\n========== Audit conclusion ==========\n'); disp(auditConclusion);
fprintf('\nSaved to:\n%s\n',resultDir);

function value=safeDivide(numerator,denominator)
    if denominator==0, value=NaN; else, value=numerator/denominator; end
end
