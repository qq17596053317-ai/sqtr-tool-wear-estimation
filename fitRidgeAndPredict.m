function yPrediction = ...
    fitRidgeAndPredict( ...
    XTrain,yTrain,XTest,lambda)

    %% 训练集标准化参数
    featureMean = mean(XTrain,1);
    featureStd = std(XTrain,[],1);

    %% 删除训练集中没有变化的特征
    keepFeature = ...
        isfinite(featureStd) & ...
        featureStd > 1e-12;

    XTrain = XTrain(:,keepFeature);
    XTest = XTest(:,keepFeature);

    featureMean = ...
        featureMean(keepFeature);

    featureStd = ...
        featureStd(keepFeature);

    %% 仅使用训练集参数进行标准化
    XTrainZ = ...
        (XTrain-featureMean)./featureStd;

    XTestZ = ...
        (XTest-featureMean)./featureStd;

    %% 中心化响应变量
    responseMean = mean(yTrain);
    yCentered = yTrain-responseMean;

    nFeatures = size(XTrainZ,2);

    %% 岭回归闭式解
    beta = ( ...
        XTrainZ'*XTrainZ + ...
        lambda*eye(nFeatures)) \ ...
        (XTrainZ'*yCentered);

    %% 测试集预测
    yPrediction = ...
        responseMean+XTestZ*beta;
end