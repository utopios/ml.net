// See https://aka.ms/new-console-template for more information

using ConsoleApp1.ClassificationText;
using ConsoleApp1.DemoBank;
using ConsoleApp1.MultiClassificationSimple;
using ConsoleApp1.regressionsimple;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;

// Console.WriteLine("Hello, World!");
// new IHM().StartPCA();
//new HouseRegression().Start();
//new FruitClassification().Start();

//Embiddings
//new Demo().Start();

//Sentiment
//new Demo().Sentiment();
//MLContext mlContext = new()
//{
//    GpuDeviceId = 0,
//    FallbackToCpu = true
//};
//Console.WriteLine("Loading data...");
//IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
//"./Data/data-tt.csv",
//    separatorChar: '\t',
//    hasHeader: false
//);
//IDataView dataViewTest = mlContext.Data.LoadFromTextFile<SentimentData>(
//"./Data/data-t.csv",
//    separatorChar: '\t',
//    hasHeader: false
//);
//var d = mlContext.Data.CreateEnumerable<SentimentData>(dataView, reuseRowObject: false).ToList().Where(r => float.TryParse(r.Sentiment.ToString(), out float f)).ToList();
//dataView = mlContext.Data.LoadFromEnumerable(d);
//DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
//IDataView trainData = dataSplit.TrainSet;
//IDataView testData = dataSplit.TestSet;

//// Create a pipeline for training the model
//var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
//                            outputColumnName: "Label",
//                            inputColumnName: "Label")
//                        .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
//                            labelColumnName: "Label",
//                            sentence1ColumnName: "Sentence"))
//                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
//                            outputColumnName: "PredictedLabel",
//                            inputColumnName: "PredictedLabel"));

//Console.WriteLine("Training model...");
//ITransformer model = pipeline.Fit(dataView);

//IDataView transformedTest = model.Transform(dataViewTest);
//MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

//Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
//Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");

//Console.WriteLine();

//new MLScrapper().MLScrapperLaunch();
new AnalyseSiteWeb().Start();
