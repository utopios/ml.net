using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.Transforms.Text;

namespace ConsoleApp1.ClassificationText;


class TextInput
{
    public string Text { get; set; }
}

class SentimentData
{
    [LoadColumn(0)]
    public string Text { get; set; }
    
    [LoadColumn(1)]
    public float Sentiment { get; set; }
}

class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}

class TextFeatures
{
    public float[] Features { get; set; }
}
public class Demo
{
    private MLContext _context;

    public Demo()
    {
        _context = new MLContext();
    }

    public void Start()
    {
        var emptyData = _context.Data.LoadFromEnumerable(new List<TextInput>());

        var pipeline = _context.Transforms.Text.NormalizeText("Text", keepPunctuations: false, keepNumbers:true)
            .Append(_context.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
            .Append(_context.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D));
        var preview = pipeline.Preview;
        var model = pipeline.Fit(emptyData);

        var prediction = _context.Model.CreatePredictionEngine<TextInput, TextFeatures>(model);

        var resultChien = prediction.Predict(new TextInput() { Text = "Chien" });
        Console.WriteLine("====== Chien =====");
        foreach (var f in resultChien.Features)
        {
            Console.Write($"{f:F4}");
        }
        
        var resultChat = prediction.Predict(new TextInput() { Text = "Chat" });
        Console.WriteLine("====== Chat =====");
        foreach (var f in resultChat.Features)
        {
            Console.Write($"{f:F4}");
        }
    }

    public void Sentiment()
    {
        // var data = _context.Data.LoadFromTextFile<SentimentData>("./Data/data-s.csv", hasHeader: true,
        //     separatorChar: ',');
        //
        // var pipeline = _context.Transforms.Expression("Label", "(x) => x == 1 ? true : false", "Sentiment")
        //     .Append(_context.Transforms.Text.FeaturizeText("Features", "Text"))
        //     .Append(_context.BinaryClassification.Trainers.SdcaLogisticRegression());
        //
        //
        // // var pipeline2 = _context.Transforms.Conversion.MapValueToKey(
        // //         outputColumnName: "Label", 
        // //         inputColumnName: "Label")
        // //     .Append(_context.MulticlassClassification.Trainers.TextClassification(
        // //         labelColumnName: "Label",
        // //         sentence1ColumnName: "Sentence",
        // //         architecture: BertArchitecture.Roberta))
        // //     .Append(_context.Transforms.Conversion.MapKeyToValue(
        // //         outputColumnName: "PredictedLabel", 
        // //         inputColumnName: "PredictedLabel"));
        // var model = pipeline.Fit(data);
        //
        // var predictionEngine = _context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        //
        // var prediction = predictionEngine.Predict(new SentimentData() { Text = "I m not happy. " });
        //
        // Console.WriteLine($"Prediction {prediction.Prediction} avec proba : {prediction.Probability}");
        //
        // prediction = predictionEngine.Predict(new SentimentData() { Text = "I m very happy. " });
        //
        // Console.WriteLine($"Prediction {prediction.Prediction} avec proba : {prediction.Probability}");
        
        
        MLContext mlContext = new()
        {
            GpuDeviceId = 0,
            FallbackToCpu = true
        };
        Console.WriteLine("Loading data...");
        IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
            "./Data/data-tt.csv",
            separatorChar: '\t',
            hasHeader: false
        );
        IDataView dataViewTest = mlContext.Data.LoadFromTextFile<SentimentData>(
            "./Data/data-t.csv",
            separatorChar: '\t',
            hasHeader: false
        );
        var d = mlContext.Data.CreateEnumerable<SentimentData>(dataView, reuseRowObject: false).ToList().Where(r => float.TryParse(r.Sentiment.ToString(), out float f)).ToList();
        dataView = mlContext.Data.LoadFromEnumerable(d);
        DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        IDataView trainData = dataSplit.TrainSet;
        IDataView testData = dataSplit.TestSet;

// Create a pipeline for training the model
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "Label",
                inputColumnName: "Label")
            .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
                labelColumnName: "Label",
                sentence1ColumnName: "Sentence"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabel",
                inputColumnName: "PredictedLabel"));

        Console.WriteLine("Training model...");
        ITransformer model = pipeline.Fit(dataView);

        IDataView transformedTest = model.Transform(dataViewTest);
        MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

        Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
        Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");

        Console.WriteLine();
    }
}