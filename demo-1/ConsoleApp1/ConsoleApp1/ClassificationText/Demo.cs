using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
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
        var data = _context.Data.LoadFromTextFile<SentimentData>("./Data/data-s.csv", hasHeader: true,
            separatorChar: ',');

        var pipeline = _context.Transforms.Expression("Label", "(x) => x == 1 ? True : False", "Sentiment")
            .Append(_context.Transforms.Text.FeaturizeText("Features", "Text"))
            .Append(_context.MulticlassClassification.Trainers.TextClassification());

        var model = pipeline.Fit(data);

        var predictionEngine = _context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        var prediction = predictionEngine.Predict(new SentimentData() { Text = "I m not fine. " });
        
        Console.WriteLine($"Prediction {prediction.Prediction} avec proba : {prediction.Probability}");

        prediction = predictionEngine.Predict(new SentimentData() { Text = "I m very happy. " });
        
        Console.WriteLine($"Prediction {prediction.Prediction} avec proba : {prediction.Probability}");
    }
}