using ConsoleApp1.Interface;
using Microsoft.ML;

namespace ConsoleApp1.MultiClassificationSimple;

public class FruitClassification : Itrainer<FruitData>
{
    private static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "fruits.csv");
    private DataOperationsCatalog.TrainTestData dataSet;
    private MLContext _context;
    public FruitClassification()
    {
        _context = new MLContext();
        var dataView = _context.Data.LoadFromTextFile<FruitData>(dataPath, hasHeader: true, separatorChar: ',');
        dataSet = _context.Data.TrainTestSplit(dataView, testFraction: 0.2);
    }

    public void Start()
    {
        var result = TrainData();
        Evaluate(result.Item1);
        Test(result.Item1, new FruitData() {Diameter = 4.3F, Height = 8.9F, Color = "Red"});
    }

    public (ITransformer, IDataView) TrainData()
    {
        var pipeline = _context.Transforms.Conversion.MapValueToKey("Label", "Type")
            .Append(_context.Transforms.Categorical.OneHotEncoding("Color"))
            .Append(_context.Transforms.Concatenate("NumericFeatures", "Diameter", "Height"))
            .Append(_context.Transforms.NormalizeMinMax("NumericFeatures"))
            .Append(_context.Transforms.Concatenate("Features", "NumericFeatures", "Color"))
            .Append(_context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(_context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        Console.WriteLine("======== Create Pipeline And Train Model =========");
        var model = pipeline.Fit(dataSet.TrainSet);
        Console.WriteLine("======== End Train ===============================");
        return (model, dataSet.TrainSet);
    }

    public void Evaluate(ITransformer model)
    {
        var predictions = model.Transform(dataSet.TestSet);
        var metrics = _context.MulticlassClassification.Evaluate(predictions);
        Console.WriteLine("=========================Metrics===================");
        Console.WriteLine($"*   MicroAccuracy : {metrics.MicroAccuracy:0.##}");
        Console.WriteLine($"*   MacroAccuray : {metrics.MacroAccuracy:0.##}");
        Console.WriteLine("=========================Metrics===================");
    }

    public void Test(ITransformer model, FruitData data)
    {
        var predictionFunction = _context.Model.CreatePredictionEngine<FruitData, FruitDataPrediction>(model);
        var prediction = predictionFunction.Predict(data);
        Console.WriteLine($"Type is {prediction.Type}");
    }
}