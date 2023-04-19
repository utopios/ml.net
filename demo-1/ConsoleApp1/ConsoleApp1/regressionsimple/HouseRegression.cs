using Microsoft.ML;

namespace ConsoleApp1.regressionsimple;

public class HouseRegression
{
    private static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "houses.csv");
    private static readonly string dataPathTest = Path.Combine(Environment.CurrentDirectory, "Data", "houses-test.csv");
    private MLContext _context;

    public HouseRegression()
    {
        _context = new MLContext();
    }

    public void Start()
    {
        var result = TrainData();
        Evaluate(result.Item1);
        Test(result.Item1, new HouseData(){Size = 2500, Bedrooms = 5});
    }

    public (ITransformer, IDataView) TrainData()
    {
        var dataView = _context.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');
        var pipeline = _context.Transforms.CopyColumns("Label", "Price")
            .Append(_context.Transforms.Concatenate("Features", "Size", "Bedrooms"))
            .Append(_context.Transforms.NormalizeMinMax("Features"))
            .Append(_context.Regression.Trainers.FastTree());
        
        Console.WriteLine("======== Create Pipeline And Train Model =========");
        var model = pipeline.Fit(dataView);
        Console.WriteLine("======== End Train ===============================");
        return (model, dataView);
    }

    public void Evaluate(ITransformer model)
    {
        var dataViewTest = _context.Data.LoadFromTextFile<HouseData>(dataPathTest, hasHeader: true, separatorChar: ',');
        var predictions = model.Transform(dataViewTest);
        var metrics = _context.Regression.Evaluate(predictions, "Label", "Score");
        Console.WriteLine("=========================Metrics===================");
        Console.WriteLine($"*   Rsquared Score : {metrics.RSquared:0.##}");
        Console.WriteLine($"*   Root Mean Squared Error : {metrics.RootMeanSquaredError:#.##}");
        Console.WriteLine("=========================Metrics===================");
    }

    public void Test(ITransformer model, HouseData houseData)
    {
        var predictionFunction = _context.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
        var prediction = predictionFunction.Predict(houseData);
        Console.WriteLine($"Price is {prediction.Price:#.##}");
    }
}