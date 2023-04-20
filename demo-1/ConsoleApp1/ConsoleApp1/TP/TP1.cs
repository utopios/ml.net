using Microsoft.ML;
using Microsoft.ML.Data;

namespace ConsoleApp1.TP;

public class BuildingData
{
    [LoadColumn(0)] public float Surface;
    [LoadColumn(1)] public string Type;
    [LoadColumn(2)] public float Year;
    [LoadColumn(3)] public float Temperature;
    [LoadColumn(4)] public float Humidity;
    [LoadColumn(5)] public float EnergyConsumption;
}

public class EnergyPrediction
{
    [ColumnName("Score")]
    public float EnergyConsumption;
}


public class TP1
{
    public void Start()
    {
        var context = new MLContext();
        var data = context.Data.LoadFromTextFile<BuildingData>("./Data/energy-data.csv", separatorChar: ',');

        var tt = context.Data.TrainTestSplit(data);
        var pipeline =
            context.Transforms.CopyColumns("Label", "EnergyConsumption")
                .Append(context.Transforms.Categorical.OneHotEncoding("TypeEncoded", "Type"))
                .Append(
                    context.Transforms.Concatenate("Features", "Surface", "TypeEncoded", "Year", "Temperature", "Humidity"))
                .Append(context.Transforms.NormalizeMinMax("Features"))
                //.Append(context.Transforms.CopyColumns("Feature", "Features"))
                .Append(context.Regression.Trainers.FastTree());
                

        var model = pipeline.Fit(data);
        var estimators = context.Regression.Trainers.FastTree();
        var results = context.Regression.CrossValidate(model.Transform(data), pipeline, 5);
        var metrics = results.OrderBy(f => f.Metrics.RSquared).Select(f => f.Metrics).ToList()[0];
        var topModel = results.OrderBy(f => f.Metrics.RSquared).Select(f => f.Model).ToList()[0];
        //var predictions = topModel.Transform(tt.TestSet);
        //var metrics = context.Regression.Evaluate(predictions, "Label", "Score");
        Console.WriteLine($"R-squared: {metrics.RSquared}");
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
        var predictionEngine = context.Model.CreatePredictionEngine<BuildingData, EnergyPrediction>(topModel);
        var newBuilding = new BuildingData
        {
            Surface = 150,
            Type = "Commercial",
            Year = 2010,
            Temperature = 20,
            Humidity = 60
        };

        var predictedEnergyConsumption = predictionEngine.Predict(newBuilding);
        Console.WriteLine($"Predicted energy consumption: {predictedEnergyConsumption.EnergyConsumption} kWh");


    }
}