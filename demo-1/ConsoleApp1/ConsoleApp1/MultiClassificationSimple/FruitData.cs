using Microsoft.ML.Data;

namespace ConsoleApp1.MultiClassificationSimple;

public class FruitData
{
    [LoadColumn(3)]
    public string Type;

    [LoadColumn(0)]
    public float Diameter;

    [LoadColumn(1)]
    public float Height;

    [LoadColumn(2)]
    public string Color;
}

public class FruitDataPrediction
{
    [ColumnName("PredictedLabel")]
    public string Type;
}