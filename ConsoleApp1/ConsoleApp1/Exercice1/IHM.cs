using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace ConsoleApp1.Exercice1;

public class IHM
{
    public void Start()
    {
        var context = new MLContext();
        var data = context.Data.LoadFromTextFile<HousingData>("./data/housing.csv", separatorChar: ',', hasHeader: true);
        var dataToProcess = context.Data.CreateEnumerable<HousingData>(data, reuseRowObject: false);
        var dataUpdated = dataToProcess.Distinct().Where(c => !float.TryParse(c.CRIM, out float result)).ToList();
        var finalData = context.Data.LoadFromEnumerable(dataUpdated);
        var preprocessingPipeline = 
            context.Transforms.Conversion.ConvertType("CRIM", "CRIM", DataKind.Double)
                .Append(context.Transforms.ReplaceMissingValues(new[] {
            new InputOutputColumnPair("CRIM"),
            new InputOutputColumnPair("ZN"),
            new InputOutputColumnPair("INDUS"),
            new InputOutputColumnPair("CHAS"),
            new InputOutputColumnPair("NOX"),
            new InputOutputColumnPair("RM"),
            new InputOutputColumnPair("AGE"),
            new InputOutputColumnPair("DIS"),
            new InputOutputColumnPair("RAD"),
            new InputOutputColumnPair("TAX"),
            new InputOutputColumnPair("PTRATIO"),
            new InputOutputColumnPair("B"),
            new InputOutputColumnPair("LSTAT"),
            new InputOutputColumnPair("MEDV")
        }, MissingValueReplacingEstimator.ReplacementMode.Mean)).Append(
                context.Transforms.Concatenate("Features",  "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT" ))
                .Append(context.Transforms.NormalizeMinMax("Features"));
    }
}