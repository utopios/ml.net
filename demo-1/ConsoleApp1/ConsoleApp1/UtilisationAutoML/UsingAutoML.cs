using ConsoleApp1.MultiClassificationSimple;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace ConsoleApp1.UtilisationAutoML;

public class UsingAutoML
{
    private MLContext _context;
    public void Start()
    {
        _context = new MLContext();
        var data = _context.Data.LoadFromTextFile<FruitData>("./Data/fruits.csv", hasHeader: true, separatorChar: ',');
        var tt = _context.Data.TrainTestSplit(data);
        var expriment = _context.Auto().CreateMulticlassClassificationExperiment(new MulticlassExperimentSettings());
        var result = expriment.Execute(tt.TrainSet);

        var bestModel = result.BestRun.Model;

        var predictions = bestModel.Transform(tt.TestSet);

        var meterics = _context.MulticlassClassification.Evaluate(predictions);
        
        Console.WriteLine(meterics.MicroAccuracy);
    } 
}