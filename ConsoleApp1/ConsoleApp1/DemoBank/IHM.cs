using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace ConsoleApp1.DemoBank;

public class IHM
{
    private MLContext _context;
    private string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "bank.csv");
    private string _dataPathTrain = Path.Combine(Environment.CurrentDirectory, "Data", "bank_2.csv");
    private string _dataPathTest = Path.Combine(Environment.CurrentDirectory, "Data", "bank_2_test.csv");
    public IHM()
    {
        _context = new MLContext();
    }

    public void Start()
    {
        //Chargement des données
        var data = _context.Data.LoadFromTextFile<BankCustomerData>(_dataPath, separatorChar:',', hasHeader:true);
        var result = data.Preview().RowView;
        var preprocessingPipeline = _context.Transforms.Conversion.MapValueToKey("Label", "Subscribed")
            .Append(_context.Transforms.Categorical.OneHotEncoding("JobEncoded", "Job"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("MaritalEncoded", "Marital"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("EducationEncoded", "Education"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("DefaultEncoded", "Default"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("HousingEncoded", "Housing"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("LoanEncoded", "Loan"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("ContactEncoded", "Contact"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("MonthEncoded", "Month"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("POutcomeEncoded", "POutcome"))
            .Append(_context.Transforms.ReplaceMissingValues("Balance", "Balance",
                MissingValueReplacingEstimator.ReplacementMode.Mean))
            .Append(_context.Transforms.Concatenate("NumericFeatures", "Age", "Balance", "Day", "PDays",
                "Duration", "Campaign", "Previous"))
            .Append(_context.Transforms.NormalizeMinMax("NumericFeatures", "NumericFeatures"))
            .Append(_context.Transforms.Concatenate("Features", "JobEncoded", "MaritalEncoded", "EducationEncoded",
                "HousingEncoded", "LoanEncoded", "ContactEncoded", "MonthEncoded", "POutcomeEncoded",
                "NumericFeatures"))
            .Append(_context.Transforms.Conversion.MapKeyToValue("Label"))
            .Append(_context.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

        var tt = _context.Data.TrainTestSplit(data, testFraction: 0.2);
        //Train
        var model = preprocessingPipeline.Fit(tt.TrainSet);
        
        //Prédictions sur les données test
        var predictions = model.Transform(tt.TestSet);

        var metric = _context.BinaryClassification.Evaluate(predictions);
        
        Console.WriteLine($"Accuracy: {metric.Accuracy}");

    }
    
    public void StartPCA()
    {
        //Chargement des données
        var dataTrain = _context.Data.LoadFromTextFile<BankCustomerData>(_dataPathTrain, separatorChar:',', hasHeader:true);
        var dataTest = _context.Data.LoadFromTextFile<BankCustomerData>(_dataPathTest, separatorChar:',', hasHeader:true);
        
        var preprocessingPipeline = _context.Transforms.Conversion.MapValueToKey("Label", "Subscribed")
            .Append(_context.Transforms.Categorical.OneHotEncoding("JobEncoded", "Job"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("MaritalEncoded", "Marital"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("EducationEncoded", "Education"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("DefaultEncoded", "Default"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("HousingEncoded", "Housing"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("LoanEncoded", "Loan"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("ContactEncoded", "Contact"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("MonthEncoded", "Month"))
            .Append(_context.Transforms.Categorical.OneHotEncoding("POutcomeEncoded", "POutcome"))
            .Append(_context.Transforms.ReplaceMissingValues("Balance", "Balance",
                MissingValueReplacingEstimator.ReplacementMode.Mean))
            .Append(_context.Transforms.Concatenate("NumericFeatures", "Age", "Balance", "Day", "PDays",
                "Duration", "Campaign", "Previous", "BankScore"))
            .Append(_context.Transforms.NormalizeMinMax("NumericFeatures", "NumericFeatures"))
            .Append(_context.Transforms.ProjectToPrincipalComponents("PcaFeatures", "NumericFeatures", rank:4))
            .Append(_context.Transforms.Concatenate("Features", "JobEncoded", "MaritalEncoded", "EducationEncoded",
                "HousingEncoded", "LoanEncoded", "ContactEncoded", "MonthEncoded", "POutcomeEncoded",
                "PcaFeatures"))
            .Append(_context.Transforms.Conversion.MapKeyToValue("Label"))
            .Append(_context.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

        //var tt = _context.Data.TrainTestSplit(data, testFraction: 0.2);
        //Train
        var model = preprocessingPipeline.Fit(dataTrain);
        
        //Prédictions sur les données test
        var predictions = model.Transform(dataTest);

        var metric = _context.BinaryClassification.Evaluate(predictions);
        
        Console.WriteLine($"Accuracy: {metric.Accuracy}");

    }
}