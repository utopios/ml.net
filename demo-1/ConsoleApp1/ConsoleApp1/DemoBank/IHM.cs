using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace ConsoleApp1.DemoBank;

public class IHM
{
    private MLContext _context;
    private string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "bank");

    public IHM()
    {
        _context = new MLContext();
    }

    public void Start()
    {
        //Chargement des données
        var data = _context.Data.LoadFromTextFile<BankCustomerData>(_dataPath);
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
            .Append(_context.Transforms.ReplaceMissingValues("balance", "balance", MissingValueReplacingEstimator.ReplacementMode.Mean));
    }
}