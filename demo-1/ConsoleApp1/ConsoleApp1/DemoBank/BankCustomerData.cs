using Microsoft.ML.Data;

namespace ConsoleApp1.DemoBank;

public class BankCustomerData
{
    [LoadColumn(0)]
    public float Age { get; set; }
    [LoadColumn(1)]
    public string Job { get; set; }
    [LoadColumn( 2)]
    public string Marital { get; set; }
    [LoadColumn(3)]
    public string Education { get; set; }
    [LoadColumn(4)]
    public string Default { get; set; }
    [LoadColumn(5)]
    public float Balance { get; set; }
    [LoadColumn(6)]
    public string Housing { get; set; }
    [LoadColumn(7)]
    public string Loan { get; set; }
    [LoadColumn(8)]
    public string Contact { get; set; }
    [LoadColumn(9)]
    public float Day { get; set; }
    [LoadColumn(10)]
    public string Month { get; set; }
    [LoadColumn(11)]
    public float Duration { get; set; }
    [LoadColumn(12)]
    public float Campaign { get; set; }
    [LoadColumn(13)]
    public float PDays { get; set; }
    [LoadColumn(14)]
    public float Previous { get; set; }
    [LoadColumn(15)]
    public string POutcome { get; set; }
    [LoadColumn(16)]
    public bool Subscribed { get; set; }
    
    [LoadColumn(17)]
    public float BankScore { get; set; }

    

    public bool HasMissingValues()
    {
        return string.IsNullOrEmpty(Job) || Age == 0;
    }
}