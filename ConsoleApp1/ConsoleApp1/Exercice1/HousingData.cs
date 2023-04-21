using Microsoft.ML.Data;

namespace ConsoleApp1.Exercice1;

public class HousingData
{
    [LoadColumn(0)]
    public string CRIM { get; set; }
    [LoadColumn(1)]
    public float ZN { get; set; }
    [LoadColumn(2)]
    public float INDUS { get; set; }
    [LoadColumn(3)]
    public float CHAS { get; set; }
    [LoadColumn(4)]
    public float NOX { get; set; }
    [LoadColumn(5)]
    public float RM { get; set; }
    [LoadColumn(6)]
    public float AGE { get; set; }
    [LoadColumn(7)]
    public float DIS { get; set; }
    [LoadColumn(8)]
    public float RAD { get; set; }
    [LoadColumn(9)]
    public float TAX { get; set; }
    [LoadColumn(10)]
    public float PTRATIO { get; set; }
    [LoadColumn(11)]
    public float B { get; set; }
    [LoadColumn(12)]
    public float LSTAT { get; set; }
    [LoadColumn(13)]
    public float MEDV { get; set; }
}