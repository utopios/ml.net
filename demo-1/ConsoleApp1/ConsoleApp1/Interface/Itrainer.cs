using Microsoft.ML;

namespace ConsoleApp1.Interface;

public interface Itrainer<T>
{
    (ITransformer, IDataView) TrainData();
    void Evaluate(ITransformer model);
    void Test(ITransformer model, T data);
}