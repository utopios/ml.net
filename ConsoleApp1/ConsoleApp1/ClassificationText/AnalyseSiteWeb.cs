using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.TorchSharp;
using System.Text.RegularExpressions;
using DataAccess;
using static TorchSharp.torch.utils;

namespace ConsoleApp1.ClassificationText
{
    class InputData
    {
        [LoadColumn(0)]
        [ColumnName(@"Sentence")]
        public string Text { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"Label")]
        public string Sentiment { get; set; }
    }

    class OutPutPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }

        public float Probability { get; set; }

        //public float Score { get; set; }
    }

    class AnalyseSiteWeb
    {
        private MLContext mlContext;
        public void Start()
        {
            //Transformation des fichiers
            //Train
            PrepareFile(Path.Combine(Environment.CurrentDirectory, "data", "lot-data.csv"), Path.Combine(Environment.CurrentDirectory, "data", "final-data.csv"));
            //PrepareFile(Path.Combine(Environment.CurrentDirectory, "data", "origin-test.csv"), Path.Combine(Environment.CurrentDirectory, "data", "test.csv"));
            //var model =  Train(Path.Combine(Environment.CurrentDirectory, "data", "final-data.csv"), ';');
            var model = LoadModel("model.zip");
            Evaluate("day of the tax authorities request", model);
            Evaluate("30 days from the request of tax authorities", model);
            Evaluate("Documentation for a relevant tax period must be in place before the deadline of income tax declaration", model);
        }

        public string Scrap(string path)
        {
            return "";
        }

        public void PrepareFile(string originPath, string destinationPath)
        {
            var dt = DataAccess.DataTable.New.ReadCsv(originPath);
            StreamWriter writer = new StreamWriter(destinationPath);
            foreach (Row r in dt.Rows)
            {
                List<string> line = new List<string>();
                foreach (var str in r.Values)
                {
                    line.Add(Regex.Replace(str, @"(\t+|\n+|\r+)", ""));
                }
                writer.WriteLine(string.Join(';', line));
            }
            writer.Close();

        }

        public void Evaluate(string text, ITransformer model)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutPrediction>(model);
            var prediction = predictionEngine.Predict(new InputData { Text = text});
            Console.WriteLine(prediction.Prediction);
        }
        public ITransformer Train(string path, char separator)
        {
            mlContext = new()
            {
                GpuDeviceId = 0,
                FallbackToCpu = true
            };
            Console.WriteLine("Loading data...");
            IDataView dataView = mlContext.Data.LoadFromTextFile<InputData>(
            path,
                separatorChar: separator,
                hasHeader: true
            );
           
            var d = mlContext.Data.CreateEnumerable<InputData>(dataView, reuseRowObject: false).ToList().Where(r => float.TryParse(r.Sentiment.ToString(), out float f)).ToList();
            dataView = mlContext.Data.LoadFromEnumerable(d);
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            // Create a pipeline for training the model
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                                        outputColumnName: "Label",
                                        inputColumnName: "Label")
                                    .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
                                        labelColumnName: "Label",
                                        sentence1ColumnName: "Sentence"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                                        outputColumnName: "PredictedLabel",
                                        inputColumnName: "PredictedLabel"));

            Console.WriteLine("Training model...");
            ITransformer model = pipeline.Fit(dataView);

            IDataView transformedTest = model.Transform(testData);
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");

            Console.WriteLine();
            mlContext.Model.Save(model, dataView.Schema, "model.zip");
            //SaveOnnx("./custom.onnx", dataView, mlContext, model);
            return model;
        }

        public ITransformer LoadModel(string path)
        {
            DataViewSchema modelSchema;
            mlContext = new()
            {
                GpuDeviceId = 0,
                FallbackToCpu = true
            };
            // Load trained model
            ITransformer trainedModel = mlContext.Model.Load(path, out modelSchema);
            return trainedModel;
        }

        public void SaveOnnx(string path, IDataView data, MLContext mlContext, ITransformer model)
        {
            
            FileStream stream = File.Create(path);
            mlContext.Model.ConvertToOnnx(model, data, stream);
            stream.Close();
        }

    }
}
