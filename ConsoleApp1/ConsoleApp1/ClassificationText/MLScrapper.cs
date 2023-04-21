using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML.TorchSharp;

namespace ConsoleApp1.ClassificationText
{
    public class ModelInput
    {
        [LoadColumn(0), ColumnName("Sentence")] public string Fillingdate;

        [LoadColumn(3), ColumnName("Label")] public string IsDeadLine;
    }

    public class NotificationPrediction : ModelInput
    {
        public string TestedDescription { get; set; }
        [ColumnName("PredictedLabel")] public float IsDeadLinePrediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }

    }

    class MLScrapper
    {
        private MLContext context;
        private string modifiedFilePath;
        public void MLScrapperLaunch()
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

            /* ---------------------- SCRAPPING PART  ----------------------*//*

            // Define the URL to scrape
            var url = "https://www.ird.gov.hk/eng/ppr/22031801.htm";

            // Create a HttpClient to fetch the web page
            var client = new HttpClient();
            var response = client.GetAsync(url).Result;
            var content = response.Content.ReadAsStringAsync().Result;

            // Load the HTML content into an HtmlDocument
            var doc = new HtmlDocument();
            doc.LoadHtml(content);

            // Find the relevant HTML element and extract the text
            var element = doc.DocumentNode.Descendants("div").Where(d => d.GetAttributeValue("class", "") == "contentstyle").FirstOrDefault();
            var text = element.InnerText;

            // Load the data
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            writer.Write($"{text},false\n");
            writer.Flush();
            stream.Position = 0;*/

            /* ---------------------- ML PART  ----------------------*/

            // Define the ML context
            context = new MLContext();

            // Load the text file as a stream
            //var streamReader = new StreamReader(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Dataset.csv"));

            //// Read the stream as a string
            //var FileContent = streamReader.ReadToEnd();

            //// Replace new line characters within a field with a semicolon separator
            //var regex = new Regex(@"""(?<field>[^""\n]*(\n[^""\n]*)*)""");
            //FileContent = regex.Replace(FileContent, m => m.Groups["field"].Value.Replace("\n", " "));

            // Writting modified text to a new csv file

            modifiedFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "data-tt.csv");
            //File.WriteAllText(modifiedFilePath, FileContent);


            var splitDataView = LoadData(context);
            /*            var previewData = splitDataView.TrainSet.Preview().RowView;*/
            ITransformer model = BuildAndTrainModel(context, splitDataView.Item1, splitDataView.Item2);
            useModelWithSingleItem(context, model);
            /*        evaluate(context, model, splitDataView.TrainSet);*/

        }
        (IDataView, IDataView) LoadData(MLContext mlcontext)
        {
            //Define data source
            IDataView trainingDataView = context.Data.LoadFromTextFile<ModelInput>(
                                         modifiedFilePath,
                                         hasHeader: false,
                                         separatorChar: '\t'
                                         /*                                     allowQuoting: true,
                                                                              allowSparse: false*/
                                         );

            var d = context.Data.CreateEnumerable<ModelInput>(trainingDataView, reuseRowObject: false).ToList().Where(r => float.TryParse(r.IsDeadLine.ToString(), out float f)).ToList();
            trainingDataView = context.Data.LoadFromEnumerable(d);

            IDataView testDataView = context.Data.LoadFromTextFile<ModelInput>(
                                       Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "data-t.csv"),
                                       hasHeader: false,
                                       separatorChar: '\t'
                                       /*                                     allowQuoting: true,
                                                                            allowSparse: false*/
                                       );

            d = context.Data.CreateEnumerable<ModelInput>(trainingDataView, reuseRowObject: false).ToList().Where(r => float.TryParse(r.IsDeadLine.ToString(), out float f)).ToList();
            testDataView = context.Data.LoadFromEnumerable(d);


            //Train data
            /*            TrainTestData splitDataTraining = context.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);*/
            return (trainingDataView, testDataView);
        }




        ITransformer BuildAndTrainModel(MLContext mlcontext, IDataView trainDataView, IDataView testDataView)
        {
            // Define the pipeline

            var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Label")
                                                         .Append(context.MulticlassClassification.Trainers.TextClassification(labelColumnName: "Label", sentence1ColumnName: "Sentence", architecture: BertArchitecture.Roberta))
                                                         .Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));



            // Train the model

            var modelTemplate = pipeline.Fit(trainDataView);
            var prediction = modelTemplate.Transform(testDataView);
            var preview = prediction.Preview();
            var metrics = context.MulticlassClassification.Evaluate(prediction);
            Console.WriteLine($"Accuracy is : {metrics.MacroAccuracy}");
            return modelTemplate;

        }

        void useModelWithSingleItem(MLContext contextTest, ITransformer modelTest)
        {
            PredictionEngine<ModelInput, NotificationPrediction> predictionFunction = contextTest.Model.CreatePredictionEngine<ModelInput, NotificationPrediction>(modelTest);
            ModelInput FillingdateSample = new ModelInput { Fillingdate = "Documentation for a relevant tax period must be in place before the deadline of income tax declaration" };
            NotificationPrediction resultPrediction = predictionFunction.Predict(FillingdateSample);

            Console.WriteLine($"Prediction of {resultPrediction.Fillingdate} is {resultPrediction.IsDeadLinePrediction} with probability {resultPrediction.Probability} and score {resultPrediction.Score}");
        }
    }
}
