using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms;
using System;

namespace SentimentAnalysisML
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Text { get; set; } = String.Empty;

        [LoadColumn(1)]
        public string Sentiment { get; set; } = String.Empty;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; } = String.Empty;
    }

    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
                path: "Data/imdb_reviews.csv",
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true,
                allowSparse: false,
                trimWhitespace: true
            );

            dataView = mlContext.Data.Cache(dataView);

            Console.WriteLine("Data Loaded Successfully!");

            var dataPipeline = mlContext.Transforms.Text.NormalizeText("CleanedText", nameof(SentimentData.Text),
                caseMode: TextNormalizingEstimator.CaseMode.Lower,
                keepNumbers: false, keepPunctuations: false, keepDiacritics: false)
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("TokenizedText", "CleanedText"))
                .Append(mlContext.Transforms.Text.RemoveStopWords("TokenizedText"))
                .Append(mlContext.Transforms.Text.FeaturizeText("Features", "TokenizedText"))

                .Append(mlContext.Transforms.Text.FeaturizeText("Features", "CleanedText"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(SentimentData.Sentiment)))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext);


            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainingPipeline = dataPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(dataView);

            Console.WriteLine("Model Trained Successfully!");

            var predictions = trainedModel.Transform(dataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score", "PredictedLabel");

            Console.WriteLine($"Accuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");


            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(trainedModel);

            while (true)
            {
                Console.Write("Enter text to analyze sentiment (or type 'exit' to quit): ");
                var inputText = Console.ReadLine();

                if (inputText?.ToLower() == "exit")
                    break;

                var newSample = new SentimentData { Text = inputText };
                var prediction = predictionEngine.Predict(newSample);

                Console.WriteLine($"\nText: {inputText}\nPredicted Sentiment: {prediction.Prediction}\n");
            }

        }
    }


}